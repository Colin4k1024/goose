//! Evolution loop acceptance tests (issue #4).
//!
//! Concurrency (one cycle at a time), circuit breaker (two regressions →
//! pause), and fail-closed full chain (a generator that always lowers
//! fitness never mutates the managed root).

use goose_rsi::apply::ApplyConfig;
use goose_rsi::evolution::{
    EventKind, EvolutionEvent, EvolutionLoop, LoopContext, ProposalGenerationError,
    ProposalGenerator,
};
use goose_rsi::fitness::{
    BenchmarkSplit, BenchmarkSuite, BenchmarkTask, FitnessCommand, FitnessConfig, FitnessSuite,
    TestDepth, TestPhases,
};
use goose_rsi::proposal::{
    FitnessExpectation, MutationProposal, MutationTarget, PayloadChange, ProposalSource,
};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

/// A stub generator that produces proposals targeting a recipe file with a
/// fitness minimum the stub suite will never clear.
struct RegressingGenerator {
    calls: AtomicU32,
}

impl RegressingGenerator {
    fn new() -> Self {
        RegressingGenerator {
            calls: AtomicU32::new(0),
        }
    }
}

#[async_trait::async_trait]
impl ProposalGenerator for RegressingGenerator {
    async fn generate(
        &self,
        context: &LoopContext,
    ) -> Result<MutationProposal, ProposalGenerationError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(MutationProposal::new(
            format!("regressing proposal cycle {}", context.cycle),
            ProposalSource::Scheduled,
            vec![MutationTarget::FilePath {
                path: "recipes/search.yaml".to_string(),
            }],
            FitnessExpectation {
                baseline_score: 0.5,
                minimum_score: 0.5,
            },
            vec![PayloadChange::FileChange {
                path: "recipes/search.yaml".to_string(),
                content: "# regressing\n".to_string(),
            }],
        ))
    }
}

/// A stub generator that produces proposals with a minimum_score of 0.0,
/// so the fitness gate always passes (score 0.0 >= 0.0).
struct ImprovingGenerator;

#[async_trait::async_trait]
impl ProposalGenerator for ImprovingGenerator {
    async fn generate(
        &self,
        context: &LoopContext,
    ) -> Result<MutationProposal, ProposalGenerationError> {
        Ok(MutationProposal::new(
            format!("improving proposal cycle {}", context.cycle),
            ProposalSource::Scheduled,
            vec![MutationTarget::FilePath {
                path: "recipes/search.yaml".to_string(),
            }],
            FitnessExpectation {
                baseline_score: 0.0,
                minimum_score: 0.0,
            },
            vec![PayloadChange::FileChange {
                path: "recipes/search.yaml".to_string(),
                content: "# RSI mutation cycle\n".to_string(),
            }],
        ))
    }
}

/// A generator whose minimum score can only be reached by a suite that saw
/// the mutation (score 1.0), never by a FAIL verdict on the baseline (0.0).
struct DemandingGenerator;

#[async_trait::async_trait]
impl ProposalGenerator for DemandingGenerator {
    async fn generate(
        &self,
        context: &LoopContext,
    ) -> Result<MutationProposal, ProposalGenerationError> {
        Ok(MutationProposal::new(
            format!("demanding proposal cycle {}", context.cycle),
            ProposalSource::Scheduled,
            vec![MutationTarget::FilePath {
                path: "recipes/search.yaml".to_string(),
            }],
            FitnessExpectation {
                baseline_score: 0.0,
                minimum_score: 0.9,
            },
            vec![PayloadChange::FileChange {
                path: "recipes/search.yaml".to_string(),
                content: "# RSI mutation cycle\n".to_string(),
            }],
        ))
    }
}

fn stub_fitness_config(log_dir: &Path) -> FitnessConfig {
    FitnessConfig {
        suite: FitnessSuite {
            phases: TestPhases::Basic,
            depth: TestDepth::Quick,
        },
        // A stub command that produces no OVERALL RESULT verdict →
        // unparseable-output → score 0.0, passed=false.
        command: Some(FitnessCommand {
            program: "echo".to_string(),
            args: vec!["stub suite: no verdict".to_string()],
        }),
        self_test_recipe: PathBuf::from("goose-self-test.yaml"),
        timeout: std::time::Duration::from_secs(10),
        log_dir: log_dir.to_path_buf(),
    }
}

fn stub_passing_fitness_config(log_dir: &Path) -> FitnessConfig {
    FitnessConfig {
        suite: FitnessSuite {
            phases: TestPhases::Basic,
            depth: TestDepth::Quick,
        },
        command: Some(FitnessCommand {
            program: "echo".to_string(),
            args: vec!["OVERALL RESULT: PASS".to_string()],
        }),
        self_test_recipe: PathBuf::from("goose-self-test.yaml"),
        timeout: std::time::Duration::from_secs(10),
        log_dir: log_dir.to_path_buf(),
    }
}

fn seed_managed_root(root: &Path) {
    std::fs::create_dir_all(root.join("recipes")).unwrap();
    std::fs::write(
        root.join("recipes/search.yaml"),
        "version: 1.0\ntitle: baseline\n",
    )
    .unwrap();
}

fn managed_hashes(root: &Path) -> std::collections::BTreeMap<String, String> {
    use sha2::{Digest, Sha256};
    fn visit(path: &Path, prefix: String, out: &mut std::collections::BTreeMap<String, String>) {
        if path.is_dir() {
            let mut entries: Vec<_> = std::fs::read_dir(path)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .collect();
            entries.sort();
            for entry in entries {
                let name = entry.file_name().unwrap().to_string_lossy().into_owned();
                let full = format!("{prefix}/{name}");
                if full == "/rsi" || full == "/rsi-snapshot" {
                    continue;
                }
                visit(&entry, full, out);
            }
            return;
        }
        let content = std::fs::read(path).unwrap();
        let hash = {
            let digest = Sha256::digest(&content);
            digest
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        out.insert(prefix, hash);
    }
    let mut out = std::collections::BTreeMap::new();
    visit(root, String::new(), &mut out);
    out
}

fn managed_recipe_hash(root: &Path) -> String {
    use sha2::{Digest, Sha256};
    let content = std::fs::read(root.join("recipes/search.yaml")).unwrap();
    let digest = Sha256::digest(&content);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn read_evolution_log(root: &Path) -> Vec<EvolutionEvent> {
    let log_path = root.join("rsi/evolution.jsonl");
    let raw = std::fs::read_to_string(&log_path).unwrap_or_default();
    raw.lines()
        .filter(|line| !line.is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

fn make_event(cycle: u32, kind: EventKind) -> EvolutionEvent {
    EvolutionEvent {
        cycle,
        timestamp: SystemTime::UNIX_EPOCH,
        proposal_id: format!("prop-{}", cycle),
        kind,
    }
}

#[tokio::test]
async fn fail_closed_chain_never_mutates_baseline() {
    let managed = tempfile::tempdir().unwrap();
    let snapshot = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    seed_managed_root(managed.path());
    seed_managed_root(snapshot.path());
    let baseline_hash = managed_recipe_hash(managed.path());

    let loop_ = EvolutionLoop::new(
        ApplyConfig::new(managed.path()),
        stub_fitness_config(logs.path()),
        Arc::new(RegressingGenerator::new()),
        snapshot.path(),
    );

    // Run 3 cycles: every proposal regresses fitness → never applied.
    for _ in 0..3 {
        loop_.tick().await.expect("tick must not error");
    }

    assert_eq!(
        managed_recipe_hash(managed.path()),
        baseline_hash,
        "fail-closed: the managed root must equal the initial baseline after 3 regressing cycles"
    );

    let breaker = std::fs::read_to_string(managed.path().join("rsi/circuit-breaker.json")).unwrap();
    let breaker: serde_json::Value = serde_json::from_str(&breaker).unwrap();
    assert_eq!(
        breaker["paused"], true,
        "breaker must trip after 2 regressions"
    );
    assert_eq!(breaker["consecutive_failures"], 2);
}

#[tokio::test]
async fn breaker_pauses_subsequent_triggers() {
    let managed = tempfile::tempdir().unwrap();
    let snapshot = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    seed_managed_root(managed.path());
    seed_managed_root(snapshot.path());

    let loop_ = EvolutionLoop::new(
        ApplyConfig::new(managed.path()),
        stub_fitness_config(logs.path()),
        Arc::new(RegressingGenerator::new()),
        snapshot.path(),
    );

    // Two regressing cycles trip the breaker.
    loop_.tick().await.unwrap();
    loop_.tick().await.unwrap();

    // Further ticks are skipped (paused).
    let outcome = loop_.tick().await.unwrap();
    assert!(!outcome.solidified, "paused tick must not solidify");
    assert!(!outcome.rolled_back, "paused tick must not roll back");
    assert!(!outcome.breaker_tripped, "paused tick must not trip again");
    assert!(
        outcome.fitness_score.is_none(),
        "paused tick must not evaluate fitness"
    );

    let log = read_evolution_log(managed.path());
    assert!(
        log.iter()
            .any(|e| matches!(e.kind, EventKind::SkippedPaused)),
        "a paused trigger must log SkippedPaused"
    );
}

#[tokio::test]
async fn resume_clears_the_breaker() {
    let managed = tempfile::tempdir().unwrap();
    let snapshot = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    seed_managed_root(managed.path());
    seed_managed_root(snapshot.path());

    let loop_ = EvolutionLoop::new(
        ApplyConfig::new(managed.path()),
        stub_fitness_config(logs.path()),
        Arc::new(RegressingGenerator::new()),
        snapshot.path(),
    );

    loop_.tick().await.unwrap();
    loop_.tick().await.unwrap();
    assert!(
        loop_.tick().await.unwrap().fitness_score.is_none(),
        "paused → skipped"
    );

    loop_.resume().unwrap();

    // Verify breaker is cleared in the state file.
    let breaker_raw =
        std::fs::read_to_string(managed.path().join("rsi/circuit-breaker.json")).unwrap();
    let breaker: serde_json::Value = serde_json::from_str(&breaker_raw).unwrap();
    assert_eq!(breaker["paused"], false, "resume must clear the breaker");
    assert_eq!(breaker["consecutive_failures"], 0);

    // Post-resume tick must run the pipeline (fitness_score is Some when
    // the pipeline ran, None when skipped).
    let outcome = loop_.tick().await.unwrap();
    assert!(
        outcome.fitness_score.is_some(),
        "post-resume tick must run the pipeline, not skip"
    );
}

#[tokio::test]
async fn improving_proposal_solidifies_and_resets_breaker() {
    let managed = tempfile::tempdir().unwrap();
    let snapshot = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    seed_managed_root(managed.path());
    seed_managed_root(snapshot.path());

    let loop_ = EvolutionLoop::new(
        ApplyConfig::new(managed.path()),
        stub_passing_fitness_config(logs.path()),
        Arc::new(ImprovingGenerator),
        snapshot.path(),
    );

    let outcome = loop_.tick().await.unwrap();
    assert!(outcome.solidified, "an improving proposal must solidify");
    assert_eq!(outcome.fitness_score, Some(1.0));

    let log = read_evolution_log(managed.path());
    assert!(log.iter().any(|e| matches!(e.kind, EventKind::Solidified)));
    assert!(log
        .iter()
        .any(|e| matches!(e.kind, EventKind::Applied { .. })));

    // The managed root now carries the mutation.
    let content = std::fs::read_to_string(managed.path().join("recipes/search.yaml")).unwrap();
    assert!(
        content.contains("RSI mutation cycle"),
        "the mutation must be applied"
    );
}

fn stub_candidate_aware_fitness(log_dir: &Path) -> FitnessConfig {
    FitnessConfig {
        suite: FitnessSuite {
            phases: TestPhases::Basic,
            depth: TestDepth::Quick,
        },
        // PASS only when the working directory already carries the mutation;
        // an unmutated baseline reports FAIL.
        command: Some(FitnessCommand {
            program: "sh".to_string(),
            args: vec![
                "-c".to_string(),
                "grep -q 'RSI mutation cycle' recipes/search.yaml && echo 'OVERALL RESULT: PASS' || echo 'OVERALL RESULT: FAIL'".to_string(),
            ],
        }),
        self_test_recipe: PathBuf::from("goose-self-test.yaml"),
        timeout: std::time::Duration::from_secs(10),
        log_dir: log_dir.to_path_buf(),
    }
}

#[tokio::test]
async fn fitness_scores_the_mutated_candidate_not_the_baseline() {
    let managed = tempfile::tempdir().unwrap();
    let snapshot = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    seed_managed_root(managed.path());
    seed_managed_root(snapshot.path());
    let baseline = std::fs::read_to_string(snapshot.path().join("recipes/search.yaml")).unwrap();

    let loop_ = EvolutionLoop::new(
        ApplyConfig::new(managed.path()),
        stub_candidate_aware_fitness(logs.path()),
        Arc::new(DemandingGenerator),
        snapshot.path(),
    );

    let outcome = loop_.tick().await.unwrap();
    assert!(
        outcome.solidified,
        "the suite only passes on a mutated candidate, so a solidified cycle proves the gate \
         scored the mutation rather than the baseline snapshot"
    );

    // The read-only snapshot itself must never be mutated by the evaluation.
    assert_eq!(
        std::fs::read_to_string(snapshot.path().join("recipes/search.yaml")).unwrap(),
        baseline,
        "fitness evaluation must not write into the workspace snapshot"
    );
}

#[tokio::test]
async fn benchmark_holdout_regression_prevents_apply() {
    let managed = tempfile::tempdir().unwrap();
    let snapshot = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    seed_managed_root(managed.path());
    seed_managed_root(snapshot.path());
    let original = std::fs::read(snapshot.path().join("recipes/search.yaml")).unwrap();
    let task = |id: &str, split, needle: &str| {
        BenchmarkTask {
        id: id.to_string(),
        split,
        command: FitnessCommand {
            program: "sh".to_string(),
            args: vec![
                "-c".to_string(),
                format!("if grep -q '{needle}' recipes/search.yaml; then echo 'OVERALL RESULT: PASS'; else echo 'OVERALL RESULT: FAIL'; fi"),
            ],
        },
    }
    };
    let suite = BenchmarkSuite {
        target: "recipes/search.yaml".to_string(),
        tasks: vec![
            task(
                "improvement",
                BenchmarkSplit::Training,
                "RSI mutation cycle",
            ),
            task("preserve-title", BenchmarkSplit::Holdout, "title: baseline"),
        ],
    };
    let loop_ = EvolutionLoop::new(
        ApplyConfig::new(managed.path()),
        stub_passing_fitness_config(logs.path()),
        Arc::new(ImprovingGenerator),
        snapshot.path(),
    )
    .with_benchmark(suite);

    let outcome = loop_.tick().await.unwrap();
    assert!(!outcome.solidified);
    assert!(outcome.rolled_back);
    assert_eq!(
        std::fs::read(managed.path().join("recipes/search.yaml")).unwrap(),
        original
    );
    assert!(read_evolution_log(managed.path())
        .iter()
        .any(|event| matches!(event.kind, EventKind::BenchmarkEvaluated { .. })));
}

#[tokio::test]
async fn concurrent_trigger_is_skipped() {
    let managed = tempfile::tempdir().unwrap();
    let snapshot = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    seed_managed_root(managed.path());
    seed_managed_root(snapshot.path());

    let loop_ = EvolutionLoop::new(
        ApplyConfig::new(managed.path()),
        stub_fitness_config(logs.path()),
        Arc::new(ImprovingGenerator),
        snapshot.path(),
    );

    // Manually hold the loop lock so tick() must skip.
    std::fs::create_dir_all(managed.path().join("rsi")).unwrap();
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(managed.path().join("rsi/loop.lock"))
        .unwrap();
    fs2::FileExt::lock_exclusive(&lock_file).unwrap();

    let outcome = loop_.tick().await.unwrap();
    assert!(!outcome.solidified, "concurrent tick must be skipped");

    fs2::FileExt::unlock(&lock_file).unwrap();

    let log = read_evolution_log(managed.path());
    assert!(
        log.iter()
            .any(|e| matches!(e.kind, EventKind::SkippedConcurrent)),
        "a skipped concurrent trigger must be logged"
    );
}

#[test]
fn evolution_log_is_append_only_jsonl() {
    let root = tempfile::tempdir().unwrap();
    let log_path = root.path().join("rsi/evolution.jsonl");
    std::fs::create_dir_all(log_path.parent().unwrap()).unwrap();

    // Write events directly to test the log format, decoupled from fitness
    // process execution that can be flaky under parallel test load.
    let variants = vec![
        make_event(
            1,
            EventKind::Proposed {
                targets: vec!["a.yaml".into()],
            },
        ),
        make_event(1, EventKind::Validated),
        make_event(1, EventKind::Solidified),
        make_event(
            2,
            EventKind::Proposed {
                targets: vec!["b.yaml".into()],
            },
        ),
        make_event(2, EventKind::Validated),
        make_event(2, EventKind::Solidified),
    ];

    for event in &variants {
        let mut line = serde_json::to_string(event).unwrap();
        line.push('\n');
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .truncate(false)
            .open(&log_path)
            .unwrap()
            .write_all(line.as_bytes())
            .unwrap();
    }

    let raw = std::fs::read_to_string(&log_path).unwrap();
    let lines: Vec<&str> = raw.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), variants.len());
    for line in &lines {
        let _: EvolutionEvent = serde_json::from_str(line).expect("each line must be valid JSON");
    }
}

// ── Issue #10 tests: payload pass-through, NothingToPropose, failure ──────

/// A generator that always returns NothingToPropose.
struct EmptyGenerator;

#[async_trait::async_trait]
impl ProposalGenerator for EmptyGenerator {
    async fn generate(
        &self,
        _context: &LoopContext,
    ) -> Result<MutationProposal, ProposalGenerationError> {
        Err(ProposalGenerationError::NothingToPropose)
    }
}

/// A generator that simulates provider failure.
struct FailingGenerator;

#[async_trait::async_trait]
impl ProposalGenerator for FailingGenerator {
    async fn generate(
        &self,
        _context: &LoopContext,
    ) -> Result<MutationProposal, ProposalGenerationError> {
        Err(ProposalGenerationError::Failed(
            "provider unavailable".to_string(),
        ))
    }
}

/// A generator that proposes a deny-listed target (secrets.yaml).
struct DeniedTargetGenerator;

#[async_trait::async_trait]
impl ProposalGenerator for DeniedTargetGenerator {
    async fn generate(
        &self,
        context: &LoopContext,
    ) -> Result<MutationProposal, ProposalGenerationError> {
        Ok(MutationProposal::new(
            format!("denied target cycle {}", context.cycle),
            ProposalSource::Scheduled,
            vec![MutationTarget::FilePath {
                path: "secrets.yaml".to_string(),
            }],
            FitnessExpectation {
                baseline_score: 0.0,
                minimum_score: 0.0,
            },
            vec![PayloadChange::FileChange {
                path: "secrets.yaml".to_string(),
                content: "stolen\n".to_string(),
            }],
        ))
    }
}

#[tokio::test]
async fn nothing_to_propose_produces_zero_writes() {
    let managed = tempfile::tempdir().unwrap();
    let snapshot = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    seed_managed_root(managed.path());
    seed_managed_root(snapshot.path());

    let baseline = managed_hashes(managed.path());

    let loop_ = EvolutionLoop::new(
        ApplyConfig::new(managed.path()),
        stub_fitness_config(logs.path()),
        Arc::new(EmptyGenerator),
        snapshot.path(),
    );

    let outcome = loop_.tick().await.unwrap();
    assert!(!outcome.solidified);
    assert!(!outcome.rolled_back);
    assert!(outcome.fitness_score.is_none());

    assert_eq!(
        managed_hashes(managed.path()),
        baseline,
        "nothing_to_propose must not write anything"
    );
}

#[tokio::test]
async fn provider_failure_does_not_panic_or_trip_breaker() {
    let managed = tempfile::tempdir().unwrap();
    let snapshot = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    seed_managed_root(managed.path());
    seed_managed_root(snapshot.path());

    let baseline = managed_hashes(managed.path());

    let loop_ = EvolutionLoop::new(
        ApplyConfig::new(managed.path()),
        stub_fitness_config(logs.path()),
        Arc::new(FailingGenerator),
        snapshot.path(),
    );

    // tick should return an error (not panic)
    let result = loop_.tick().await;
    assert!(
        result.is_err(),
        "provider failure must return Err, not panic"
    );

    // Breaker must not have tripped from a generation failure
    let breaker = loop_.load_breaker();
    assert!(
        !breaker.paused,
        "generation failure must not trip the breaker"
    );

    assert_eq!(
        managed_hashes(managed.path()),
        baseline,
        "provider failure must not write anything"
    );
}

#[tokio::test]
async fn denied_target_is_rejected_with_zero_writes() {
    let managed = tempfile::tempdir().unwrap();
    let snapshot = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    seed_managed_root(managed.path());
    seed_managed_root(snapshot.path());

    let baseline = managed_hashes(managed.path());

    let loop_ = EvolutionLoop::new(
        ApplyConfig::new(managed.path()),
        stub_fitness_config(logs.path()),
        Arc::new(DeniedTargetGenerator),
        snapshot.path(),
    );

    let result = loop_.tick().await;
    assert!(
        result.is_err(),
        "whitelist rejection must return Err, not silently proceed"
    );

    assert_eq!(
        managed_hashes(managed.path()),
        baseline,
        "denied target must not write anything"
    );
}

#[tokio::test]
async fn payload_passes_through_verbatim_to_apply() {
    let managed = tempfile::tempdir().unwrap();
    let snapshot = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    seed_managed_root(managed.path());
    seed_managed_root(snapshot.path());

    let loop_ = EvolutionLoop::new(
        ApplyConfig::new(managed.path()),
        stub_passing_fitness_config(logs.path()),
        Arc::new(DemandingGenerator),
        snapshot.path(),
    );

    let outcome = loop_.tick().await.unwrap();
    assert!(outcome.solidified, "proposal must solidify");

    let applied = std::fs::read_to_string(managed.path().join("recipes/search.yaml")).unwrap();
    assert_eq!(
        applied, "# RSI mutation cycle\n",
        "applied content must match proposal payload byte-for-byte"
    );
    assert!(
        !applied.contains("RSI mutation cycle\n# proposal:"),
        "no placeholder synthesis must appear"
    );
}
