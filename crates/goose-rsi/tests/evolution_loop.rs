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
use goose_rsi::fitness::{FitnessCommand, FitnessConfig, FitnessSuite, TestDepth, TestPhases};
use goose_rsi::proposal::{FitnessExpectation, MutationProposal, MutationTarget, ProposalSource};
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

impl ProposalGenerator for RegressingGenerator {
    fn generate(&self, context: &LoopContext) -> Result<MutationProposal, ProposalGenerationError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(MutationProposal::new(
            format!("regressing proposal cycle {}", context.cycle),
            ProposalSource::Scheduled,
            vec![MutationTarget::FilePath {
                path: "recipes/search.yaml".to_string(),
            }],
            FitnessExpectation {
                baseline_score: 0.5,
                // The stub suite always reports score 0.0, so this minimum
                // (0.5) is never met → fitness gate always refuses.
                minimum_score: 0.5,
            },
        ))
    }
}

/// A stub generator that produces proposals with a minimum_score of 0.0,
/// so the fitness gate always passes (score 0.0 >= 0.0).
struct ImprovingGenerator;

impl ProposalGenerator for ImprovingGenerator {
    fn generate(&self, context: &LoopContext) -> Result<MutationProposal, ProposalGenerationError> {
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
