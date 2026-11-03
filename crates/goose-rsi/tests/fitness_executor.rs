//! Fitness executor acceptance tests (issue #2).
//!
//! Anti-goodhart (fitness definitions outside the mutation whitelist),
//! fault injection (timeout / crash / bad exit / unparseable output) without
//! panics, determinism, read-only snapshots, and no residual state after an
//! executor is killed mid-run.

use goose_rsi::fitness::{
    run_fitness, FitnessCommand, FitnessConfig, FitnessSuite, TestDepth, TestPhases,
};
use goose_rsi::proposal::{FitnessExpectation, MutationProposal, MutationTarget, ProposalSource};
use goose_rsi::whitelist::{validate_target, WhitelistCheck};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

fn proposal() -> MutationProposal {
    MutationProposal::new(
        "fitness test proposal",
        ProposalSource::Scheduled,
        vec![MutationTarget::FilePath {
            path: "recipes/search.yaml".to_string(),
        }],
        FitnessExpectation {
            baseline_score: 0.5,
            minimum_score: 0.5,
        },
        vec![],
    )
}

fn config_with(command: FitnessCommand, log_dir: &Path) -> FitnessConfig {
    FitnessConfig {
        suite: FitnessSuite {
            phases: TestPhases::Basic,
            depth: TestDepth::Quick,
        },
        command: Some(command),
        self_test_recipe: Path::new("goose-self-test.yaml").to_path_buf(),
        timeout: Duration::from_secs(30),
        log_dir: log_dir.to_path_buf(),
    }
}

fn bash(script: &str) -> FitnessCommand {
    FitnessCommand {
        program: "bash".to_string(),
        args: vec!["-c".to_string(), script.to_string()],
    }
}

fn hash_tree(path: &Path) -> BTreeMap<String, u128> {
    fn visit(path: &Path, prefix: String, out: &mut BTreeMap<String, u128>) {
        if path.is_dir() {
            let mut entries: Vec<_> = std::fs::read_dir(path)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .collect();
            entries.sort();
            for entry in entries {
                let name = entry.file_name().unwrap().to_string_lossy().into_owned();
                visit(&entry, format!("{prefix}/{name}"), out);
            }
            return;
        }
        let bytes = std::fs::read(path).unwrap();
        let mut hash: u128 = 0;
        // FNV-1a over 128 bits: stable across runs, enough for change detection
        for byte in bytes {
            hash ^= byte as u128;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        out.insert(prefix, hash);
    }
    let mut out = BTreeMap::new();
    visit(path, String::new(), &mut out);
    out
}

fn snapshot_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("snapshot tempdir");
    std::fs::write(dir.path().join("recipe.yaml"), "version: 1.0\n").unwrap();
    std::fs::create_dir_all(dir.path().join("recipes")).unwrap();
    std::fs::write(dir.path().join("recipes/a.yaml"), "version: 1.0\n").unwrap();
    dir
}

const PASS_OUTPUT: &str = "\
================================
✅ OVERALL RESULT: PASS

📊 Quick Stats:
• Tests Run: 10
• Passed: 10 (100%)
• Failed: 0 (0%)

✅ Working Features:
• File operations: ✓
• Shell commands: ✓
================================
";

const FAIL_OUTPUT: &str = "\
================================
❌ OVERALL RESULT: FAIL

📊 Quick Stats:
• Tests Run: 10
• Passed: 4 (40%)
• Failed: 6 (60%)

✅ Working Features:
• File operations: ✗
• Shell commands: ✓
• Extensions: ✗
================================
";

#[test]
fn fitness_definitions_live_outside_the_mutation_whitelist() {
    // The engine scores itself against these surfaces; none of them may be
    // reachable by a mutation proposal.
    for frozen in [
        "crates/goose-rsi/src/fitness.rs",
        "crates/goose-rsi/src/whitelist.rs",
        "goose-self-test.yaml",
    ] {
        let error = validate_target(&MutationTarget::FilePath {
            path: frozen.to_string(),
        })
        .expect_err("fitness/eval surfaces must be outside the mutation whitelist");
        assert!(
            matches!(error.check, WhitelistCheck::NotWhitelisted { .. }),
            "expected NotWhitelisted for {frozen}, got {error:?}"
        );
    }
}

#[test]
fn suite_selector_maps_to_self_test_semantics() {
    assert_eq!(TestPhases::DeclaredVision.as_str(), "declared-vision");
    assert_eq!(TestPhases::ZaiCodingPlan.as_str(), "zai-coding-plan");
    assert_eq!(TestDepth::Quick.as_str(), "quick");

    let command = FitnessCommand::default_suite_command(
        FitnessSuite {
            phases: TestPhases::Basic,
            depth: TestDepth::Quick,
        },
        Path::new("/repo/goose-self-test.yaml"),
        Path::new("/logs/suite-ws"),
    );
    assert_eq!(command.program, "goose");
    let joined = command.args.join(" ");
    assert!(joined.contains("--recipe /repo/goose-self-test.yaml"));
    assert!(joined.contains("test_phases=basic"));
    assert!(joined.contains("test_depth=quick"));
    assert!(joined.contains("workspace_dir=/logs/suite-ws"));
    assert!(joined.contains("cleanup_after=false"));
}

#[tokio::test]
async fn passing_suite_scores_full_and_leaves_snapshot_untouched() {
    let logs = tempfile::tempdir().unwrap();
    let snapshot = snapshot_dir();
    let before = hash_tree(snapshot.path());

    let report = run_fitness(
        &config_with(bash(&format!("printf '%s' '{}'", PASS_OUTPUT)), logs.path()),
        &proposal(),
        snapshot.path(),
    )
    .await;

    assert!(report.passed);
    assert_eq!(report.score, 1.0);
    assert!(report.failed_phases.is_empty());
    assert_eq!(
        before,
        hash_tree(snapshot.path()),
        "snapshot must stay read-only"
    );

    let raw_log = std::fs::read_to_string(&report.raw_log_ref).unwrap();
    assert!(raw_log.contains("=== stdout ==="));
    assert!(raw_log.contains("OVERALL RESULT: PASS"));
}

#[tokio::test]
async fn failing_suite_reports_zero_score_and_failed_phases() {
    let logs = tempfile::tempdir().unwrap();
    let snapshot = snapshot_dir();

    let report = run_fitness(
        &config_with(bash(&format!("printf '%s' '{}'", FAIL_OUTPUT)), logs.path()),
        &proposal(),
        snapshot.path(),
    )
    .await;

    assert!(!report.passed);
    assert_eq!(report.score, 0.0);
    assert_eq!(report.failed_phases, vec!["File operations", "Extensions"]);
}

#[tokio::test]
async fn timeout_degrades_to_failure_without_panicking() {
    let logs = tempfile::tempdir().unwrap();
    let mut config = config_with(
        FitnessCommand {
            program: "sleep".to_string(),
            args: vec!["30".to_string()],
        },
        logs.path(),
    );
    config.timeout = Duration::from_millis(150);

    let report = run_fitness(&config, &proposal(), snapshot_dir().path()).await;

    assert!(!report.passed);
    assert_eq!(report.score, 0.0);
    assert_eq!(report.failed_phases, vec!["timeout"]);
}

#[tokio::test]
async fn child_sigkill_degrades_to_crash_report() {
    let logs = tempfile::tempdir().unwrap();
    let report = run_fitness(
        &config_with(bash("kill -9 $$"), logs.path()),
        &proposal(),
        snapshot_dir().path(),
    )
    .await;

    assert!(!report.passed);
    assert_eq!(report.score, 0.0);
    assert_eq!(report.failed_phases, vec!["crash:9"]);
}

#[tokio::test]
async fn nonzero_exit_degrades_to_exit_report() {
    let logs = tempfile::tempdir().unwrap();
    let report = run_fitness(
        &config_with(
            FitnessCommand {
                program: "false".to_string(),
                args: vec![],
            },
            logs.path(),
        ),
        &proposal(),
        snapshot_dir().path(),
    )
    .await;

    assert!(!report.passed);
    assert_eq!(report.score, 0.0);
    assert_eq!(report.failed_phases, vec!["exit:1"]);
}

#[tokio::test]
async fn unparseable_output_fails_closed() {
    let logs = tempfile::tempdir().unwrap();
    let report = run_fitness(
        &config_with(bash("echo 'random noise'"), logs.path()),
        &proposal(),
        snapshot_dir().path(),
    )
    .await;

    assert!(!report.passed);
    assert_eq!(report.score, 0.0);
    assert_eq!(report.failed_phases, vec!["unparseable-output"]);
}

#[tokio::test]
async fn same_snapshot_produces_the_same_deterministic_core() {
    let logs = tempfile::tempdir().unwrap();
    let snapshot = snapshot_dir();
    let config = config_with(bash(&format!("printf '%s' '{}'", PASS_OUTPUT)), logs.path());

    let proposal = proposal();
    let first = run_fitness(&config, &proposal, snapshot.path()).await;
    let second = run_fitness(&config, &proposal, snapshot.path()).await;

    assert!(first.deterministic_core_eq(&second));
    // Non-deterministic fields differ or not, but the score is stable.
    assert_eq!(first.score, second.score);
}

#[tokio::test]
async fn killed_executor_leaves_no_residual_state() {
    let logs = tempfile::tempdir().unwrap();
    // Simulate a previous executor dying mid-write: a partial raw log and a
    // stale suite workspace must not block the next invocation.
    let proposal = proposal();
    std::fs::write(
        logs.path().join(format!("fitness-{}.log", proposal.id)),
        "partial",
    )
    .unwrap();
    std::fs::create_dir_all(logs.path().join(format!("suite-workspace-{}", proposal.id))).unwrap();

    // Abort a running fitness execution mid-flight (executor killed).
    let mut hanging = config_with(
        FitnessCommand {
            program: "sleep".to_string(),
            args: vec!["30".to_string()],
        },
        logs.path(),
    );
    hanging.timeout = Duration::from_secs(30);
    let hanging_config = hanging.clone();
    let hanging_proposal = proposal.clone();
    let task = tokio::spawn(async move {
        run_fitness(&hanging_config, &hanging_proposal, snapshot_dir().path()).await
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    task.abort();

    // The very next invocation works and overwrites the partial log.
    let report = run_fitness(
        &config_with(bash(&format!("printf '%s' '{}'", PASS_OUTPUT)), logs.path()),
        &proposal,
        snapshot_dir().path(),
    )
    .await;
    assert!(report.passed);
    let raw_log = std::fs::read_to_string(&report.raw_log_ref).unwrap();
    assert!(
        raw_log.contains("=== stdout ==="),
        "stale log must be replaced"
    );
}

#[tokio::test]
async fn report_serializes_to_json_for_evidence() {
    let logs = tempfile::tempdir().unwrap();
    let report = run_fitness(
        &config_with(bash(&format!("printf '%s' '{}'", PASS_OUTPUT)), logs.path()),
        &proposal(),
        snapshot_dir().path(),
    )
    .await;
    let json = serde_json::to_string_pretty(&report).unwrap();
    assert!(json.contains("\"passed\": true"));
    assert!(json.contains("\"score\": 1.0"));
}
