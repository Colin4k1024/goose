//! Produce one real `FitnessReport` JSON by executing the fitness executor
//! end to end (issue #2 evidence generator).
//!
//! Usage:
//! ```text
//! cargo run -p goose-rsi --example fitness_report -- <workspace_dir> <log_dir>
//! ```
//! The suite command is the deterministic stub suite defined below: a real
//! subprocess execution of the executor's full code path (spawn, capture,
//! parse, persist raw log). Swap in `FitnessConfig::command = None` to run
//! the real goose self-test recipe when a provider is configured.

use goose_rsi::fitness::{
    run_fitness, FitnessCommand, FitnessConfig, FitnessSuite, TestDepth, TestPhases,
};
use goose_rsi::proposal::{FitnessExpectation, MutationProposal, MutationTarget, ProposalSource};
use std::path::PathBuf;
use std::time::Duration;

fn main() {
    let workspace = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let log_dir = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("goose-rsi-fitness-evidence"));

    let stub_suite = r#"================================
✅ OVERALL RESULT: PASS

📊 Quick Stats:
• Tests Run: 10
• Passed: 9 (90%)
• Failed: 1 (10%)

✅ Working Features:
• File operations: ✓
• Shell commands: ✓
• Extensions: ✗
================================
"#;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let report = runtime.block_on(run_fitness(
        &FitnessConfig {
            suite: FitnessSuite {
                phases: TestPhases::Basic,
                depth: TestDepth::Quick,
            },
            command: Some(FitnessCommand {
                program: "bash".to_string(),
                args: vec!["-c".to_string(), format!("printf '%s' '{}'", stub_suite)],
            }),
            self_test_recipe: PathBuf::from("goose-self-test.yaml"),
            timeout: Duration::from_secs(60),
            log_dir: log_dir.clone(),
        },
        &MutationProposal::new(
            "evidence run: deterministic stub suite",
            ProposalSource::Scheduled,
            vec![MutationTarget::FilePath {
                path: "recipes/search.yaml".to_string(),
            }],
            FitnessExpectation {
                baseline_score: 0.5,
                minimum_score: 0.5,
            },
        ),
        &workspace,
    ));

    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize report")
    );
    eprintln!("raw log: {}", report.raw_log_ref.display());
}
