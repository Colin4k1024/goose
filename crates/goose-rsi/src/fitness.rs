//! Frozen fitness executor (issue #2, RSI M1).
//!
//! Runs an eval suite (by default the goose self-test recipe, reusing its
//! `test_phases`/`test_depth` semantics) inside a sandboxed working copy and
//! produces a structured [`FitnessReport`] used as the RSI acceptance gate.
//!
//! Guarantees:
//!
//! * **anti-goodhart**: this module and the eval suite it runs live outside
//!   the RSI mutation whitelist — the engine cannot mutate the standard it is
//!   scored against (asserted by tests, enforced by the frozen whitelist in
//!   [`crate::whitelist`]);
//! * **no panic, no lock**: timeouts and child crashes degrade to a
//!   failure report; the executor is stateless and holds no cross-run locks,
//!   so a SIGKILLed executor never blocks the next invocation;
//! * **determinism**: the same proposal + the same workspace snapshot produce
//!   the same score; only timestamps, durations and log paths may differ;
//! * **read-only snapshot**: the executor never writes into the workspace
//!   snapshot it is given; suite artifacts are directed to the
//!   executor-managed log directory.
//!
//! The default suite command shells out to
//! `goose run --recipe <self-test> --params test_phases=<..> --params
//! test_depth=<..> ...`. The command is injectable so tests (and CI without
//! provider credentials) run the same code path against stub commands.

use crate::proposal::MutationProposal;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, SystemTime};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::Instant;

/// Which phases of the eval suite to run. Mirrors the `test_phases`
/// parameter of goose-self-test.yaml.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestPhases {
    All,
    Basic,
    Extensions,
    Delegation,
    CodeMode,
    DeclaredVision,
    Reasoning,
    ZaiCodingPlan,
    AcpEffort,
    Compaction,
    Advanced,
}

impl TestPhases {
    pub const fn as_str(self) -> &'static str {
        match self {
            TestPhases::All => "all",
            TestPhases::Basic => "basic",
            TestPhases::Extensions => "extensions",
            TestPhases::Delegation => "delegation",
            TestPhases::CodeMode => "code-mode",
            TestPhases::DeclaredVision => "declared-vision",
            TestPhases::Reasoning => "reasoning",
            TestPhases::ZaiCodingPlan => "zai-coding-plan",
            TestPhases::AcpEffort => "acp-effort",
            TestPhases::Compaction => "compaction",
            TestPhases::Advanced => "advanced",
        }
    }
}

/// Testing depth. Mirrors the `test_depth` parameter of goose-self-test.yaml.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestDepth {
    Quick,
    Standard,
    Deep,
}

impl TestDepth {
    pub const fn as_str(self) -> &'static str {
        match self {
            TestDepth::Quick => "quick",
            TestDepth::Standard => "standard",
            TestDepth::Deep => "deep",
        }
    }
}

/// Selector over the eval suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FitnessSuite {
    pub phases: TestPhases,
    pub depth: TestDepth,
}

impl Default for FitnessSuite {
    fn default() -> Self {
        FitnessSuite {
            phases: TestPhases::All,
            depth: TestDepth::Standard,
        }
    }
}

/// The command the executor runs. Injectable for tests and CI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FitnessCommand {
    pub program: String,
    pub args: Vec<String>,
}

impl FitnessCommand {
    /// Default suite command: run the goose self-test recipe with the
    /// suite selector mapped onto `test_phases`/`test_depth` and the suite
    /// workspace directed into the executor-managed log directory (never the
    /// snapshot under test).
    pub fn default_suite_command(
        suite: FitnessSuite,
        self_test_recipe: &Path,
        suite_workspace_dir: &Path,
    ) -> Self {
        FitnessCommand {
            program: "goose".to_string(),
            args: vec![
                "run".to_string(),
                "--recipe".to_string(),
                self_test_recipe.display().to_string(),
                "--params".to_string(),
                format!("test_phases={}", suite.phases.as_str()),
                "--params".to_string(),
                format!("test_depth={}", suite.depth.as_str()),
                "--params".to_string(),
                format!("workspace_dir={}", suite_workspace_dir.display()),
                // keep artifacts for the raw log; the executor manages cleanup
                "--params".to_string(),
                "cleanup_after=false".to_string(),
            ],
        }
    }
}

/// Executor configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FitnessConfig {
    pub suite: FitnessSuite,
    /// Override for the suite command. When `None`, the default goose
    /// self-test command is used with [`FitnessConfig::self_test_recipe`].
    pub command: Option<FitnessCommand>,
    /// Path to goose-self-test.yaml for the default command.
    pub self_test_recipe: PathBuf,
    /// Hard wall-clock budget for one suite run.
    pub timeout: Duration,
    /// Directory for raw logs and suite artifacts.
    pub log_dir: PathBuf,
}

/// Structured verdict of one fitness execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitnessReport {
    pub proposal_id: String,
    /// `true` when the suite ran to completion and reported PASS.
    pub passed: bool,
    /// Score in `[0.0, 1.0]`: parsed pass percentage when available, `1.0`
    /// on an explicit PASS without percentages, `0.0` on any failure,
    /// timeout, crash or unparseable output (fail-closed).
    pub score: f64,
    /// Suite features that reported failures, or the degradation reason
    /// (`"timeout"`, `"crash:<signal>"`, `"exit:<code>"`,
    /// `"unparseable-output"`).
    pub failed_phases: Vec<String>,
    /// Wall time of the execution. Not deterministic by nature.
    pub duration: Duration,
    /// Path to the captured raw output. Not deterministic by nature.
    pub raw_log_ref: PathBuf,
    /// When the report was produced. Not deterministic by nature.
    pub timestamp: SystemTime,
}

impl FitnessReport {
    /// Compare the deterministic core of two reports (everything except
    /// duration, log path and timestamp).
    pub fn deterministic_core_eq(&self, other: &FitnessReport) -> bool {
        self.proposal_id == other.proposal_id
            && self.passed == other.passed
            && self.score == other.score
            && self.failed_phases == other.failed_phases
    }
}

/// Degradation-safe fitness execution. Never panics on child failure; a
/// timeout or crash yields a failing report.
///
/// `workspace_snapshot` is used read-only: it becomes the working directory
/// of the suite command, while all executor and suite output goes to
/// `config.log_dir`.
pub async fn run_fitness(
    config: &FitnessConfig,
    proposal: &MutationProposal,
    workspace_snapshot: &Path,
) -> FitnessReport {
    let started = Instant::now();
    let timestamp = SystemTime::now();
    let log_dir = &config.log_dir;
    let raw_log_path = log_dir.join(format!("fitness-{}.log", proposal.id));

    let spec = match &config.command {
        Some(spec) => spec.clone(),
        None => {
            let suite_workspace = log_dir.join(format!("suite-workspace-{}", proposal.id));
            FitnessCommand::default_suite_command(
                config.suite,
                &config.self_test_recipe,
                &suite_workspace,
            )
        }
    };
    let mut command = Command::new(&spec.program);
    command
        .args(&spec.args)
        .current_dir(workspace_snapshot)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let mut report = match tokio::time::timeout(config.timeout, command.output()).await {
        Err(_elapsed) => degraded(
            proposal,
            vec!["timeout".to_string()],
            &raw_log_path,
            started,
            timestamp,
        ),
        Ok(Err(_spawn_failed)) => degraded(
            proposal,
            vec!["spawn-failed".to_string()],
            &raw_log_path,
            started,
            timestamp,
        ),
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            let report = match output.status.code() {
                None => {
                    let signal = signal_of(&output.status);
                    degraded(
                        proposal,
                        vec![format!("crash:{signal}")],
                        &raw_log_path,
                        started,
                        timestamp,
                    )
                }
                Some(0) => parse_success(proposal, &stdout, &raw_log_path, started, timestamp),
                Some(code) => degraded(
                    proposal,
                    vec![format!("exit:{code}")],
                    &raw_log_path,
                    started,
                    timestamp,
                ),
            };
            persist_raw_log(&raw_log_path, &stdout, &stderr)
                .await
                .unwrap_or_else(|error| {
                    eprintln!("goose-rsi: failed to persist fitness log: {error}");
                });
            return report;
        }
    };

    if let Err(error) = std::fs::create_dir_all(log_dir) {
        eprintln!("goose-rsi: failed to create fitness log dir: {error}");
    }
    if let Err(error) = tokio::fs::write(
        &raw_log_path,
        format!(
            "goose-rsi fitness: suite never produced output ({})\n",
            report
                .failed_phases
                .first()
                .map(String::as_str)
                .unwrap_or("unknown")
        ),
    )
    .await
    {
        eprintln!("goose-rsi: failed to persist fitness degradation note: {error}");
    }
    report.duration = started.elapsed();
    report
}

fn degraded(
    proposal: &MutationProposal,
    failed_phases: Vec<String>,
    raw_log_path: &Path,
    started: Instant,
    timestamp: SystemTime,
) -> FitnessReport {
    FitnessReport {
        proposal_id: proposal.id.to_string(),
        passed: false,
        score: 0.0,
        failed_phases,
        duration: started.elapsed(),
        raw_log_ref: raw_log_path.to_path_buf(),
        timestamp,
    }
}

/// Parse a successful suite run (exit 0) from the self-test summary format:
/// `OVERALL RESULT: PASS|FAIL`, `Passed: X (Y%)` and feature lines with
/// `✓`/`✗` markers. Fail-closed: output without a recognizable overall
/// verdict scores 0.0.
fn parse_success(
    proposal: &MutationProposal,
    stdout: &str,
    raw_log_path: &Path,
    started: Instant,
    timestamp: SystemTime,
) -> FitnessReport {
    let Some(overall_pass) = overall_verdict(stdout) else {
        return degraded(
            proposal,
            vec!["unparseable-output".to_string()],
            raw_log_path,
            started,
            timestamp,
        );
    };

    let score = if overall_pass {
        parse_pass_percentage(stdout).unwrap_or(1.0)
    } else {
        0.0
    };
    let failed_phases = parse_failed_features(stdout);
    FitnessReport {
        proposal_id: proposal.id.to_string(),
        passed: overall_pass,
        score,
        failed_phases,
        duration: started.elapsed(),
        raw_log_ref: raw_log_path.to_path_buf(),
        timestamp,
    }
}

/// Line-exact verdict detection against the self-test summary contract
/// (`✅ OVERALL RESULT: PASS`). Fail-closed semantics:
///
/// * any line ending in `OVERALL RESULT: FAIL` → `Some(false)`;
/// * otherwise a line ending in `OVERALL RESULT: PASS` → `Some(true)`;
/// * otherwise (including substrings like `OVERALL RESULT: PASSED_ANYTHING`)
///   → `None`.
fn overall_verdict(stdout: &str) -> Option<bool> {
    if stdout
        .lines()
        .any(|line| line.trim().ends_with("OVERALL RESULT: FAIL"))
    {
        return Some(false);
    }
    if stdout
        .lines()
        .any(|line| line.trim().ends_with("OVERALL RESULT: PASS"))
    {
        return Some(true);
    }
    None
}

fn parse_pass_percentage(stdout: &str) -> Option<f64> {
    for line in stdout.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("• Passed:") {
            let percent = rest.split('(').nth(1)?.split('%').next()?;
            let value: f64 = percent.trim().parse().ok()?;
            return Some(value.clamp(0.0, 100.0) / 100.0);
        }
    }
    None
}

fn parse_failed_features(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim().trim_start_matches('•').trim();
            let (name, verdict) = trimmed.split_once(':')?;
            let verdict = verdict.trim();
            if verdict.contains('✗') {
                Some(name.trim().to_string())
            } else {
                None
            }
        })
        .collect()
}

#[cfg(unix)]
fn signal_of(status: &std::process::ExitStatus) -> String {
    use std::os::unix::process::ExitStatusExt;
    status
        .signal()
        .map(|signal| signal.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(not(unix))]
fn signal_of(_status: &std::process::ExitStatus) -> String {
    "unknown".to_string()
}

async fn persist_raw_log(path: &Path, stdout: &str, stderr: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let mut file = tokio::fs::File::create(path).await?;
    file.write_all(b"=== stdout ===\n").await?;
    file.write_all(stdout.as_bytes()).await?;
    file.write_all(b"\n=== stderr ===\n").await?;
    file.write_all(stderr.as_bytes()).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_matching_is_line_exact_and_fail_closed() {
        assert_eq!(overall_verdict("✅ OVERALL RESULT: PASS\n"), Some(true));
        assert_eq!(overall_verdict("❌ OVERALL RESULT: FAIL\n"), Some(false));
        // substrings that merely contain the marker do not count
        assert_eq!(overall_verdict("OVERALL RESULT: PASSED_ANYTHING"), None);
        assert_eq!(overall_verdict("OVERALL RESULT: FAILURE_MODE_TEST"), None);
        assert_eq!(overall_verdict("random noise"), None);
        assert_eq!(overall_verdict(""), None);
        // mixed verdicts: any FAIL line wins (fail-closed)
        assert_eq!(
            overall_verdict("✅ OVERALL RESULT: PASS\n❌ OVERALL RESULT: FAIL"),
            Some(false)
        );
    }

    #[test]
    fn pass_percentage_parses_the_summary_contract() {
        let output = "📊 Quick Stats:\n• Tests Run: 10\n• Passed: 9 (90%)\n• Failed: 1 (10%)";
        assert_eq!(parse_pass_percentage(output), Some(0.9));
        assert_eq!(parse_pass_percentage("• Passed: 10 (100%)"), Some(1.0));
        assert_eq!(parse_pass_percentage("no stats"), None);
        // out-of-range percentages are clamped into the score domain
        assert_eq!(parse_pass_percentage("• Passed: 1 (250%)"), Some(1.0));
    }

    #[test]
    fn failed_features_collect_only_cross_marks() {
        let output = "\
✅ Working Features:
• File operations: ✓
• Shell commands: ✗
• Extensions: ✗
• no colon line ✗
";
        assert_eq!(
            parse_failed_features(output),
            vec!["Shell commands".to_string(), "Extensions".to_string()]
        );
    }
}
