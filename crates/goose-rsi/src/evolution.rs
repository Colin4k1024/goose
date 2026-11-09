//! Scheduled evolution loop (issue #4, RSI M2).
//!
//! Cron-triggered orchestrator that runs the full
//! Detect → Propose → Validate → Fitness → Apply → Solidify pipeline per
//! cycle, reusing the modules from issues #1-#3. The loop is driven by an
//! external scheduler (goose's `scheduler_trait`) calling [`EvolutionLoop::tick`];
//! it is not a resident thread, keeping token cost bounded to one cycle per
//! trigger.
//!
//! Concurrency and convergence invariants:
//!
//! 1. **at most one cycle at a time** — a cross-process exclusive lock
//!    (`fs2` flock) guards the loop; a second trigger while one is running
//!    is skipped and logged;
//! 2. **circuit breaker** — two consecutive fitness regressions pause the
//!    loop; further cron triggers produce no proposals or writes until an
//!    explicit [`resume`];
//! 3. **fail-closed full chain** — a proposal generator that always lowers
//!    fitness never mutates the managed root: fitness gate refuses apply,
//!    the proposal is rolled back, and after two regressions the breaker
//!    trips;
//! 4. **the gate scores the mutation, not the baseline** — the change set is
//!    materialized into a throwaway candidate copy of the workspace snapshot
//!    and the eval suite runs there, so the report describes the candidate
//!    that would be applied rather than the untouched snapshot.

use crate::apply::{self, apply, materialize_candidate, recover, ApplyConfig, Change};
use crate::fitness::{
    run_benchmark, run_fitness, BenchmarkResult, BenchmarkSuite, FitnessConfig, FitnessReport,
};
use crate::proposal::{MutationProposal, MutationTarget, PayloadChange};
use crate::state::{tag, ProposalState, ProposalStateMachine};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

/// Injectable proposal generator: the Detect→Propose step. Production wires
/// this to an LLM call; tests use a deterministic stub.
#[async_trait::async_trait]
pub trait ProposalGenerator: Send + Sync {
    async fn generate(
        &self,
        context: &LoopContext,
    ) -> Result<MutationProposal, ProposalGenerationError>;
}

/// Why proposal generation failed (non-fatal: the cycle is skipped).
#[derive(Debug, thiserror::Error)]
pub enum ProposalGenerationError {
    #[error("no improvement to propose in this cycle")]
    NothingToPropose,
    #[error("proposal generation failed: {0}")]
    Failed(String),
}

/// Read-only context handed to the proposal generator each cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopContext {
    pub cycle: u32,
    pub last_fitness: Option<f64>,
    pub last_proposal_id: Option<String>,
}

/// One append-only event in the evolution log (consumed by issue #5 CLI).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionEvent {
    pub cycle: u32,
    pub timestamp: SystemTime,
    pub proposal_id: String,
    #[serde(flatten)]
    pub kind: EventKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum EventKind {
    Proposed {
        targets: Vec<String>,
    },
    Validated,
    FitnessEvaluated {
        report: FitnessReport,
    },
    BenchmarkEvaluated {
        result: BenchmarkResult,
    },
    Applied {
        backup_version: u32,
    },
    Solidified,
    RolledBack {
        reason: String,
    },
    /// The circuit breaker tripped after consecutive regressions.
    BreakerTripped {
        consecutive_failures: u32,
    },
    /// A cron trigger was skipped because a cycle was already running.
    SkippedConcurrent,
    /// A cron trigger was skipped because the breaker is paused.
    SkippedPaused,
}

/// Circuit-breaker state, persisted to `rsi/circuit-breaker.json`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CircuitBreaker {
    pub consecutive_failures: u32,
    pub paused: bool,
}

impl CircuitBreaker {
    /// Two consecutive fitness regressions trip the breaker.
    const TRIP_THRESHOLD: u32 = 2;
}

/// One evolution cycle's outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CycleOutcome {
    pub cycle: u32,
    pub solidified: bool,
    pub rolled_back: bool,
    pub breaker_tripped: bool,
    pub fitness_score: Option<f64>,
}

/// The evolution loop. Held by the scheduler; [`EvolutionLoop::tick`] runs
/// one cycle.
pub struct EvolutionLoop {
    apply_config: ApplyConfig,
    fitness_config: FitnessConfig,
    generator: Arc<dyn ProposalGenerator>,
    /// Workspace snapshot root for fitness evaluation (read-only copy of the
    /// managed root).
    workspace_snapshot: PathBuf,
    /// Path to the evolution log JSONL.
    log_path: PathBuf,
    breaker_path: PathBuf,
    lock_path: PathBuf,
    /// Cycle counter state file.
    state_path: PathBuf,
    /// Max proposals per cycle (default 1).
    max_proposals_per_cycle: u32,
    benchmark: Option<BenchmarkSuite>,
}

impl EvolutionLoop {
    pub fn new(
        apply_config: ApplyConfig,
        fitness_config: FitnessConfig,
        generator: Arc<dyn ProposalGenerator>,
        workspace_snapshot: impl Into<PathBuf>,
    ) -> Self {
        let rsi_dir = apply_config.rsi_dir();
        EvolutionLoop {
            apply_config,
            fitness_config,
            generator,
            workspace_snapshot: workspace_snapshot.into(),
            log_path: rsi_dir.join("evolution.jsonl"),
            breaker_path: rsi_dir.join("circuit-breaker.json"),
            lock_path: rsi_dir.join("loop.lock"),
            state_path: rsi_dir.join("loop-state.json"),
            max_proposals_per_cycle: 1,
            benchmark: None,
        }
    }

    pub fn with_benchmark(mut self, benchmark: BenchmarkSuite) -> Self {
        self.benchmark = Some(benchmark);
        self
    }

    pub fn with_max_proposals_per_cycle(mut self, max: u32) -> Self {
        self.max_proposals_per_cycle = max;
        self
    }

    /// Run one evolution cycle. Safe to call from a cron trigger: concurrent
    /// calls are skipped, a tripped breaker skips everything.
    pub async fn tick(&self) -> Result<CycleOutcome, LoopError> {
        std::fs::create_dir_all(self.apply_config.rsi_dir())?;

        let _lock = match self.acquire_lock() {
            Ok(lock) => lock,
            Err(LoopError::AlreadyRunning) => {
                self.append_event(0, "", &EventKind::SkippedConcurrent)?;
                return Ok(CycleOutcome {
                    cycle: self.next_cycle_number(),
                    solidified: false,
                    rolled_back: false,
                    breaker_tripped: false,
                    fitness_score: None,
                });
            }
            Err(e) => return Err(e),
        };

        // Startup recovery: heal any crashed apply from a previous cycle.
        recover(&self.apply_config)?;

        let breaker = self.load_breaker();
        if breaker.paused {
            self.append_event(0, "", &EventKind::SkippedPaused)?;
            return Ok(CycleOutcome {
                cycle: self.next_cycle_number(),
                solidified: false,
                rolled_back: false,
                breaker_tripped: false,
                fitness_score: None,
            });
        }

        let cycle = self.increment_cycle();
        let context = LoopContext {
            cycle,
            last_fitness: self.last_fitness_score(),
            last_proposal_id: self.last_proposal_id(),
        };

        let outcome = self.run_cycle(cycle, &context).await?;
        Ok(outcome)
    }

    /// Resume the loop after a breaker trip.
    pub fn resume(&self) -> Result<(), LoopError> {
        let mut breaker = self.load_breaker();
        breaker.paused = false;
        breaker.consecutive_failures = 0;
        self.save_breaker(&breaker)?;
        Ok(())
    }

    async fn run_cycle(
        &self,
        cycle: u32,
        context: &LoopContext,
    ) -> Result<CycleOutcome, LoopError> {
        let mut proposal = match self.generator.generate(context).await {
            Ok(proposal) => proposal,
            Err(ProposalGenerationError::NothingToPropose) => {
                return Ok(CycleOutcome {
                    cycle,
                    solidified: false,
                    rolled_back: false,
                    breaker_tripped: false,
                    fitness_score: None,
                });
            }
            Err(error) => return Err(LoopError::GenerationFailed(error.to_string())),
        };

        if let Some(suite) = &self.benchmark {
            let expected = MutationTarget::FilePath {
                path: suite.target.clone(),
            };
            if proposal.targets != vec![expected]
                || !matches!(proposal.payload.as_slice(), [PayloadChange::FileChange { path, .. }] if path == &suite.target)
            {
                return Err(LoopError::GenerationFailed(
                    "proposal must change only the benchmark target".to_string(),
                ));
            }
        }

        let machine = ProposalStateMachine::new(proposal.clone());
        let targets: Vec<String> = proposal
            .targets
            .iter()
            .map(|target| match target {
                MutationTarget::FilePath { path } => path.clone(),
                MutationTarget::ConfigKey { key } => format!("<config.yaml>:{key}"),
            })
            .collect();
        self.append_event(
            cycle,
            &proposal.id.to_string(),
            &EventKind::Proposed { targets },
        )?;

        // Validate (whitelist) → Validating.
        machine
            .begin_validation()
            .map_err(LoopError::WhitelistFailed)?;

        self.append_event(cycle, &proposal.id.to_string(), &EventKind::Validated)?;

        // The suite must score the *mutation*, not the untouched snapshot:
        // materialize the change set into a throwaway candidate copy and run
        // fitness there. The copy is removed when the cycle ends.
        let changes = self.synthesize_changes(&proposal);
        let candidate = self.stage_candidate(&changes)?;
        let report = if let Some(suite) = &self.benchmark {
            let evidence_dir = self.fitness_config.log_dir.join(proposal.id.to_string());
            std::fs::create_dir_all(&evidence_dir)?;
            std::fs::copy(
                self.workspace_snapshot.join(&suite.target),
                evidence_dir.join("baseline.yaml"),
            )?;
            std::fs::copy(
                candidate.root.join(&suite.target),
                evidence_dir.join("candidate.yaml"),
            )?;
            let result = run_benchmark(
                &self.fitness_config,
                suite,
                &proposal,
                &self.workspace_snapshot,
                &candidate.root,
            )
            .await?;
            proposal.fitness.baseline_score = result.baseline_score;
            proposal.fitness.minimum_score = result.baseline_score;
            let report = result.report.clone();
            self.append_event(
                cycle,
                &proposal.id.to_string(),
                &EventKind::BenchmarkEvaluated { result },
            )?;
            report
        } else {
            run_fitness(&self.fitness_config, &proposal, &candidate.root).await
        };
        self.append_event(
            cycle,
            &proposal.id.to_string(),
            &EventKind::FitnessEvaluated {
                report: report.clone(),
            },
        )?;

        let passed_gate = report.passed && report.score >= proposal.fitness.minimum_score;

        if !passed_gate {
            // Fitness regressed: record, roll back (no-op since nothing was
            // applied yet), update the breaker. The state machine stays in
            // Validating — there is no legal Validating→RolledBack edge, and
            // the machine is local (not persisted), so we simply drop it.
            self.append_event(
                cycle,
                &proposal.id.to_string(),
                &EventKind::RolledBack {
                    reason: format!(
                        "fitness gate: passed={} score={} minimum={}",
                        report.passed, report.score, proposal.fitness.minimum_score
                    ),
                },
            )?;
            let tripped = self.record_regression();
            if tripped {
                self.append_event(
                    cycle,
                    &proposal.id.to_string(),
                    &EventKind::BreakerTripped {
                        consecutive_failures: self.load_breaker().consecutive_failures,
                    },
                )?;
            }
            return Ok(CycleOutcome {
                cycle,
                solidified: false,
                rolled_back: true,
                breaker_tripped: tripped,
                fitness_score: Some(report.score),
            });
        }

        // Apply (fitness gate passes internally too).
        let applied = apply(&self.apply_config, &proposal, &report, &changes, None)?;
        machine
            .transit(ProposalState::Validating, ProposalState::Applied)
            .ok();
        self.append_event(
            cycle,
            &proposal.id.to_string(),
            &EventKind::Applied {
                backup_version: applied.backup_version,
            },
        )?;

        // Solidify: the mutation becomes the new baseline.
        machine
            .transit_typed::<tag::Applied, tag::Solidified>()
            .ok();
        self.append_event(cycle, &proposal.id.to_string(), &EventKind::Solidified)?;

        // Success resets the regression counter.
        let mut breaker = self.load_breaker();
        breaker.consecutive_failures = 0;
        self.save_breaker(&breaker)?;

        Ok(CycleOutcome {
            cycle,
            solidified: true,
            rolled_back: false,
            breaker_tripped: false,
            fitness_score: Some(report.score),
        })
    }

    /// Materialize the change set into a throwaway candidate copy of the
    /// workspace snapshot. The eval suite scores this copy, so the resulting
    /// report describes the mutation that would actually be applied — not
    /// the untouched baseline.
    fn stage_candidate(&self, changes: &[Change]) -> Result<CandidateSnapshot, LoopError> {
        let root =
            std::env::temp_dir().join(format!("goose-rsi-candidate-{}", uuid::Uuid::new_v4()));
        copy_tree(&self.workspace_snapshot, &root)?;
        match materialize_candidate(&root, changes) {
            Ok(()) => Ok(CandidateSnapshot { root }),
            Err(e) => {
                let _ = std::fs::remove_dir_all(&root);
                Err(e.into())
            }
        }
    }

    /// Convert the proposal's payload into apply-layer `Change` items.
    /// No content is synthesized at runtime — every byte comes from the
    /// proposal itself.
    fn synthesize_changes(&self, proposal: &MutationProposal) -> Vec<Change> {
        proposal
            .payload
            .iter()
            .map(|p| match p {
                PayloadChange::FileChange { path, content } => Change::FileChange {
                    path: path.clone(),
                    content: content.clone(),
                },
                PayloadChange::ConfigKeyChange { key, value } => Change::ConfigKeyChange {
                    key: key.clone(),
                    value: value.clone(),
                },
            })
            .collect()
    }

    fn record_regression(&self) -> bool {
        let mut breaker = self.load_breaker();
        breaker.consecutive_failures += 1;
        if breaker.consecutive_failures >= CircuitBreaker::TRIP_THRESHOLD {
            breaker.paused = true;
        }
        let tripped = breaker.paused;
        let _ = self.save_breaker(&breaker);
        tripped
    }

    fn acquire_lock(&self) -> Result<LoopLock, LoopError> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&self.lock_path)?;
        fs2::FileExt::try_lock_exclusive(&file).map_err(|_| LoopError::AlreadyRunning)?;
        Ok(LoopLock { _file: file })
    }

    pub fn load_breaker(&self) -> CircuitBreaker {
        std::fs::read_to_string(&self.breaker_path)
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or_default()
    }

    fn save_breaker(&self, breaker: &CircuitBreaker) -> Result<(), LoopError> {
        write_json_atomic(&self.breaker_path, breaker)
    }

    fn next_cycle_number(&self) -> u32 {
        self.load_state().cycle
    }

    fn increment_cycle(&self) -> u32 {
        let mut state = self.load_state();
        state.cycle += 1;
        let _ = write_json_atomic(&self.state_path, &state);
        state.cycle
    }

    fn load_state(&self) -> LoopState {
        std::fs::read_to_string(&self.state_path)
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or_default()
    }

    fn last_fitness_score(&self) -> Option<f64> {
        self.load_state().last_fitness
    }

    fn last_proposal_id(&self) -> Option<String> {
        self.load_state().last_proposal_id
    }

    fn append_event(
        &self,
        cycle: u32,
        proposal_id: &str,
        kind: &EventKind,
    ) -> Result<(), LoopError> {
        let event = EvolutionEvent {
            cycle,
            timestamp: SystemTime::now(),
            proposal_id: proposal_id.to_string(),
            kind: kind.clone(),
        };
        if let Some(parent) = self.log_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut line = serde_json::to_string(&event).map_err(LoopError::Serialization)?;
        line.push('\n');
        // Append-only: O_APPEND ensures the write is atomic and never
        // overwrites earlier events.
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;
        file.write_all(line.as_bytes())?;
        Ok(())
    }
}

/// Persisted cycle counter and last-result memory.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct LoopState {
    cycle: u32,
    last_fitness: Option<f64>,
    last_proposal_id: Option<String>,
}

/// Cross-process loop lock guard. The OS releases the flock on process death.
struct LoopLock {
    _file: std::fs::File,
}

/// Throwaway copy of the workspace snapshot with the change set materialized.
/// Removed when the cycle ends, whatever the outcome.
struct CandidateSnapshot {
    root: PathBuf,
}

impl Drop for CandidateSnapshot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn copy_tree(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let target = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum LoopError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("another evolution cycle is already running")]
    AlreadyRunning,
    #[error("proposal generation failed: {0}")]
    GenerationFailed(String),
    #[error("whitelist rejected the proposal")]
    WhitelistFailed(crate::state::BeginValidationError),
    #[error("apply failed: {0}")]
    ApplyFailed(#[from] apply::ApplyError),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), LoopError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let serialized = serde_json::to_string_pretty(value)?;
    let temp = path.with_extension("tmp");
    std::fs::write(&temp, serialized.as_bytes())?;
    std::fs::rename(&temp, path)?;
    Ok(())
}
