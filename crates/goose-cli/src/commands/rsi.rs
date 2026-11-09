use anyhow::{Context, Result};
use goose::config::paths::Paths;
use goose_rsi::apply::{list_backup_versions, rollback_to_version, ApplyConfig};
use goose_rsi::audit::{append_event, read_breaker, read_events, read_loop_state, render_history};
use goose_rsi::evolution::{EventKind, EvolutionEvent, EvolutionLoop, LoopContext};
use goose_rsi::fitness::{
    BenchmarkSplit, BenchmarkSuite, FitnessConfig, FitnessSuite, TestDepth, TestPhases,
};
use goose_rsi::proposal::MutationProposal;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// RSI subcommands for auditing and managing the evolution loop.
#[derive(clap::Subcommand)]
pub enum RsiCommand {
    /// Install a sample recipe and fixed benchmark without replacing existing files
    Init,
    /// Show current RSI status: circuit breaker, cycle count, backup versions
    Status,
    /// List evolution events from the append-only JSONL log
    History {
        /// Maximum number of events to show (newest first)
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Roll back the managed root to a backup version and append a RolledBack event
    Rollback {
        /// Backup version number to restore
        version: u32,
    },
    /// Run one evolution cycle using rsi/benchmark.json as the fitness gate
    Evolve,
}

pub async fn handle_rsi_command(command: &RsiCommand) -> Result<()> {
    let managed_root = Paths::config_dir();
    match command {
        RsiCommand::Init => {
            println!("{}", init_rsi(&managed_root)?);
        }
        RsiCommand::Status => {
            println!("{}", status_report(&managed_root));
        }
        RsiCommand::History { limit } => {
            println!("{}", history_report(&managed_root, *limit));
        }
        RsiCommand::Rollback { version } => {
            println!("{}", rollback_and_log(&managed_root, *version)?);
        }
        RsiCommand::Evolve => {
            let outcome = evolve_one_cycle(&managed_root).await?;
            println!("{outcome}");
        }
    }
    Ok(())
}

fn init_rsi(managed_root: &Path) -> Result<String> {
    use std::io::Write;

    let recipe = managed_root.join("recipes/search.yaml");
    let benchmark = managed_root.join("rsi/benchmark.json");
    anyhow::ensure!(
        std::fs::symlink_metadata(&recipe).is_err()
            && std::fs::symlink_metadata(&benchmark).is_err(),
        "RSI sample already exists; existing recipe or benchmark was not replaced"
    );
    std::fs::create_dir_all(recipe.parent().unwrap())?;
    std::fs::create_dir_all(benchmark.parent().unwrap())?;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&recipe)?
        .write_all(include_str!("../../examples/rsi/recipes/search.yaml").as_bytes())?;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&benchmark)?
        .write_all(include_str!("../../examples/rsi/rsi/benchmark.json").as_bytes())?;
    Ok(format!(
        "Installed sample RSI recipe at {} and benchmark at {}",
        recipe.display(),
        benchmark.display()
    ))
}

fn status_report(managed_root: &Path) -> String {
    let rsi_dir = managed_root.join("rsi");
    let apply_config = ApplyConfig::new(managed_root);

    let breaker = read_breaker(&rsi_dir);
    let snapshot = read_loop_state(&rsi_dir);
    let backups = list_backup_versions(&apply_config);
    let event_count = read_events(&rsi_dir.join("evolution.jsonl")).len();

    let mut report = String::from("RSI Status\n──────────────────────────────────────\n");
    report.push_str(&format!("  Managed root:  {}\n", managed_root.display()));
    report.push_str(&format!(
        "  Circuit breaker: {} (consecutive failures: {})\n",
        if breaker.paused { "PAUSED" } else { "healthy" },
        breaker.consecutive_failures
    ));
    report.push_str(&format!("  Cycle count:   {}\n", snapshot.cycle));
    match snapshot.last_fitness {
        Some(score) => report.push_str(&format!("  Last fitness:  {score:.4}\n")),
        None => report.push_str("  Last fitness:  (none)\n"),
    }
    let benchmark = std::fs::read(rsi_dir.join("benchmark.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<BenchmarkSuite>(&bytes).ok())
        .filter(|suite| suite.validate().is_ok());
    match benchmark {
        Some(suite) => report.push_str(&format!(
            "  Benchmark:     {} ({} tasks)\n",
            suite.target,
            suite.tasks.len()
        )),
        None => report.push_str("  Benchmark:     missing or invalid\n"),
    }
    if backups.is_empty() {
        report.push_str("  Backups:       (none)\n");
    } else {
        report.push_str(&format!("  Backups:       {backups:?}\n"));
    }
    report.push_str(&format!("  Events logged: {event_count}"));
    report
}

fn history_report(managed_root: &Path, limit: usize) -> String {
    let log_path = managed_root.join("rsi/evolution.jsonl");
    if !log_path.exists() {
        return format!("No evolution log found at {}", log_path.display());
    }
    let events = read_events(&log_path);
    if events.is_empty() {
        return "Evolution log is empty or unreadable.".to_string();
    }
    render_history(&events, limit)
}

fn rollback_and_log(managed_root: &Path, version: u32) -> Result<String> {
    let apply_config = ApplyConfig::new(managed_root);
    let rolled = rollback_to_version(&apply_config, version)
        .with_context(|| format!("rollback to version {version} failed"))?;

    append_event(
        &managed_root.join("rsi/evolution.jsonl"),
        &EvolutionEvent {
            cycle: 0,
            timestamp: std::time::SystemTime::now(),
            proposal_id: String::new(),
            kind: EventKind::RolledBack {
                reason: format!("manual rollback to version {version}"),
            },
        },
    )?;

    Ok(format!(
        "Rolled back managed root to backup version {} (restored {} targets).",
        rolled.backup_version,
        rolled.restored_targets.len()
    ))
}

// ── Issue #10: goose rsi evolve + LLM proposal generator ──────────────────

/// Run one evolution cycle on the user's managed root.
async fn evolve_one_cycle(managed_root: &Path) -> Result<String> {
    let benchmark_path = managed_root.join("rsi/benchmark.json");
    let benchmark: BenchmarkSuite =
        serde_json::from_slice(&std::fs::read(&benchmark_path).with_context(|| {
            format!("RSI benchmark is required at {}", benchmark_path.display())
        })?)?;
    benchmark.validate().map_err(anyhow::Error::msg)?;
    let target_path = managed_root.join(&benchmark.target);
    let current_recipe = std::fs::read_to_string(&target_path)
        .with_context(|| format!("benchmark target is missing: {}", target_path.display()))?;
    goose::recipe::validate_recipe::validate_recipe_template_from_content(
        &current_recipe,
        target_path.parent().map(|path| path.display().to_string()),
    )?;

    let apply_config = ApplyConfig::new(managed_root);
    let snapshot_dir = tempfile::tempdir()?;
    for name in ["recipes", "prompts"] {
        let source = managed_root.join(name);
        if source.is_dir() {
            copy_evaluation_inputs(&source, &snapshot_dir.path().join(name))?;
        }
    }

    let fitness_config = default_fitness_config(managed_root);
    let generator = Arc::new(LlmProposalGenerator {
        managed_root: managed_root.to_path_buf(),
        benchmark: benchmark.clone(),
    });

    let loop_ = EvolutionLoop::new(apply_config, fitness_config, generator, snapshot_dir.path())
        .with_benchmark(benchmark);

    let outcome = loop_.tick().await?;
    let mut summary = format!(
        "cycle {}: solidified={}, rolled_back={}, breaker_tripped={}, fitness_score={:?}",
        outcome.cycle,
        outcome.solidified,
        outcome.rolled_back,
        outcome.breaker_tripped,
        outcome.fitness_score
    );
    if outcome.fitness_score.is_some() {
        if let Some(result) = read_events(&managed_root.join("rsi/evolution.jsonl"))
            .into_iter()
            .rev()
            .find_map(|event| match event.kind {
                EventKind::BenchmarkEvaluated { result } if event.cycle == outcome.cycle => {
                    Some(result)
                }
                _ => None,
            })
        {
            summary.push_str(&format!(
                "\ntraining: {:.3} → {:.3}; holdout: {:.3} → {:.3}",
                result.baseline_score,
                result.candidate_score,
                result.baseline_holdout_score,
                result.candidate_holdout_score
            ));
        }
    }
    Ok(summary)
}

fn copy_evaluation_inputs(source: &Path, target: &Path) -> Result<()> {
    std::fs::create_dir_all(target)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let dest = target.join(entry.file_name());
        if kind.is_dir() {
            copy_evaluation_inputs(&entry.path(), &dest)?;
        } else if kind.is_file() {
            std::fs::copy(entry.path(), dest)?;
        }
    }
    Ok(())
}

fn default_fitness_config(managed_root: &Path) -> FitnessConfig {
    FitnessConfig {
        suite: FitnessSuite {
            phases: TestPhases::Basic,
            depth: TestDepth::Quick,
        },
        command: None,
        self_test_recipe: PathBuf::from("goose-self-test.yaml"),
        timeout: std::time::Duration::from_secs(300),
        log_dir: managed_root.join("rsi/fitness-logs"),
    }
}

/// LLM-backed proposal generator for one benchmark target.
pub struct LlmProposalGenerator {
    managed_root: PathBuf,
    benchmark: BenchmarkSuite,
}

#[async_trait::async_trait]
impl goose_rsi::evolution::ProposalGenerator for LlmProposalGenerator {
    async fn generate(
        &self,
        _context: &LoopContext,
    ) -> Result<MutationProposal, goose_rsi::evolution::ProposalGenerationError> {
        generate_with_provider(&self.managed_root, &self.benchmark).await
    }
}

async fn generate_with_provider(
    managed_root: &Path,
    benchmark: &BenchmarkSuite,
) -> Result<MutationProposal, goose_rsi::evolution::ProposalGenerationError> {
    use goose::config::Config;
    use goose_rsi::proposal::{FitnessExpectation, MutationTarget, PayloadChange, ProposalSource};

    use goose_rsi::evolution::ProposalGenerationError;

    let target_path = managed_root.join(&benchmark.target);
    let current_content = std::fs::read_to_string(&target_path)
        .map_err(|error| ProposalGenerationError::Failed(error.to_string()))?;
    if current_content.len() > 32_000 {
        return Err(ProposalGenerationError::Failed(
            "recipe exceeds the proposal context limit".to_string(),
        ));
    }

    let config = Config::global();
    let provider_name = config
        .get_goose_provider()
        .map_err(|error| ProposalGenerationError::Failed(error.to_string()))?;
    let model_name = config
        .get_goose_model()
        .map_err(|error| ProposalGenerationError::Failed(error.to_string()))?;
    let provider = goose::providers::create(&provider_name, vec![])
        .await
        .map_err(|error| ProposalGenerationError::Failed(error.to_string()))?;

    let model_config = goose_providers::model::ModelConfig {
        model_name: model_name.clone(),
        context_limit: None,
        temperature: Some(0.2),
        max_tokens: Some(8192),
        toolshim: false,
        toolshim_model: None,
        request_params: None,
        reasoning: None,
        supports_vision: None,
        request_headers: None,
    };
    provider
        .apply_model_selection(&model_config)
        .await
        .map_err(|error| ProposalGenerationError::Failed(error.to_string()))?;

    let system_prompt = build_system_prompt(&benchmark.target);
    let user_message = goose_providers::conversation::message::Message::user()
        .with_text(build_user_prompt(benchmark, &current_content));

    let (response, _usage) = provider
        .complete(&model_config, &system_prompt, &[user_message], &[])
        .await
        .map_err(|error| ProposalGenerationError::Failed(error.to_string()))?;

    let text = response.content.iter().find_map(|block| block.as_text());
    let text = match text {
        Some(t) => t,
        None => {
            return Err(ProposalGenerationError::Failed(
                "provider returned no text proposal".to_string(),
            ))
        }
    };

    match parse_proposal_response(text) {
        Some((path, content)) if path == benchmark.target && content != current_content => {
            validate_recipe_change(&current_content, &content, &target_path)
                .map_err(|error| ProposalGenerationError::Failed(error.to_string()))?;
            let target = MutationTarget::FilePath { path: path.clone() };
            Ok(MutationProposal::new(
                "llm-generated",
                ProposalSource::Session,
                vec![target.clone()],
                FitnessExpectation {
                    baseline_score: 0.0,
                    minimum_score: 1.0,
                },
                vec![PayloadChange::FileChange { path, content }],
            ))
        }
        Some((_, content)) if content == current_content => {
            Err(ProposalGenerationError::NothingToPropose)
        }
        _ => Err(ProposalGenerationError::Failed(
            "model proposal is invalid or targets a different recipe".to_string(),
        )),
    }
}

fn build_system_prompt(target: &str) -> String {
    format!(
        "You are an RSI (Recursive Self-Improvement) proposal generator.\n\
         Improve the existing goose recipe at {target}.\n\
         Change only its prompt or instructions fields. Preserve every other field.\n\
         Respond with ONLY a JSON object, no markdown fences:\n\
         {{\"file_path\": \"{target}\", \"content\": \"<full file content>\"}}\n\
         Return valid goose recipe YAML. Do not change extensions, settings, parameters, or permissions."
    )
}

fn build_user_prompt(benchmark: &BenchmarkSuite, current_content: &str) -> String {
    let training = benchmark
        .tasks
        .iter()
        .filter(|task| task.split == BenchmarkSplit::Training)
        .map(|task| {
            format!(
                "{}: {} {:?}",
                task.id, task.command.program, task.command.args
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "Current recipe at {}:\n```yaml\n{}\n```\n\nTraining tasks and checks:\n{}\n\nPropose a focused improvement to the recipe text for these tasks. The proposal will be checked against additional hidden tasks.",
        benchmark.target, current_content, training
    )
}

fn validate_recipe_change(before: &str, after: &str, target_path: &Path) -> Result<()> {
    let mut previous: serde_yaml::Mapping = serde_yaml::from_str(before)?;
    let mut proposed: serde_yaml::Mapping = serde_yaml::from_str(after)?;
    let prompt_key = serde_yaml::Value::String("prompt".to_string());
    let instructions_key = serde_yaml::Value::String("instructions".to_string());
    let previous_prompt = previous.remove(&prompt_key);
    let proposed_prompt = proposed.remove(&prompt_key);
    let previous_instructions = previous.remove(&instructions_key);
    let proposed_instructions = proposed.remove(&instructions_key);
    anyhow::ensure!(
        previous == proposed
            && (previous_prompt != proposed_prompt
                || previous_instructions != proposed_instructions),
        "proposal may only change prompt or instructions"
    );
    goose::recipe::validate_recipe::validate_recipe_template_from_content(
        after,
        target_path.parent().map(|path| path.display().to_string()),
    )?;
    Ok(())
}

fn parse_proposal_response(text: &str) -> Option<(String, String)> {
    let json_start = text.find('{')?;
    let json_end = text.rfind('}')?;
    let json_str = text.get(json_start..=json_end)?;
    let parsed: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let file_path = parsed.get("file_path")?.as_str()?.to_string();
    let content = parsed.get("content")?.as_str()?.to_string();
    if file_path.is_empty() || content.is_empty() {
        return None;
    }
    Some((file_path, content))
}

#[cfg(test)]
mod tests {
    use super::*;
    use goose_rsi::apply::{apply, Change};
    use goose_rsi::fitness::{BenchmarkTask, FitnessCommand};
    use goose_rsi::proposal::{FitnessExpectation, MutationTarget, PayloadChange, ProposalSource};
    use std::time::{Duration, SystemTime};

    fn seed(root: &Path) -> MutationProposal {
        std::fs::create_dir_all(root.join("recipes")).unwrap();
        std::fs::write(root.join("recipes/target.yaml"), "version: 1\n").unwrap();
        MutationProposal::new(
            "cli test",
            ProposalSource::Scheduled,
            vec![MutationTarget::FilePath {
                path: "recipes/target.yaml".to_string(),
            }],
            FitnessExpectation {
                baseline_score: 0.0,
                minimum_score: 0.0,
            },
            vec![PayloadChange::FileChange {
                path: "recipes/target.yaml".to_string(),
                content: "version: 2\n".to_string(),
            }],
        )
    }

    fn passing_report(proposal: &MutationProposal) -> goose_rsi::fitness::FitnessReport {
        goose_rsi::fitness::FitnessReport {
            proposal_id: proposal.id.to_string(),
            passed: true,
            score: 1.0,
            failed_phases: Vec::new(),
            duration: Duration::from_millis(1),
            raw_log_ref: std::path::PathBuf::from("/dev/null"),
            timestamp: SystemTime::UNIX_EPOCH,
        }
    }

    #[test]
    fn history_reports_truncated_log_gracefully() {
        let root = tempfile::tempdir().unwrap();
        let log = root.path().join("rsi/evolution.jsonl");
        std::fs::create_dir_all(log.parent().unwrap()).unwrap();
        std::fs::write(&log, "{\"cycle\": 1, \"broken").unwrap();
        assert_eq!(
            history_report(root.path(), 10),
            "Evolution log is empty or unreadable."
        );
        assert_eq!(
            history_report(&root.path().join("missing"), 10),
            format!(
                "No evolution log found at {}",
                root.path().join("missing/rsi/evolution.jsonl").display()
            )
        );
    }

    #[test]
    fn rollback_restores_and_appends_rolled_back_event() {
        let root = tempfile::tempdir().unwrap();
        let proposal = seed(root.path());
        let config = ApplyConfig::new(root.path());
        apply(
            &config,
            &proposal,
            &passing_report(&proposal),
            &[Change::FileChange {
                path: "recipes/target.yaml".to_string(),
                content: "version: 2\n".to_string(),
            }],
            None,
        )
        .unwrap();

        let output = rollback_and_log(root.path(), 1).unwrap();
        assert!(output.contains("backup version 1"));

        let events = read_events(&root.path().join("rsi/evolution.jsonl"));
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].kind, EventKind::RolledBack { .. }));
        assert_eq!(
            std::fs::read_to_string(root.path().join("recipes/target.yaml")).unwrap(),
            "version: 1\n"
        );
    }

    #[test]
    fn status_reports_healthy_defaults() {
        let root = tempfile::tempdir().unwrap();
        let report = status_report(root.path());
        assert!(report.contains("healthy"));
        assert!(report.contains("(none)"));
        assert!(report.contains("Cycle count:   0"));
    }

    #[test]
    fn proposal_uses_current_recipe_and_hides_holdout_tasks() {
        let task = |id: &str, split, answer: &str| BenchmarkTask {
            id: id.to_string(),
            split,
            command: FitnessCommand {
                program: "sh".to_string(),
                args: vec![answer.to_string()],
            },
        };
        let benchmark = BenchmarkSuite {
            target: "recipes/search.yaml".to_string(),
            tasks: vec![
                task("train", BenchmarkSplit::Training, "visible answer"),
                task("holdout", BenchmarkSplit::Holdout, "hidden answer"),
            ],
        };
        let prompt = build_user_prompt(&benchmark, "instructions: current recipe");
        assert!(prompt.contains("current recipe"));
        assert!(prompt.contains("visible answer"));
        assert!(!prompt.contains("hidden answer"));
    }

    #[test]
    fn recipe_mutation_is_limited_to_prompt_and_instructions() {
        let before = "version: 1.0.0\ntitle: Search\ndescription: Search docs\ninstructions: old\n";
        let safe =
            "version: 1.0.0\ntitle: Search\ndescription: Search docs\ninstructions: improved\n";
        let unsafe_change =
            "version: 1.0.0\ntitle: Changed\ndescription: Search docs\ninstructions: improved\n";
        let path = Path::new("recipes/search.yaml");
        assert!(validate_recipe_change(before, safe, path).is_ok());
        assert!(validate_recipe_change(before, unsafe_change, path).is_err());
    }

    #[test]
    fn init_installs_a_valid_fixed_benchmark_without_overwriting() {
        let root = tempfile::tempdir().unwrap();
        init_rsi(root.path()).unwrap();
        let recipe_path = root.path().join("recipes/search.yaml");
        let benchmark_path = root.path().join("rsi/benchmark.json");
        let recipe = std::fs::read_to_string(&recipe_path).unwrap();
        goose::recipe::validate_recipe::validate_recipe_template_from_content(
            &recipe,
            recipe_path.parent().map(|path| path.display().to_string()),
        )
        .unwrap();
        let benchmark: BenchmarkSuite =
            serde_json::from_slice(&std::fs::read(&benchmark_path).unwrap()).unwrap();
        benchmark.validate().unwrap();
        assert_eq!(benchmark.tasks.len(), 4);
        assert!(init_rsi(root.path()).is_err());
        assert_eq!(std::fs::read_to_string(recipe_path).unwrap(), recipe);
    }
}
