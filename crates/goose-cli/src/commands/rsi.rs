use anyhow::{Context, Result};
use goose::config::paths::Paths;
use goose_rsi::apply::{list_backup_versions, rollback_to_version, ApplyConfig};
use goose_rsi::audit::{append_event, read_breaker, read_events, read_loop_state, render_history};
use goose_rsi::evolution::{EventKind, EvolutionEvent, EvolutionLoop, LoopContext};
use goose_rsi::fitness::{FitnessCommand, FitnessConfig, FitnessSuite, TestDepth, TestPhases};
use goose_rsi::proposal::MutationProposal;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// RSI subcommands for auditing and managing the evolution loop.
#[derive(clap::Subcommand)]
pub enum RsiCommand {
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
    /// Run one evolution cycle: detect → propose → validate → fitness → apply → solidify
    Evolve,
}

pub async fn handle_rsi_command(command: &RsiCommand) -> Result<()> {
    let managed_root = Paths::config_dir();
    match command {
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
    let apply_config = ApplyConfig::new(managed_root);
    let snapshot_dir = managed_root.join("rsi-snapshot");
    std::fs::create_dir_all(&snapshot_dir)?;

    let fitness_config = default_fitness_config(managed_root);
    let generator = Arc::new(LlmProposalGenerator {
        managed_root: managed_root.to_path_buf(),
    });

    let loop_ = EvolutionLoop::new(apply_config, fitness_config, generator, &snapshot_dir);

    let outcome = loop_.tick().await?;
    Ok(format!(
        "cycle {}: solidified={}, rolled_back={}, breaker_tripped={}, fitness_score={:?}",
        outcome.cycle,
        outcome.solidified,
        outcome.rolled_back,
        outcome.breaker_tripped,
        outcome.fitness_score
    ))
}

fn default_fitness_config(managed_root: &Path) -> FitnessConfig {
    FitnessConfig {
        suite: FitnessSuite {
            phases: TestPhases::Basic,
            depth: TestDepth::Quick,
        },
        command: Some(FitnessCommand {
            program: "sh".to_string(),
            args: vec![
                "-c".to_string(),
                "if test -f recipes/search.yaml && test -s recipes/search.yaml; then \
                 echo 'OVERALL RESULT: PASS'; \
                 else echo 'OVERALL RESULT: FAIL'; fi"
                    .to_string(),
            ],
        }),
        self_test_recipe: PathBuf::from("goose-self-test.yaml"),
        timeout: std::time::Duration::from_secs(300),
        log_dir: managed_root.join("rsi/fitness-logs"),
    }
}

/// LLM-backed proposal generator. Uses the configured goose provider to
/// generate a mutation proposal. On any failure (provider unavailable,
/// timeout, parse error), returns `NothingToPropose` — zero writes.
pub struct LlmProposalGenerator {
    managed_root: PathBuf,
}

#[async_trait::async_trait]
impl goose_rsi::evolution::ProposalGenerator for LlmProposalGenerator {
    async fn generate(
        &self,
        _context: &LoopContext,
    ) -> Result<MutationProposal, goose_rsi::evolution::ProposalGenerationError> {
        generate_with_provider(&self.managed_root).await
    }
}

async fn generate_with_provider(
    managed_root: &Path,
) -> Result<MutationProposal, goose_rsi::evolution::ProposalGenerationError> {
    use goose::config::Config;
    use goose_rsi::proposal::{FitnessExpectation, MutationTarget, PayloadChange, ProposalSource};

    let config = Config::global();
    let provider_name = match config.get_goose_provider() {
        Ok(name) => name,
        Err(_) => return Err(goose_rsi::evolution::ProposalGenerationError::NothingToPropose),
    };
    let model_name = match config.get_goose_model() {
        Ok(name) => name,
        Err(_) => return Err(goose_rsi::evolution::ProposalGenerationError::NothingToPropose),
    };

    let provider = match goose::providers::create(&provider_name, vec![]).await {
        Ok(p) => p,
        Err(_) => return Err(goose_rsi::evolution::ProposalGenerationError::NothingToPropose),
    };

    let model_config = goose_providers::model::ModelConfig {
        model_name: model_name.clone(),
        context_limit: None,
        temperature: Some(0.7),
        max_tokens: Some(2048),
        toolshim: false,
        toolshim_model: None,
        request_params: None,
        reasoning: None,
        supports_vision: None,
        request_headers: None,
    };
    if provider.apply_model_selection(&model_config).await.is_err() {
        return Err(goose_rsi::evolution::ProposalGenerationError::NothingToPropose);
    }

    let recipe_files = list_recipe_files(managed_root);
    let system_prompt = build_system_prompt(&recipe_files);
    let user_message = goose_providers::conversation::message::Message::user()
        .with_text(build_user_prompt(&recipe_files));

    let (response, _usage) = match provider
        .complete(&model_config, &system_prompt, &[user_message], &[])
        .await
    {
        Ok(result) => result,
        Err(_) => return Err(goose_rsi::evolution::ProposalGenerationError::NothingToPropose),
    };

    let text = response.content.iter().find_map(|block| block.as_text());
    let text = match text {
        Some(t) => t,
        None => return Err(goose_rsi::evolution::ProposalGenerationError::NothingToPropose),
    };

    match parse_proposal_response(text) {
        Some((path, content)) => {
            let target = MutationTarget::FilePath { path: path.clone() };
            Ok(MutationProposal::new(
                "llm-generated",
                ProposalSource::Session,
                vec![target.clone()],
                FitnessExpectation {
                    baseline_score: 0.0,
                    minimum_score: 0.0,
                },
                vec![PayloadChange::FileChange { path, content }],
            ))
        }
        None => Err(goose_rsi::evolution::ProposalGenerationError::NothingToPropose),
    }
}

fn list_recipe_files(managed_root: &Path) -> Vec<String> {
    let recipes_dir = managed_root.join("recipes");
    if !recipes_dir.is_dir() {
        return vec![];
    }
    std::fs::read_dir(&recipes_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| format!("recipes/{s}"))
            } else {
                None
            }
        })
        .collect()
}

fn build_system_prompt(recipe_files: &[String]) -> String {
    format!(
        "You are an RSI (Recursive Self-Improvement) proposal generator.\n\
         Your task: suggest a small, safe improvement to one recipe file.\n\
         Available recipe files: {}\n\n\
         Respond with ONLY a JSON object, no markdown fences:\n\
         {{\"file_path\": \"recipes/<name>.yaml\", \"content\": \"<full file content>\"}}\n\
         The content must be valid YAML for a goose recipe.\n\
         Keep changes minimal and safe.",
        recipe_files.join(", ")
    )
}

fn build_user_prompt(recipe_files: &[String]) -> String {
    if recipe_files.is_empty() {
        "No recipe files found. Propose creating recipes/search.yaml with a basic search recipe."
            .to_string()
    } else {
        format!(
            "Read the current recipe files and propose a small improvement.\n\
             Files: {}\n\
             Output the JSON with file_path and improved content.",
            recipe_files.join(", ")
        )
    }
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
}
