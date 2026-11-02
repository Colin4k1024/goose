use anyhow::{Context, Result};
use goose::config::paths::Paths;
use goose_rsi::apply::{list_backup_versions, rollback_to_version, ApplyConfig};
use goose_rsi::evolution::{EventKind, EvolutionEvent};
use std::io::{BufRead, BufReader};

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
}

pub fn handle_rsi_command(command: &RsiCommand) -> Result<()> {
    let managed_root = Paths::config_dir();
    match command {
        RsiCommand::Status => handle_rsi_status(&managed_root),
        RsiCommand::History { limit } => handle_rsi_history(&managed_root, *limit),
        RsiCommand::Rollback { version } => handle_rsi_rollback(&managed_root, *version),
    }
}

fn handle_rsi_status(managed_root: &std::path::Path) -> Result<()> {
    let rsi_dir = managed_root.join("rsi");
    let apply_config = ApplyConfig::new(managed_root);

    let breaker = read_breaker(&rsi_dir);
    let loop_state = read_loop_state(&rsi_dir);
    let backups = list_backup_versions(&apply_config);

    println!("RSI Status");
    println!("──────────────────────────────────────");
    println!("  Managed root:  {}", managed_root.display());
    println!(
        "  Circuit breaker: {} (consecutive failures: {})",
        if breaker.paused { "PAUSED" } else { "healthy" },
        breaker.consecutive_failures
    );
    println!("  Cycle count:   {}", loop_state.0);
    if let Some(score) = loop_state.1 {
        println!("  Last fitness:  {:.4}", score);
    } else {
        println!("  Last fitness:  (none)");
    }
    if backups.is_empty() {
        println!("  Backups:       (none)");
    } else {
        println!("  Backups:       {:?}", backups);
    }

    let log_path = rsi_dir.join("evolution.jsonl");
    if log_path.exists() {
        let count = count_log_events(&log_path);
        println!("  Events logged: {}", count);
    } else {
        println!("  Events logged: 0 (no log file)");
    }

    Ok(())
}

fn handle_rsi_history(managed_root: &std::path::Path, limit: usize) -> Result<()> {
    let log_path = managed_root.join("rsi/evolution.jsonl");
    if !log_path.exists() {
        println!("No evolution log found at {}", log_path.display());
        return Ok(());
    }

    let events = read_events_graceful(&log_path);
    if events.is_empty() {
        println!("Evolution log is empty or unreadable.");
        return Ok(());
    }

    let start = events.len().saturating_sub(limit);
    for event in events[start..].iter().rev() {
        print_event(event);
    }

    Ok(())
}

fn handle_rsi_rollback(managed_root: &std::path::Path, version: u32) -> Result<()> {
    let apply_config = ApplyConfig::new(managed_root);

    let rolled = rollback_to_version(&apply_config, version)
        .with_context(|| format!("rollback to version {} failed", version))?;

    println!(
        "Rolled back managed root to backup version {} (restored {} targets).",
        rolled.backup_version,
        rolled.restored_targets.len()
    );

    let log_path = managed_root.join("rsi/evolution.jsonl");
    append_rolled_back_event(&log_path, version)?;

    Ok(())
}

fn append_rolled_back_event(log_path: &std::path::Path, version: u32) -> Result<()> {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let event = EvolutionEvent {
        cycle: 0,
        timestamp: std::time::SystemTime::now(),
        proposal_id: String::new(),
        kind: EventKind::RolledBack {
            reason: format!("manual rollback to version {}", version),
        },
    };
    let mut line = serde_json::to_string(&event)?;
    line.push('\n');
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    file.write_all(line.as_bytes())?;
    Ok(())
}

fn print_event(event: &EvolutionEvent) {
    let kind_str = match &event.kind {
        EventKind::Proposed { targets } => {
            format!("Proposed (targets: {})", targets.join(", "))
        }
        EventKind::Validated => "Validated".to_string(),
        EventKind::FitnessEvaluated { report } => {
            format!(
                "FitnessEvaluated (score: {:.4}, passed: {})",
                report.score, report.passed
            )
        }
        EventKind::Applied { backup_version } => {
            format!("Applied (backup v{})", backup_version)
        }
        EventKind::Solidified => "Solidified".to_string(),
        EventKind::RolledBack { reason } => {
            format!("RolledBack ({})", reason)
        }
        EventKind::BreakerTripped {
            consecutive_failures,
        } => {
            format!("BreakerTripped (failures: {})", consecutive_failures)
        }
        EventKind::SkippedConcurrent => "SkippedConcurrent".to_string(),
        EventKind::SkippedPaused => "SkippedPaused".to_string(),
    };
    println!(
        "  [cycle {}] {} — {}",
        event.cycle, event.proposal_id, kind_str
    );
}

struct BreakerStatus {
    paused: bool,
    consecutive_failures: u32,
}

fn read_breaker(rsi_dir: &std::path::Path) -> BreakerStatus {
    let path = rsi_dir.join("circuit-breaker.json");
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .map(|v| BreakerStatus {
            paused: v.get("paused").and_then(|p| p.as_bool()).unwrap_or(false),
            consecutive_failures: v
                .get("consecutive_failures")
                .and_then(|f| f.as_u64())
                .unwrap_or(0) as u32,
        })
        .unwrap_or(BreakerStatus {
            paused: false,
            consecutive_failures: 0,
        })
}

fn read_loop_state(rsi_dir: &std::path::Path) -> (u32, Option<f64>) {
    let path = rsi_dir.join("loop-state.json");
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .map(|v| {
            let cycle = v.get("cycle").and_then(|c| c.as_u64()).unwrap_or(0) as u32;
            let fitness = v.get("last_fitness").and_then(|f| f.as_f64());
            (cycle, fitness)
        })
        .unwrap_or((0, None))
}

fn count_log_events(log_path: &std::path::Path) -> usize {
    read_events_graceful(log_path).len()
}

/// Read events from the JSONL log, skipping any corrupt or truncated lines.
fn read_events_graceful(log_path: &std::path::Path) -> Vec<EvolutionEvent> {
    let file = match std::fs::File::open(log_path) {
        Ok(file) => file,
        Err(_) => return Vec::new(),
    };
    let reader = BufReader::new(file);
    reader
        .lines()
        .map_while(std::result::Result::ok)
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<EvolutionEvent>(&line).ok())
        .collect()
}
