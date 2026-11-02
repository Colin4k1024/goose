//! Evolution-log and RSI-state auditing (issue #5).
//!
//! The evolution log is append-only JSONL: one [`EvolutionEvent`] per line,
//! written with `O_APPEND` so an earlier line is never rewritten. Readers are
//! deliberately tolerant — a crash leaves at most one truncated final line,
//! and that line must be skipped without losing earlier events or panicking.
//!
//! The CLI (`goose rsi status|history|rollback`) is a thin printer over these
//! functions, so the data invariants are covered by tests against the code
//! the binary actually runs.

use crate::evolution::{CircuitBreaker, EvolutionEvent};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// Read the evolution log, skipping empty, corrupt or truncated lines.
///
/// A missing file is not an error: it means no cycle has run yet.
pub fn read_events(log_path: &Path) -> Vec<EvolutionEvent> {
    let Ok(file) = std::fs::File::open(log_path) else {
        return Vec::new();
    };
    BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<EvolutionEvent>(&line).ok())
        .collect()
}

/// Append one event to the log. Never truncates or rewrites earlier lines.
pub fn append_event(log_path: &Path, event: &EvolutionEvent) -> std::io::Result<()> {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut line = serde_json::to_string(event)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    line.push('\n');
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    file.write_all(line.as_bytes())
}

/// One-line rendering of an event for `goose rsi history`.
pub fn render_event(event: &EvolutionEvent) -> String {
    let kind = match &event.kind {
        crate::evolution::EventKind::Proposed { targets } => {
            format!("Proposed (targets: {})", targets.join(", "))
        }
        crate::evolution::EventKind::Validated => "Validated".to_string(),
        crate::evolution::EventKind::FitnessEvaluated { report } => format!(
            "FitnessEvaluated (score: {:.4}, passed: {})",
            report.score, report.passed
        ),
        crate::evolution::EventKind::Applied { backup_version } => {
            format!("Applied (backup v{backup_version})")
        }
        crate::evolution::EventKind::Solidified => "Solidified".to_string(),
        crate::evolution::EventKind::RolledBack { reason } => {
            format!("RolledBack ({reason})")
        }
        crate::evolution::EventKind::BreakerTripped {
            consecutive_failures,
        } => format!("BreakerTripped (failures: {consecutive_failures})"),
        crate::evolution::EventKind::SkippedConcurrent => "SkippedConcurrent".to_string(),
        crate::evolution::EventKind::SkippedPaused => "SkippedPaused".to_string(),
    };
    format!("  [cycle {}] {} — {}", event.cycle, event.proposal_id, kind)
}

/// Render the newest `limit` events, one per line (newest first).
pub fn render_history(events: &[EvolutionEvent], limit: usize) -> String {
    let start = events.len().saturating_sub(limit);
    events[start..]
        .iter()
        .rev()
        .map(render_event)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Circuit-breaker state persisted by the evolution loop, or the default
/// (`healthy`) when the file is missing or unreadable.
pub fn read_breaker(rsi_dir: &Path) -> CircuitBreaker {
    std::fs::read_to_string(rsi_dir.join("circuit-breaker.json"))
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

/// Persisted loop progress (`loop-state.json`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LoopSnapshot {
    pub cycle: u32,
    pub last_fitness: Option<f64>,
}

pub fn read_loop_state(rsi_dir: &Path) -> LoopSnapshot {
    std::fs::read_to_string(rsi_dir.join("loop-state.json"))
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::EventKind;
    use std::time::SystemTime;

    fn event(cycle: u32, kind: EventKind) -> EvolutionEvent {
        EvolutionEvent {
            cycle,
            timestamp: SystemTime::UNIX_EPOCH,
            proposal_id: format!("prop-{cycle}"),
            kind,
        }
    }

    #[test]
    fn append_never_rewrites_existing_lines() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("rsi/evolution.jsonl");
        append_event(&log, &event(1, EventKind::Validated)).unwrap();
        let first = std::fs::read_to_string(&log).unwrap();
        append_event(&log, &event(2, EventKind::Solidified)).unwrap();
        let both = std::fs::read_to_string(&log).unwrap();
        assert!(both.starts_with(&first), "earlier bytes must be untouched");
        assert_eq!(read_events(&log).len(), 2);
    }

    #[test]
    fn render_history_is_newest_first_and_limited() {
        let events = vec![
            event(1, EventKind::Validated),
            event(2, EventKind::Solidified),
            event(3, EventKind::SkippedPaused),
        ];
        let rendered = render_history(&events, 2);
        let lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("cycle 3"));
        assert!(lines[1].contains("cycle 2"));
    }
}
