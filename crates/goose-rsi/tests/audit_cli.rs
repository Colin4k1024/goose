//! RSI audit CLI acceptance tests (issue #5).
//!
//! Data invariants:
//!
//! 1. The evolution log is append-only; a truncated or corrupt last line is
//!    skipped without affecting earlier events;
//! 2. `rollback` restores the managed root and appends a RolledBack event;
//! 3. A corrupt/truncated log file degrades gracefully — no panic.

use goose_rsi::apply::{apply, rollback_to_version, ApplyConfig, Change};
use goose_rsi::evolution::{EventKind, EvolutionEvent};
use goose_rsi::fitness::FitnessReport;
use goose_rsi::proposal::{FitnessExpectation, MutationProposal, MutationTarget, ProposalSource};
use std::io::Write;
use std::path::Path;
use std::time::{Duration, SystemTime};

fn proposal_with_score(minimum: f64) -> MutationProposal {
    MutationProposal::new(
        "audit test",
        ProposalSource::Scheduled,
        vec![MutationTarget::FilePath {
            path: "recipes/target.yaml".to_string(),
        }],
        FitnessExpectation {
            baseline_score: 0.0,
            minimum_score: minimum,
        },
    )
}

fn passing_report(proposal: &MutationProposal, score: f64) -> FitnessReport {
    FitnessReport {
        proposal_id: proposal.id.to_string(),
        passed: score >= proposal.fitness.minimum_score,
        score,
        failed_phases: Vec::new(),
        duration: Duration::from_millis(1),
        raw_log_ref: std::path::PathBuf::from("/dev/null"),
        timestamp: SystemTime::UNIX_EPOCH,
    }
}

fn seed(root: &Path) {
    std::fs::create_dir_all(root.join("recipes")).unwrap();
    std::fs::write(
        root.join("recipes/target.yaml"),
        "version: 1\ntitle: baseline\n",
    )
    .unwrap();
}

fn write_event(log_path: &Path, event: &EvolutionEvent) {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut line = serde_json::to_string(event).unwrap();
    line.push('\n');
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .unwrap();
    file.write_all(line.as_bytes()).unwrap();
}

fn read_events(log_path: &Path) -> Vec<EvolutionEvent> {
    use std::io::{BufRead, BufReader};
    let file = match std::fs::File::open(log_path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    BufReader::new(file)
        .lines()
        .map_while(std::result::Result::ok)
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<EvolutionEvent>(&line).ok())
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

#[test]
fn truncated_last_line_is_skipped_without_losing_earlier_events() {
    let root = tempfile::tempdir().unwrap();
    let log_path = root.path().join("rsi/evolution.jsonl");

    // Write 3 valid events.
    write_event(
        &log_path,
        &make_event(
            1,
            EventKind::Proposed {
                targets: vec!["a".into()],
            },
        ),
    );
    write_event(&log_path, &make_event(1, EventKind::Validated));
    write_event(&log_path, &make_event(1, EventKind::Solidified));

    // Append a truncated (incomplete) JSON line.
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&log_path)
        .unwrap();
    file.write_all(b"{\"cycle\": 2, \"incomplete\": tru")
        .unwrap();

    let events = read_events(&log_path);
    assert_eq!(events.len(), 3, "the truncated line must be skipped");
    assert!(events.iter().all(|e| e.cycle == 1));
}

#[test]
fn fully_corrupt_log_degrades_to_empty() {
    let root = tempfile::tempdir().unwrap();
    let log_path = root.path().join("rsi/evolution.jsonl");
    std::fs::create_dir_all(log_path.parent().unwrap()).unwrap();
    std::fs::write(&log_path, "not json at all\n{also broken\n").unwrap();

    let events = read_events(&log_path);
    assert!(
        events.is_empty(),
        "a fully corrupt log must degrade to empty"
    );
}

#[test]
fn mixed_valid_and_corrupt_lines_preserve_valid_ones() {
    let root = tempfile::tempdir().unwrap();
    let log_path = root.path().join("rsi/evolution.jsonl");

    write_event(
        &log_path,
        &make_event(1, EventKind::Proposed { targets: vec![] }),
    );
    std::fs::OpenOptions::new()
        .append(true)
        .open(&log_path)
        .unwrap()
        .write_all(b"corrupt line\n")
        .unwrap();
    write_event(&log_path, &make_event(2, EventKind::Solidified));

    let events = read_events(&log_path);
    assert_eq!(
        events.len(),
        2,
        "valid lines before and after a corrupt line survive"
    );
    assert_eq!(events[0].cycle, 1);
    assert_eq!(events[1].cycle, 2);
}

#[test]
fn rollback_appends_rolled_back_event_to_log() {
    let root = tempfile::tempdir().unwrap();
    seed(root.path());
    let config = ApplyConfig::new(root.path());
    let log_path = root.path().join("rsi/evolution.jsonl");

    let proposal = proposal_with_score(0.0);
    let changes = vec![Change::FileChange {
        path: "recipes/target.yaml".to_string(),
        content: "version: 2\ntitle: mutated\n".to_string(),
    }];
    apply(
        &config,
        &proposal,
        &passing_report(&proposal, 1.0),
        &changes,
        None,
    )
    .unwrap();

    // Write a Solidified event before rollback.
    write_event(
        &log_path,
        &make_event(1, EventKind::Applied { backup_version: 1 }),
    );

    // Rollback.
    let rolled = rollback_to_version(&config, 1).unwrap();
    assert_eq!(rolled.backup_version, 1);
    assert_eq!(rolled.restored_targets.len(), 1);

    // Simulate the CLI appending a RolledBack event.
    let rollback_event = make_event(
        0,
        EventKind::RolledBack {
            reason: "manual rollback to version 1".to_string(),
        },
    );
    write_event(&log_path, &rollback_event);

    let events = read_events(&log_path);
    assert_eq!(events.len(), 2);
    assert!(matches!(events[1].kind, EventKind::RolledBack { .. }));
}

#[test]
fn empty_log_directory_does_not_panic() {
    let root = tempfile::tempdir().unwrap();
    let log_path = root.path().join("rsi/evolution.jsonl");
    // No file exists.
    let events = read_events(&log_path);
    assert!(events.is_empty());
}

#[test]
fn event_serde_round_trip_preserves_all_variants() {
    let variants = vec![
        make_event(
            1,
            EventKind::Proposed {
                targets: vec!["a.yaml".into(), "b.yaml".into()],
            },
        ),
        make_event(1, EventKind::Validated),
        make_event(
            1,
            EventKind::FitnessEvaluated {
                report: passing_report(&proposal_with_score(0.0), 0.9),
            },
        ),
        make_event(1, EventKind::Applied { backup_version: 3 }),
        make_event(1, EventKind::Solidified),
        make_event(
            1,
            EventKind::RolledBack {
                reason: "test".into(),
            },
        ),
        make_event(
            1,
            EventKind::BreakerTripped {
                consecutive_failures: 2,
            },
        ),
        make_event(1, EventKind::SkippedConcurrent),
        make_event(1, EventKind::SkippedPaused),
    ];

    for event in &variants {
        let json = serde_json::to_string(event).unwrap();
        let parsed: EvolutionEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.cycle, event.cycle);
        assert_eq!(parsed.proposal_id, event.proposal_id);
    }
}

#[test]
fn history_shows_fitness_score_for_solidified_chain() {
    let root = tempfile::tempdir().unwrap();
    let log_path = root.path().join("rsi/evolution.jsonl");

    let report = passing_report(&proposal_with_score(0.5), 0.85);
    write_event(
        &log_path,
        &make_event(
            1,
            EventKind::Proposed {
                targets: vec!["recipes/x.yaml".into()],
            },
        ),
    );
    write_event(&log_path, &make_event(1, EventKind::Validated));
    write_event(
        &log_path,
        &make_event(1, EventKind::FitnessEvaluated { report }),
    );
    write_event(
        &log_path,
        &make_event(1, EventKind::Applied { backup_version: 1 }),
    );
    write_event(&log_path, &make_event(1, EventKind::Solidified));

    let events = read_events(&log_path);
    assert_eq!(events.len(), 5);

    // The fitness event must carry the score.
    let fitness_event = events
        .iter()
        .find(|e| matches!(e.kind, EventKind::FitnessEvaluated { .. }))
        .unwrap();
    if let EventKind::FitnessEvaluated { report } = &fitness_event.kind {
        assert!((report.score - 0.85).abs() < 0.01);
        assert!(report.passed);
    } else {
        panic!("expected FitnessEvaluated");
    }
}
