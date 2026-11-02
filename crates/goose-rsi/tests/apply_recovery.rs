//! Apply layer acceptance tests (issue #3).
//!
//! Data invariants:
//!
//! 1. after an apply the managed root is either fully new or fully old
//!    (hash level) — no mixed intermediate state;
//! 2. `rollback(v)` restores every managed file to the hashes recorded in
//!    backup v;
//! 3. an apply killed mid-flight (real child-process abort) is detected by
//!    the startup recovery check and restored to the old version;
//! 4. a proposal without a passing fitness report is refused with zero
//!    writes.

use goose_rsi::apply::{
    apply, list_backup_versions, recover, rollback_to_version, ApplyConfig, ApplyError, Change,
    Failpoint, RecoveryReport,
};
use goose_rsi::fitness::FitnessReport;
use goose_rsi::proposal::{FitnessExpectation, MutationProposal, MutationTarget, ProposalSource};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

fn proposal_with(targets: Vec<MutationTarget>) -> MutationProposal {
    MutationProposal::new(
        "apply test",
        ProposalSource::Scheduled,
        targets,
        FitnessExpectation {
            baseline_score: 0.5,
            minimum_score: 0.5,
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
        raw_log_ref: PathBuf::from("/dev/null"),
        timestamp: SystemTime::UNIX_EPOCH,
    }
}

fn three_file_proposal() -> (MutationProposal, Vec<Change>) {
    let proposal = proposal_with(vec![
        MutationTarget::FilePath {
            path: "recipes/a.yaml".to_string(),
        },
        MutationTarget::FilePath {
            path: "recipes/b.yaml".to_string(),
        },
        MutationTarget::FilePath {
            path: "recipes/c.yaml".to_string(),
        },
    ]);
    let changes = vec![
        Change::FileChange {
            path: "recipes/a.yaml".to_string(),
            content: "a.yaml: new".to_string(),
        },
        Change::FileChange {
            path: "recipes/b.yaml".to_string(),
            content: "b.yaml: new".to_string(),
        },
        Change::FileChange {
            path: "recipes/c.yaml".to_string(),
            content: "c.yaml: new".to_string(),
        },
    ];
    (proposal, changes)
}

fn seed_managed_root(root: &Path) {
    std::fs::create_dir_all(root.join("recipes")).unwrap();
    for name in ["a.yaml", "b.yaml", "c.yaml"] {
        std::fs::write(root.join("recipes").join(name), format!("{name}: old")).unwrap();
    }
}

/// Hash tree of every managed file (excluding the `rsi/` bookkeeping dir).
fn managed_hashes(root: &Path) -> BTreeMap<String, String> {
    fn visit(path: &Path, prefix: String, out: &mut BTreeMap<String, String>) {
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
        let content = std::fs::read(path).unwrap();
        let digest = sha256_hex_for_test(&content);
        out.insert(prefix, digest);
    }
    let mut out = BTreeMap::new();
    if root.join("recipes").exists() {
        visit(&root.join("recipes"), "/recipes".to_string(), &mut out);
    }
    if root.join("config.yaml").exists() {
        out.insert(
            "/config.yaml".to_string(),
            sha256_hex_for_test(&std::fs::read(root.join("config.yaml")).unwrap()),
        );
    }
    out
}

fn sha256_hex_for_test(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn apply_is_all_or_nothing_at_hash_level() {
    let root = tempfile::tempdir().unwrap();
    seed_managed_root(root.path());
    let old = managed_hashes(root.path());
    let config = ApplyConfig::new(root.path());
    let (proposal, changes) = three_file_proposal();

    let applied = apply(
        &config,
        &proposal,
        &passing_report(&proposal, 0.9),
        &changes,
        None,
    )
    .expect("apply must succeed");

    assert_eq!(applied.backup_version, 1);
    assert_eq!(applied.targets.len(), 3);
    // Fully new: every managed file carries its new content hash.
    for name in ["a.yaml", "b.yaml", "c.yaml"] {
        let content = std::fs::read_to_string(root.path().join("recipes").join(name)).unwrap();
        assert_eq!(content, format!("{name}: new"));
    }
    let new = managed_hashes(root.path());
    assert_ne!(
        old, new,
        "the managed root must have moved to the new version"
    );
    assert_eq!(list_backup_versions(&config), vec![1]);
}

#[test]
fn rollback_restores_exact_backup_hashes() {
    let root = tempfile::tempdir().unwrap();
    seed_managed_root(root.path());
    let old = managed_hashes(root.path());
    let config = ApplyConfig::new(root.path());
    let (proposal, changes) = three_file_proposal();

    apply(
        &config,
        &proposal,
        &passing_report(&proposal, 0.9),
        &changes,
        None,
    )
    .unwrap();
    assert_ne!(managed_hashes(root.path()), old);

    let rolled = rollback_to_version(&config, 1).expect("rollback must succeed");
    assert_eq!(rolled.backup_version, 1);
    assert_eq!(managed_hashes(root.path()), old, "hash-level restore");
}

#[test]
fn failed_fitness_gate_means_zero_writes() {
    let root = tempfile::tempdir().unwrap();
    seed_managed_root(root.path());
    let old = managed_hashes(root.path());
    let config = ApplyConfig::new(root.path());
    let (proposal, changes) = three_file_proposal();

    // Below the minimum score.
    let failing = passing_report(&proposal, 0.2);
    assert!(!failing.passed);
    let error =
        apply(&config, &proposal, &failing, &changes, None).expect_err("fitness gate must refuse");
    assert!(matches!(error, ApplyError::FitnessGate { .. }));
    assert_eq!(
        managed_hashes(root.path()),
        old,
        "zero writes on gate refusal"
    );
    assert!(list_backup_versions(&config).is_empty());
    // No journal left behind either.
    assert!(matches!(recover(&config), Ok(RecoveryReport::Clean)));
}

#[test]
fn whitelist_is_reenforced_at_apply_time() {
    let root = tempfile::tempdir().unwrap();
    let config = ApplyConfig::new(root.path());
    let proposal = proposal_with(vec![MutationTarget::FilePath {
        path: "secrets.yaml".to_string(),
    }]);
    let changes = vec![Change::FileChange {
        path: "secrets.yaml".to_string(),
        content: "stolen".to_string(),
    }];
    let error = apply(
        &config,
        &proposal,
        &passing_report(&proposal, 1.0),
        &changes,
        None,
    )
    .expect_err("deny-listed target must be refused at apply time");
    assert!(matches!(error, ApplyError::Whitelist(_)));
    assert!(!root.path().join("secrets.yaml").exists());
}

#[test]
fn change_set_smuggling_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    seed_managed_root(root.path());
    let old = managed_hashes(root.path());
    let config = ApplyConfig::new(root.path());
    let (proposal, _) = three_file_proposal();
    // The change set targets a file the proposal never declared.
    let smuggled = vec![
        Change::FileChange {
            path: "recipes/a.yaml".to_string(),
            content: "a.yaml: new".to_string(),
        },
        Change::FileChange {
            path: "recipes/b.yaml".to_string(),
            content: "b.yaml: new".to_string(),
        },
        Change::FileChange {
            path: "prompts/smuggled.md".to_string(),
            content: "smuggled".to_string(),
        },
    ];
    let error = apply(
        &config,
        &proposal,
        &passing_report(&proposal, 0.9),
        &smuggled,
        None,
    )
    .expect_err("smuggling must be rejected");
    assert!(matches!(error, ApplyError::TargetMismatch { .. }));
    assert_eq!(managed_hashes(root.path()), old);
}

#[test]
fn config_key_changes_round_trip_through_rollback() {
    let root = tempfile::tempdir().unwrap();
    seed_managed_root(root.path());
    std::fs::write(root.path().join("config.yaml"), "GOOSE_MODEL: old-model\n").unwrap();
    let old = managed_hashes(root.path());
    let config = ApplyConfig::new(root.path());

    let proposal = proposal_with(vec![MutationTarget::ConfigKey {
        key: "GOOSE_TEMPERATURE".to_string(),
    }]);
    let changes = vec![Change::ConfigKeyChange {
        key: "GOOSE_TEMPERATURE".to_string(),
        value: serde_json::json!(0.3),
    }];
    apply(
        &config,
        &proposal,
        &passing_report(&proposal, 0.9),
        &changes,
        None,
    )
    .unwrap();

    let updated = std::fs::read_to_string(root.path().join("config.yaml")).unwrap();
    assert!(updated.contains("GOOSE_TEMPERATURE"));
    assert!(updated.contains("GOOSE_MODEL"), "existing keys survive");

    rollback_to_version(&config, 1).unwrap();
    assert_eq!(managed_hashes(root.path()), old);
}

#[test]
fn two_config_keys_in_one_proposal_do_not_clobber_each_other() {
    let root = tempfile::tempdir().unwrap();
    let config = ApplyConfig::new(root.path());
    let proposal = proposal_with(vec![
        MutationTarget::ConfigKey {
            key: "GOOSE_MODEL".to_string(),
        },
        MutationTarget::ConfigKey {
            key: "GOOSE_TEMPERATURE".to_string(),
        },
    ]);
    let changes = vec![
        Change::ConfigKeyChange {
            key: "GOOSE_MODEL".to_string(),
            value: serde_json::json!("test-model"),
        },
        Change::ConfigKeyChange {
            key: "GOOSE_TEMPERATURE".to_string(),
            value: serde_json::json!(0.7),
        },
    ];
    apply(
        &config,
        &proposal,
        &passing_report(&proposal, 0.9),
        &changes,
        None,
    )
    .unwrap();
    let updated = std::fs::read_to_string(root.path().join("config.yaml")).unwrap();
    assert!(updated.contains("GOOSE_MODEL"));
    assert!(updated.contains("GOOSE_TEMPERATURE"));
}

#[test]
fn new_files_are_removed_again_by_rollback() {
    let root = tempfile::tempdir().unwrap();
    seed_managed_root(root.path());
    let config = ApplyConfig::new(root.path());
    let proposal = proposal_with(vec![MutationTarget::FilePath {
        path: "recipes/fresh.yaml".to_string(),
    }]);
    let changes = vec![Change::FileChange {
        path: "recipes/fresh.yaml".to_string(),
        content: "fresh: true".to_string(),
    }];
    apply(
        &config,
        &proposal,
        &passing_report(&proposal, 0.9),
        &changes,
        None,
    )
    .unwrap();
    assert!(root.path().join("recipes/fresh.yaml").exists());
    rollback_to_version(&config, 1).unwrap();
    assert!(
        !root.path().join("recipes/fresh.yaml").exists(),
        "previously absent file must be removed"
    );
}

/// Child mode: run inside a spawned test binary process so the abort below
/// is a real process death (SIGABRT, no destructors — same on-disk state as
/// SIGKILL at that step).
#[test]
#[ignore = "spawned by crash tests via env switches"]
fn crash_child_apply() {
    let root = std::env::var("GOOSE_RSI_TEST_MANAGED_ROOT").expect("managed root env");
    let failpoint = match std::env::var("GOOSE_RSI_TEST_FAILPOINT")
        .expect("failpoint env")
        .as_str()
    {
        "after_journal" => Failpoint::AfterJournalWrite,
        "after_first_rename" => Failpoint::AfterFirstRename,
        "after_commit" => Failpoint::AfterCommitBeforeJournalRemoval,
        other => panic!("unknown failpoint {other}"),
    };
    let config = ApplyConfig::new(&root);
    let (proposal, changes) = three_file_proposal();
    let _ = apply(
        &config,
        &proposal,
        &passing_report(&proposal, 0.9),
        &changes,
        Some(failpoint),
    );
    panic!("apply should have aborted the process at the failpoint");
}

fn run_crash_child(root: &Path, failpoint: &str) {
    let exe = std::env::current_exe().expect("current test binary");
    let output = std::process::Command::new(exe)
        .args(["--exact", "crash_child_apply", "--ignored", "--nocapture"])
        .env("GOOSE_RSI_TEST_MANAGED_ROOT", root)
        .env("GOOSE_RSI_TEST_FAILPOINT", failpoint)
        .output()
        .expect("spawn crash child");
    assert!(
        !output.status.success(),
        "crash child must die at the failpoint, got {:?}",
        output.status
    );
}

#[test]
fn recovery_after_kill_mid_apply_restores_old_version() {
    let root = tempfile::tempdir().unwrap();
    seed_managed_root(root.path());
    let old = managed_hashes(root.path());
    let config = ApplyConfig::new(root.path());

    run_crash_child(root.path(), "after_first_rename");

    // Mixed on-disk state right after the kill: first file new, rest old.
    let a = std::fs::read_to_string(root.path().join("recipes/a.yaml")).unwrap();
    assert_eq!(a, "a.yaml: new");
    let b = std::fs::read_to_string(root.path().join("recipes/b.yaml")).unwrap();
    assert_eq!(b, "b.yaml: old");

    let report = recover(&config).expect("recovery must succeed");
    assert_eq!(report, RecoveryReport::RestoredToOld { backup_version: 1 });
    assert_eq!(
        managed_hashes(root.path()),
        old,
        "fully restored to the old version"
    );
    // Recovery is idempotent: a second scan finds nothing to do.
    assert!(matches!(recover(&config), Ok(RecoveryReport::Clean)));
}

#[test]
fn recovery_after_kill_before_any_rename_leaves_old_intact() {
    let root = tempfile::tempdir().unwrap();
    seed_managed_root(root.path());
    let old = managed_hashes(root.path());
    let config = ApplyConfig::new(root.path());

    run_crash_child(root.path(), "after_journal");

    assert_eq!(managed_hashes(root.path()), old);
    let report = recover(&config).unwrap();
    assert_eq!(report, RecoveryReport::RestoredToOld { backup_version: 1 });
    assert_eq!(managed_hashes(root.path()), old);
}

#[test]
fn recovery_after_kill_after_commit_keeps_new_version() {
    let root = tempfile::tempdir().unwrap();
    seed_managed_root(root.path());
    let config = ApplyConfig::new(root.path());
    let (_, changes) = three_file_proposal();
    let expected: BTreeMap<String, String> = changes
        .iter()
        .map(|change| {
            let Change::FileChange { path, content } = change else {
                panic!("file change expected");
            };
            (
                format!("/recipes/{}", path.strip_prefix("recipes/").unwrap()),
                sha256_hex_for_test(content.as_bytes()),
            )
        })
        .collect();

    run_crash_child(root.path(), "after_commit");

    // The commit fully landed; recovery only cleans the journal.
    let report = recover(&config).unwrap();
    assert_eq!(report, RecoveryReport::CommittedKept { backup_version: 1 });
    assert_eq!(managed_hashes(root.path()), expected);
}

#[test]
fn lock_does_not_outlive_a_killed_process() {
    let root = tempfile::tempdir().unwrap();
    seed_managed_root(root.path());
    let config = ApplyConfig::new(root.path());
    run_crash_child(root.path(), "after_journal");
    // The flock died with the child process: a fresh apply succeeds.
    let (proposal, changes) = three_file_proposal();
    let applied = apply(
        &config,
        &proposal,
        &passing_report(&proposal, 0.9),
        &changes,
        None,
    );
    assert!(
        applied.is_ok(),
        "no stale lock may block the next apply: {applied:?}"
    );
}

#[test]
fn tampered_backup_content_is_detected_on_rollback() {
    let root = tempfile::tempdir().unwrap();
    seed_managed_root(root.path());
    let config = ApplyConfig::new(root.path());
    let (proposal, changes) = three_file_proposal();
    apply(
        &config,
        &proposal,
        &passing_report(&proposal, 0.9),
        &changes,
        None,
    )
    .unwrap();

    // Corrupt the backup content behind the manifest's back.
    std::fs::write(
        root.path().join("rsi/backups/v1/recipes/a.yaml"),
        "a.yaml: tampered",
    )
    .unwrap();

    let error =
        rollback_to_version(&config, 1).expect_err("hash drift in the backup must be detected");
    assert!(matches!(error, ApplyError::MissingBackup { .. }));
}

#[test]
fn rollback_to_missing_version_fails_cleanly() {
    let root = tempfile::tempdir().unwrap();
    let config = ApplyConfig::new(root.path());
    let error = rollback_to_version(&config, 99).expect_err("missing backup must be an error");
    assert!(matches!(
        error,
        ApplyError::MissingBackup { version: 99, .. }
    ));
}
