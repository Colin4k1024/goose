//! Atomic apply layer (issue #3, RSI M1 L3).
//!
//! Writes a fitness-approved proposal's changes into the managed root (the
//! goose user config directory in production) as one all-or-nothing group:
//!
//! 1. **gates** — the proposal must pass the frozen whitelist again and the
//!    fitness report must clear the proposal's minimum score;
//! 2. **backup** — every touched file is copied into a versioned backup
//!    directory with a SHA-256 manifest;
//! 3. **journal** — `pending-apply.json` records the in-flight apply;
//! 4. **commit** — all new contents are staged first, then renamed into
//!    place one by one;
//! 5. **recover** — [`recover`] (run at startup) detects a journal left
//!    behind by a crashed apply and restores the old version, so the managed
//!    root is always either fully old or fully new (hash-level);
//! 6. **rollback** — [`rollback_to_version`] restores any backup version and
//!    verifies every restored file's hash.
//!
//! Crash injection: [`Failpoint`] aborts the process at a chosen step with
//! no cleanup — exactly the state a SIGKILL would leave — so recovery is
//! tested against real process death.
//!
//! Path handling contract (inherited from the whitelist review of #1): file
//! targets validated by [`crate::whitelist`] are used **verbatim** as
//! relative paths under the managed root. This module never decodes,
//! canonicalizes or re-normalizes them, keeping the validated path and the
//! written path identical.

use crate::fitness::FitnessReport;
use crate::proposal::{MutationProposal, MutationTarget};
use crate::whitelist::{validate_target, ValidationError};
use serde::{Deserialize, Serialize};
use serde_yaml::Mapping as YamlMapping;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

/// One mutation in a change set. Targets must match the proposal's declared
/// targets exactly — smuggling extra targets past the proposal is rejected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Change {
    /// Create or replace a managed file (config-dir relative, verbatim).
    FileChange { path: String, content: String },
    /// Set a flat config key in `config.yaml`.
    ConfigKeyChange {
        key: String,
        value: serde_json::Value,
    },
}

impl Change {
    pub fn target(&self) -> MutationTarget {
        match self {
            Change::FileChange { path, .. } => MutationTarget::FilePath { path: path.clone() },
            Change::ConfigKeyChange { key, .. } => MutationTarget::ConfigKey { key: key.clone() },
        }
    }
}

/// Where the apply layer operates. In production the goose user config dir;
/// tests point this at a tempdir.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplyConfig {
    pub managed_root: PathBuf,
}

impl ApplyConfig {
    pub fn new(managed_root: impl Into<PathBuf>) -> Self {
        ApplyConfig {
            managed_root: managed_root.into(),
        }
    }

    pub(crate) fn rsi_dir(&self) -> PathBuf {
        self.managed_root.join("rsi")
    }
    fn backups_dir(&self) -> PathBuf {
        self.rsi_dir().join("backups")
    }
    fn backup_dir(&self, version: u32) -> PathBuf {
        self.backups_dir().join(format!("v{version}"))
    }
    fn journal_path(&self) -> PathBuf {
        self.rsi_dir().join("pending-apply.json")
    }
    fn staging_dir(&self) -> PathBuf {
        self.rsi_dir().join("staging")
    }
    fn lock_path(&self) -> PathBuf {
        self.rsi_dir().join("apply.lock")
    }
}

/// Fault-injection points for crash-recovery tests. Each one aborts the
/// process (`SIGABRT`, no cleanup, no lock release code) at a precise step
/// of the apply pipeline, leaving exactly the on-disk state a SIGKILL would.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Failpoint {
    /// After the journal exists, before any target is touched.
    AfterJournalWrite,
    /// After the first target rename, mid-group.
    AfterFirstRename,
    /// After all targets committed, before the journal is removed.
    AfterCommitBeforeJournalRemoval,
}

#[derive(Debug, thiserror::Error)]
pub enum ApplyError {
    #[error("fitness gate rejected the proposal: {reason}")]
    FitnessGate { reason: String },
    #[error("whitelist rejected the proposal")]
    Whitelist(#[from] ValidationError),
    #[error("change set does not match proposal targets: {mismatch}")]
    TargetMismatch { mismatch: String },
    #[error("post-apply hash verification failed for {target}")]
    HashMismatch { target: String },
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("journal or manifest is corrupt: {reason}")]
    Corrupt { reason: String },
    #[error("backup v{version} is missing or incomplete: {reason}")]
    MissingBackup { version: u32, reason: String },
    #[error("another apply is already in progress")]
    AlreadyLocked,
}

/// One backed-up target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackupEntry {
    /// Managed-root-relative path (verbatim), or the config sentinel.
    pub target: String,
    /// Whether the target existed before the apply.
    pub existed: bool,
    /// SHA-256 of the pre-apply content (hex), when it existed.
    pub old_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackupManifest {
    pub version: u32,
    pub proposal_id: String,
    pub created_at: std::time::SystemTime,
    pub entries: Vec<BackupEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PendingJournal {
    backup_version: u32,
    /// `in_progress` (crashed mid-apply → restore old) or `committed`
    /// (crash after commit → keep new, just clean the journal).
    state: PendingState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PendingState {
    InProgress,
    Committed,
}

/// Result of a successful apply.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Applied {
    pub proposal_id: String,
    pub backup_version: u32,
    /// Managed-root-relative targets written (deduplicated, ordered).
    pub targets: Vec<String>,
    /// SHA-256 (hex) of the new contents, in target order.
    pub new_sha256: Vec<String>,
}

/// Result of a startup recovery scan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum RecoveryReport {
    /// No journal found: nothing to do.
    Clean,
    /// Crashed mid-apply: everything restored to the old version.
    RestoredToOld { backup_version: u32 },
    /// Crashed after commit: the new version stands, journal cleaned.
    CommittedKept { backup_version: u32 },
}

/// Result of a rollback.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RolledBack {
    pub backup_version: u32,
    pub restored_targets: Vec<String>,
}

/// Sentinel target label covering the shared `config.yaml`.
const CONFIG_TARGET_SENTINEL: &str = "<config.yaml>";
const CONFIG_FILE: &str = "config.yaml";

/// Apply a fitness-approved proposal atomically.
///
/// Zero writes happen before both gates pass: whitelist re-validation and
/// the fitness report (`report.passed && report.score >= minimum_score`).
pub fn apply(
    config: &ApplyConfig,
    proposal: &MutationProposal,
    report: &FitnessReport,
    changes: &[Change],
    failpoint: Option<Failpoint>,
) -> Result<Applied, ApplyError> {
    if !report.passed || report.score < proposal.fitness.minimum_score {
        return Err(ApplyError::FitnessGate {
            reason: format!(
                "report passed={} score={} minimum={}",
                report.passed, report.score, proposal.fitness.minimum_score
            ),
        });
    }
    // Defense in depth: re-run the frozen whitelist over every target.
    for target in &proposal.targets {
        validate_target(target)?;
    }
    verify_changeset_matches_proposal(proposal, changes)?;

    std::fs::create_dir_all(config.rsi_dir())?;
    let _lock = acquire_lock(config)?;

    let version = next_backup_version(config)?;
    let planned = planned_writes(config, changes)?;
    let manifest = create_backup(config, version, proposal, &planned)?;

    write_json(
        &config.journal_path(),
        &PendingJournal {
            backup_version: version,
            state: PendingState::InProgress,
        },
    )?;
    if failpoint == Some(Failpoint::AfterJournalWrite) {
        abort_process();
    }

    commit_planned(config, &planned, failpoint)?;

    // Post-commit verification: the group is either fully new or the apply
    // is rolled back to the old version.
    if let Err(error) = verify_planned_hashes(config, &planned) {
        restore_backup(config, &manifest)?;
        let _ = std::fs::remove_file(config.journal_path());
        return Err(error);
    }

    write_json(
        &config.journal_path(),
        &PendingJournal {
            backup_version: version,
            state: PendingState::Committed,
        },
    )?;
    if failpoint == Some(Failpoint::AfterCommitBeforeJournalRemoval) {
        abort_process();
    }
    let _ = std::fs::remove_file(config.journal_path());

    Ok(Applied {
        proposal_id: proposal.id.to_string(),
        backup_version: version,
        targets: planned.iter().map(|write| write.label.clone()).collect(),
        new_sha256: planned
            .iter()
            .map(|write| sha256_hex(write.content.as_bytes()))
            .collect(),
    })
}

/// Startup recovery: detect a crashed apply and restore consistency.
pub fn recover(config: &ApplyConfig) -> Result<RecoveryReport, ApplyError> {
    let journal_path = config.journal_path();
    let Ok(raw) = std::fs::read_to_string(&journal_path) else {
        return Ok(RecoveryReport::Clean);
    };
    let journal: PendingJournal =
        serde_json::from_str(&raw).map_err(|error| ApplyError::Corrupt {
            reason: format!("journal unreadable: {error}"),
        })?;
    match journal.state {
        PendingState::InProgress => {
            let manifest = load_manifest(config, journal.backup_version)?;
            restore_backup(config, &manifest)?;
            let _ = std::fs::remove_file(&journal_path);
            Ok(RecoveryReport::RestoredToOld {
                backup_version: journal.backup_version,
            })
        }
        PendingState::Committed => {
            let _ = std::fs::remove_file(&journal_path);
            Ok(RecoveryReport::CommittedKept {
                backup_version: journal.backup_version,
            })
        }
    }
}

/// Restore the managed root to a recorded backup version. Every restored
/// target is hash-verified against the manifest.
pub fn rollback_to_version(config: &ApplyConfig, version: u32) -> Result<RolledBack, ApplyError> {
    std::fs::create_dir_all(config.rsi_dir())?;
    let _lock = acquire_lock(config)?;
    let manifest = load_manifest(config, version)?;
    restore_backup(config, &manifest)?;
    Ok(RolledBack {
        backup_version: version,
        restored_targets: manifest
            .entries
            .iter()
            .map(|entry| entry.target.clone())
            .collect(),
    })
}

/// Versions available for rollback, ascending.
pub fn list_backup_versions(config: &ApplyConfig) -> Vec<u32> {
    let mut versions: Vec<u32> = std::fs::read_dir(config.backups_dir())
        .map(|entries| {
            entries
                .filter_map(|entry| {
                    let name = entry.ok()?.file_name().into_string().ok()?;
                    name.strip_prefix('v')?.parse().ok()
                })
                .collect()
        })
        .unwrap_or_default();
    versions.sort_unstable();
    versions
}

/// The final content each touched file must have after the apply.
///
/// File changes map directly; all config-key changes fold into one
/// `config.yaml` render (later keys win), so two keys in one proposal can
/// never overwrite each other.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedWrite {
    /// Human-facing label (verbatim path or the config sentinel).
    label: String,
    /// Managed-root-relative destination, verbatim.
    relative: PathBuf,
    content: String,
}

fn planned_writes(
    config: &ApplyConfig,
    changes: &[Change],
) -> Result<Vec<PlannedWrite>, ApplyError> {
    // BTreeMap keyed by relative path: deterministic order, last-write-wins
    // per file, config keys folded into one entry.
    let mut files: BTreeMap<PathBuf, PlannedWrite> = BTreeMap::new();
    let mut config_mapping: Option<(YamlMapping, String)> = None;

    for change in changes {
        match change {
            Change::FileChange { path, content } => {
                files.insert(
                    PathBuf::from(path),
                    PlannedWrite {
                        label: path.clone(),
                        relative: PathBuf::from(path),
                        content: content.clone(),
                    },
                );
            }
            Change::ConfigKeyChange { key, value } => {
                let entry = config_mapping.get_or_insert_with(|| {
                    (
                        load_config_mapping(config),
                        CONFIG_TARGET_SENTINEL.to_string(),
                    )
                });
                let yaml_value =
                    serde_yaml::to_value(value).map_err(|error| ApplyError::Corrupt {
                        reason: format!("config value for {key} not serializable: {error}"),
                    })?;
                entry.0.insert(key.as_str().into(), yaml_value);
            }
        }
    }

    if let Some((mapping, label)) = config_mapping {
        let content = serde_yaml::to_string(&mapping).map_err(|error| ApplyError::Corrupt {
            reason: format!("config.yaml not serializable: {error}"),
        })?;
        files.insert(
            PathBuf::from(CONFIG_FILE),
            PlannedWrite {
                label,
                relative: PathBuf::from(CONFIG_FILE),
                content,
            },
        );
    }

    Ok(files.into_values().collect())
}

fn load_config_mapping(config: &ApplyConfig) -> YamlMapping {
    let config_path = config.managed_root.join(CONFIG_FILE);
    if config_path.exists() {
        std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|raw| serde_yaml::from_str(&raw).ok())
            .unwrap_or_default()
    } else {
        YamlMapping::default()
    }
}

/// Holds the advisory exclusive apply lock. The OS releases the flock when
/// the owning process dies, so a SIGKILLed apply never leaves a stale lock.
struct ApplyLock {
    _file: File,
}

fn acquire_lock(config: &ApplyConfig) -> Result<ApplyLock, ApplyError> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(config.lock_path())?;
    fs2::FileExt::try_lock_exclusive(&file).map_err(|_| ApplyError::AlreadyLocked)?;
    Ok(ApplyLock { _file: file })
}

fn next_backup_version(config: &ApplyConfig) -> Result<u32, ApplyError> {
    Ok(list_backup_versions(config).last().copied().unwrap_or(0) + 1)
}

fn create_backup(
    config: &ApplyConfig,
    version: u32,
    proposal: &MutationProposal,
    planned: &[PlannedWrite],
) -> Result<BackupManifest, ApplyError> {
    let backup_dir = config.backup_dir(version);
    std::fs::create_dir_all(&backup_dir)?;
    let mut entries = Vec::new();
    for write in planned {
        let source = config.managed_root.join(&write.relative);
        let entry = if source.exists() {
            let content = std::fs::read(&source)?;
            let backup_path = backup_dir.join(&write.relative);
            if let Some(parent) = backup_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&backup_path, &content)?;
            BackupEntry {
                target: write.label.clone(),
                existed: true,
                old_sha256: Some(sha256_hex(&content)),
            }
        } else {
            BackupEntry {
                target: write.label.clone(),
                existed: false,
                old_sha256: None,
            }
        };
        entries.push(entry);
    }
    let manifest = BackupManifest {
        version,
        proposal_id: proposal.id.to_string(),
        created_at: std::time::SystemTime::now(),
        entries,
    };
    write_json(&backup_dir.join("manifest.json"), &manifest)?;
    Ok(manifest)
}

/// Stage every new content first, then rename each into place. A crash
/// before the first rename leaves the old version fully intact; a crash
/// between renames is repaired by [`recover`] from the backup.
fn commit_planned(
    config: &ApplyConfig,
    planned: &[PlannedWrite],
    failpoint: Option<Failpoint>,
) -> Result<(), ApplyError> {
    let staging = config.staging_dir();
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging)?;

    let mut staged: Vec<(PathBuf, &PlannedWrite)> = Vec::new();
    for (index, write) in planned.iter().enumerate() {
        let staged_path = staging.join(format!("change-{index}"));
        std::fs::write(&staged_path, write.content.as_bytes())?;
        staged.push((staged_path, write));
    }

    for (index, (staged_path, write)) in staged.iter().enumerate() {
        let target = config.managed_root.join(&write.relative);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::rename(staged_path, &target)?;
        if index == 0 && failpoint == Some(Failpoint::AfterFirstRename) {
            abort_process();
        }
    }
    let _ = std::fs::remove_dir_all(&staging);
    Ok(())
}

fn verify_planned_hashes(config: &ApplyConfig, planned: &[PlannedWrite]) -> Result<(), ApplyError> {
    for write in planned {
        let target = config.managed_root.join(&write.relative);
        let actual = sha256_hex(&std::fs::read(&target)?);
        let expected = sha256_hex(write.content.as_bytes());
        if actual != expected {
            return Err(ApplyError::HashMismatch {
                target: write.label.clone(),
            });
        }
    }
    Ok(())
}

fn restore_backup(config: &ApplyConfig, manifest: &BackupManifest) -> Result<(), ApplyError> {
    let backup_dir = config.backup_dir(manifest.version);
    for entry in &manifest.entries {
        let relative = if entry.target == CONFIG_TARGET_SENTINEL {
            PathBuf::from(CONFIG_FILE)
        } else {
            PathBuf::from(&entry.target)
        };
        let live = config.managed_root.join(&relative);
        match entry.existed {
            true => {
                let backup_path = backup_dir.join(&relative);
                let content =
                    std::fs::read(&backup_path).map_err(|error| ApplyError::MissingBackup {
                        version: manifest.version,
                        reason: format!("{}: {error}", entry.target),
                    })?;
                let restored_hash = sha256_hex(&content);
                if Some(&restored_hash) != entry.old_sha256.as_ref() {
                    return Err(ApplyError::MissingBackup {
                        version: manifest.version,
                        reason: format!("backup content hash drift for {}", entry.target),
                    });
                }
                if let Some(parent) = live.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&live, &content)?;
                let after = sha256_hex(&std::fs::read(&live)?);
                if after != restored_hash {
                    return Err(ApplyError::HashMismatch {
                        target: entry.target.clone(),
                    });
                }
            }
            false => {
                // The target did not exist before the apply: remove it again.
                if live.exists() {
                    std::fs::remove_file(&live)?;
                }
            }
        }
    }
    Ok(())
}

fn load_manifest(config: &ApplyConfig, version: u32) -> Result<BackupManifest, ApplyError> {
    let manifest_path = config.backup_dir(version).join("manifest.json");
    let raw =
        std::fs::read_to_string(&manifest_path).map_err(|error| ApplyError::MissingBackup {
            version,
            reason: format!("manifest unreadable: {error}"),
        })?;
    serde_json::from_str(&raw).map_err(|error| ApplyError::Corrupt {
        reason: format!("manifest v{version} unreadable: {error}"),
    })
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), ApplyError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let serialized = serde_json::to_string_pretty(value).map_err(|error| ApplyError::Corrupt {
        reason: format!("journal not serializable: {error}"),
    })?;
    // write-then-rename so the journal itself is never half-written
    let temp = path.with_extension("tmp");
    std::fs::write(&temp, serialized.as_bytes())?;
    std::fs::rename(&temp, path)?;
    Ok(())
}

fn verify_changeset_matches_proposal(
    proposal: &MutationProposal,
    changes: &[Change],
) -> Result<(), ApplyError> {
    let mut declared = proposal.targets.clone();
    let mut actual: Vec<MutationTarget> = changes.iter().map(|change| change.target()).collect();
    if declared.len() != actual.len() {
        return Err(ApplyError::TargetMismatch {
            mismatch: format!(
                "proposal declares {} targets, change set has {}",
                declared.len(),
                actual.len()
            ),
        });
    }
    declared.sort();
    actual.sort();
    if declared != actual {
        return Err(ApplyError::TargetMismatch {
            mismatch: "change set targets differ from proposal targets".to_string(),
        });
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

/// Real process abort: no destructors, no lock release code — the on-disk
/// state is indistinguishable from a SIGKILL at the same step.
fn abort_process() -> ! {
    std::process::abort()
}
