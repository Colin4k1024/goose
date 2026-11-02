//! Whitelist validation acceptance tests (issue #1).
//!
//! Positive: legal recipe change passes. Negative: deny-listed targets are
//! rejected with zero filesystem side effects. Fail-closed: unknown surfaces
//! and malformed targets are rejected.

use goose_rsi::proposal::{FitnessExpectation, MutationProposal, MutationTarget, ProposalSource};
use goose_rsi::whitelist::{validate_target, WhitelistCheck, DENIED_CONFIG_KEY_PATTERNS};
use std::fs;

fn proposal_with(targets: Vec<MutationTarget>) -> MutationProposal {
    MutationProposal::new(
        "test proposal",
        ProposalSource::Session,
        targets,
        FitnessExpectation {
            baseline_score: 0.5,
            minimum_score: 0.5,
        },
    )
}

fn file(path: &str) -> MutationTarget {
    MutationTarget::FilePath {
        path: path.to_string(),
    }
}

fn config(key: &str) -> MutationTarget {
    MutationTarget::ConfigKey {
        key: key.to_string(),
    }
}

#[test]
fn legal_recipe_proposal_passes_validate() {
    let proposal = proposal_with(vec![
        file("recipe/search.yaml"),
        file("recipes/search.yaml"),
        file("recipes/nested/dag/entry.yaml"),
        file("prompts/system.md"),
        config("GOOSE_MODEL"),
    ]);
    assert_eq!(proposal.validate(), Ok(()));
}

#[test]
fn deny_listed_secrets_are_rejected_without_filesystem_writes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let sentinel = dir.path().join("sentinel.yaml");
    fs::write(&sentinel, "unchanged").expect("write sentinel");

    let snapshot = || {
        let metadata = fs::metadata(&sentinel).expect("sentinel metadata");
        (
            fs::read_dir(dir.path())
                .expect("read dir")
                .map(|entry| {
                    entry
                        .expect("entry")
                        .file_name()
                        .to_string_lossy()
                        .into_owned()
                })
                .collect::<Vec<_>>(),
            metadata.modified().expect("mtime"),
        )
    };
    let before = snapshot();

    for denied in [
        "secrets.yaml",
        "secrets.yaml.lock",
        "backup/secrets.yaml.lock",
        "extensions/goose-mcp-server",
    ] {
        let proposal = proposal_with(vec![file(denied)]);
        let error = proposal
            .validate()
            .expect_err("deny-listed target must be rejected");
        assert!(
            matches!(error.check, WhitelistCheck::Denied { .. }),
            "expected Denied for {denied}, got {error:?}"
        );
    }

    let after = snapshot();
    assert_eq!(before.0, after.0, "no files may appear or disappear");
    assert_eq!(
        before.1, after.1,
        "validate() must not touch the filesystem"
    );
}

#[test]
fn deny_listed_config_keys_are_rejected() {
    for denied in [
        "GOOSE_MODE",
        "permissions",
        "permissions.interactive",
        "tool.permissions.auto",
        "extensions",
        "extensions.my-mcp-server",
        "extensions.my-mcp-server.args",
        "GOOSE_PROVIDER",
        "providers.openai.GOOSE_PROVIDER",
        "providers.openai.api_key",
        "providers.gemini.api_key.value",
        "session.credentials.refresh_token",
        "auth.token.value",
    ] {
        let result = validate_target(&config(denied));
        let error = match result {
            Err(error) => error,
            Ok(()) => panic!("deny-listed config key must be rejected: {denied}"),
        };
        assert!(
            matches!(error.check, WhitelistCheck::Denied { .. }),
            "expected Denied for {denied}, got {error:?}"
        );
    }
    // The frozen deny list itself must contain the contract anchors.
    assert!(DENIED_CONFIG_KEY_PATTERNS.contains(&"GOOSE_MODE"));
}

#[test]
fn non_security_config_keys_are_allowed() {
    for allowed in ["GOOSE_MODEL", "GOOSE_TEMPERATURE", "hints.enabled"] {
        assert_eq!(validate_target(&config(allowed)), Ok(()));
    }
}

#[test]
fn unknown_file_surfaces_fail_closed() {
    for unknown in [
        "crates/goose/src/lib.rs",
        "config.yaml",
        "recipes/entry.yml",
        "Cargo.toml",
        "binary/goose",
        // non-yaml secret spellings are not on the whitelist either: the
        // fail-closed default rejects them even though the deny pattern
        // `secrets*.yaml*` does not literally match
        "secrets.yml.lock",
    ] {
        let error =
            validate_target(&file(unknown)).expect_err("non-whitelisted surface must be rejected");
        assert!(
            matches!(error.check, WhitelistCheck::NotWhitelisted { .. }),
            "expected NotWhitelisted for {unknown}, got {error:?}"
        );
    }
}

#[test]
fn malformed_and_traversal_targets_are_rejected() {
    for invalid in [
        "",
        "/etc/passwd",
        "\\windows\\style",
        "recipes/../secrets.yaml",
        "recipes/./a.yaml",
        "recipes//a.yaml",
        "recipes/",
    ] {
        let error = validate_target(&file(invalid))
            .expect_err("malformed target must be rejected, not normalized into the whitelist");
        assert!(
            matches!(error.check, WhitelistCheck::NotWhitelisted { .. }),
            "expected NotWhitelisted for {invalid:?}, got {error:?}"
        );
    }
    for invalid in ["", ".GOOSE_MODEL", "GOOSE_MODEL.", "a..b"] {
        assert!(validate_target(&config(invalid)).is_err());
    }
}

#[test]
fn empty_target_set_is_rejected() {
    let error = proposal_with(vec![])
        .validate()
        .expect_err("empty proposal must be rejected");
    assert!(matches!(error.check, WhitelistCheck::NotWhitelisted { .. }));
}

#[test]
fn one_bad_target_rejects_the_whole_proposal() {
    let proposal = proposal_with(vec![file("recipes/ok.yaml"), file("secrets.yaml")]);
    let error = proposal.validate().expect_err("mixed proposal must fail");
    assert!(matches!(error.check, WhitelistCheck::Denied { .. }));
}

#[test]
fn fitness_expectation_travels_with_the_proposal() {
    let proposal = proposal_with(vec![file("recipes/a.yaml")]);
    assert_eq!(proposal.fitness.baseline_score, 0.5);
    assert_eq!(proposal.source, ProposalSource::Session);
}

#[test]
fn validation_runs_again_at_begin_validation_time() {
    use goose_rsi::state::{BeginValidationError, ProposalStateMachine};
    let machine = ProposalStateMachine::new(proposal_with(vec![file("secrets.yaml")]));
    assert!(matches!(
        machine.begin_validation(),
        Err(BeginValidationError::Whitelist(_))
    ));
}
