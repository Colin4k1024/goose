//! Mutation proposal data model.

use crate::state::ProposalState;
use crate::whitelist::{self, ValidationError};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Where a mutation proposal originated from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalSource {
    /// Produced interactively during a goose session.
    Session,
    /// Produced by the scheduled evolution loop (issue #4).
    Scheduled,
}

/// The fitness expectation attached to a proposal. The apply layer (issue #3)
/// refuses to keep a proposal applied whose measured fitness score falls below
/// [`FitnessExpectation::minimum_score`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FitnessExpectation {
    /// Fitness score of the workspace baseline the proposal was built against.
    pub baseline_score: f64,
    /// Measured fitness must be >= this value for the proposal to stay applied.
    pub minimum_score: f64,
}

/// A single mutation surface targeted by a proposal.
///
/// File paths are POSIX-style paths relative to the goose user config
/// directory (e.g. `recipes/improve-search.yaml`). Config keys are dotted
/// keys inside `config.yaml` (e.g. `GOOSE_MODEL`). Secrets never travel as
/// config keys: they live in `secrets.yaml*`, which the deny list blocks as a
/// file path.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MutationTarget {
    FilePath { path: String },
    ConfigKey { key: String },
}

impl MutationTarget {
    /// Human-readable rendering used in error messages.
    pub fn display(&self) -> String {
        match self {
            MutationTarget::FilePath { path } => format!("file:{path}"),
            MutationTarget::ConfigKey { key } => format!("config:{key}"),
        }
    }
}

/// A proposed self-modification, validated against the frozen whitelist
/// before it may advance from `Proposed` to `Validating`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationProposal {
    pub id: Uuid,
    /// What the mutation changes and why (human-readable).
    pub description: String,
    pub source: ProposalSource,
    /// The complete set of mutation surfaces this proposal touches. The apply
    /// layer treats them as one atomic group (issue #3).
    pub targets: Vec<MutationTarget>,
    /// The fitness gate this proposal must clear (issue #2/#3).
    pub fitness: FitnessExpectation,
}

impl MutationProposal {
    /// Create a new proposal in the `Proposed` state.
    pub fn new(
        description: impl Into<String>,
        source: ProposalSource,
        targets: Vec<MutationTarget>,
        fitness: FitnessExpectation,
    ) -> Self {
        MutationProposal {
            id: Uuid::new_v4(),
            description: description.into(),
            source,
            targets,
            fitness,
        }
    }

    /// Fail-closed whitelist validation of every mutation target.
    ///
    /// This is a pure function: it never touches the filesystem, so a
    /// rejected proposal cannot leave partial writes behind.
    pub fn validate(&self) -> Result<(), ValidationError> {
        whitelist::validate_targets(&self.targets)
    }

    /// The state every freshly created proposal starts in.
    pub fn initial_state() -> ProposalState {
        ProposalState::Proposed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recipe_proposal() -> MutationProposal {
        MutationProposal::new(
            "tighten search recipe",
            ProposalSource::Session,
            vec![MutationTarget::FilePath {
                path: "recipes/search.yaml".to_string(),
            }],
            FitnessExpectation {
                baseline_score: 0.5,
                minimum_score: 0.5,
            },
        )
    }

    #[test]
    fn new_proposal_starts_proposed() {
        let proposal = recipe_proposal();
        assert_eq!(MutationProposal::initial_state(), ProposalState::Proposed);
        assert!(!proposal.id.is_nil());
    }

    #[test]
    fn serde_roundtrip_keeps_targets() {
        let proposal = recipe_proposal();
        let json = serde_json::to_string(&proposal).unwrap();
        let back: MutationProposal = serde_json::from_str(&json).unwrap();
        assert_eq!(back.targets, proposal.targets);
        assert_eq!(back.id, proposal.id);
    }
}
