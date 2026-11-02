//! Proposal state machine: `Proposed → Validating → Applied → Solidified | RolledBack`.
//!
//! Two enforcement layers:
//!
//! * **Type level** — the [`tag`] typestate markers plus the sealed
//!   [`tag::TransitionFrom`] trait make illegal transitions inexpressible:
//!   `machine.transit_typed::<tag::Solidified, tag::Proposed>()` does not
//!   compile because `Proposed` does not implement
//!   `TransitionFrom<Solidified>`.
//! * **Runtime** — [`ProposalStateMachine::transit`] checks transition
//!   legality and performs an atomic compare-and-set, so a proposal can never
//!   be moved by two owners at once. A losing owner observes
//!   [`TransitionError::ConcurrentModification`].

use crate::proposal::MutationProposal;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU8, Ordering};

/// Lifecycle states of a mutation proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalState {
    /// Created, not yet validated against the whitelist.
    Proposed = 0,
    /// Whitelist-validated, being evaluated by the fitness executor.
    Validating = 1,
    /// Fitness gate passed and the mutation has been written atomically.
    Applied = 2,
    /// Fitness held up over evaluation; the mutation becomes the new baseline.
    Solidified = 3,
    /// Rolled back to the previous version (terminal).
    RolledBack = 4,
}

impl ProposalState {
    /// Terminal states can never transition again.
    pub const fn is_terminal(self) -> bool {
        matches!(self, ProposalState::Solidified | ProposalState::RolledBack)
    }

    /// The frozen transition table.
    pub const fn can_transit_to(self, to: ProposalState) -> bool {
        use ProposalState::*;
        matches!(
            (self, to),
            (Proposed, Validating)
                | (Validating, Applied)
                | (Applied, Solidified)
                | (Applied, RolledBack)
        )
    }

    pub const fn from_u8(value: u8) -> Option<ProposalState> {
        match value {
            0 => Some(ProposalState::Proposed),
            1 => Some(ProposalState::Validating),
            2 => Some(ProposalState::Applied),
            3 => Some(ProposalState::Solidified),
            4 => Some(ProposalState::RolledBack),
            _ => None,
        }
    }
}

/// Why a transition attempt failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TransitionError {
    /// The state pair is not in the frozen transition table.
    #[error("illegal transition {from:?} -> {to:?}")]
    Illegal {
        from: ProposalState,
        to: ProposalState,
    },
    /// Another owner already moved the proposal: the observed state differs
    /// from the expected one.
    #[error("concurrent modification: expected {expected:?}, current {current:?}")]
    ConcurrentModification {
        expected: ProposalState,
        current: ProposalState,
    },
    /// The in-memory state byte is not a known state.
    #[error("corrupted state byte: {0}")]
    Corrupted(u8),
}

/// Whether a state pair is legal. Marker trait alias kept for readability at
/// call sites that report transition legality.
pub type TransitionLegality = Result<(), TransitionError>;

/// Compile-time state markers. See [`tag::TransitionFrom`].
pub mod tag {
    use super::ProposalState;
    use std::fmt::Debug;

    mod sealed {
        pub trait Sealed {}
    }

    /// A zero-sized marker for one [`ProposalState`], enabling the typestate
    /// API. Implementations are sealed inside this module.
    pub trait StateTag: sealed::Sealed + Copy + Debug + 'static {
        const STATE: ProposalState;
    }

    macro_rules! state_tag {
        ($name:ident, $state:ident) => {
            /// Typestate marker.
            #[derive(Debug, Clone, Copy, PartialEq, Eq)]
            pub struct $name;

            impl sealed::Sealed for $name {}

            impl StateTag for $name {
                const STATE: ProposalState = ProposalState::$state;
            }
        };
    }

    state_tag!(Proposed, Proposed);
    state_tag!(Validating, Validating);
    state_tag!(Applied, Applied);
    state_tag!(Solidified, Solidified);
    state_tag!(RolledBack, RolledBack);

    /// Legal transitions, expressed as trait implementations. Only these
    /// pairs exist:
    ///
    /// * `Proposed → Validating`
    /// * `Validating → Applied`
    /// * `Applied → Solidified`
    /// * `Applied → RolledBack`
    ///
    /// Any other pair fails to compile, so illegal transitions are
    /// inexpressible through the typed API.
    pub trait TransitionFrom<From>: StateTag {}

    impl TransitionFrom<Proposed> for Validating {}
    impl TransitionFrom<Validating> for Applied {}
    impl TransitionFrom<Applied> for Solidified {}
    impl TransitionFrom<Applied> for RolledBack {}
}

/// A proposal plus its CAS-protected state, shareable across threads
/// (`Arc<ProposalStateMachine>`).
#[derive(Debug)]
pub struct ProposalStateMachine {
    proposal: MutationProposal,
    state: AtomicU8,
}

impl ProposalStateMachine {
    /// Wrap a proposal; it starts in [`ProposalState::Proposed`].
    pub fn new(proposal: MutationProposal) -> Self {
        ProposalStateMachine {
            proposal,
            state: AtomicU8::new(ProposalState::Proposed as u8),
        }
    }

    pub fn proposal(&self) -> &MutationProposal {
        &self.proposal
    }

    pub fn state(&self) -> Result<ProposalState, TransitionError> {
        ProposalState::from_u8(self.state.load(Ordering::Acquire)).ok_or(
            TransitionError::Corrupted(self.state.load(Ordering::Acquire)),
        )
    }

    /// Runtime transition with atomic compare-and-set semantics:
    ///
    /// * illegal state pairs → [`TransitionError::Illegal`];
    /// * the current state differs from `from` (another owner moved it) →
    ///   [`TransitionError::ConcurrentModification`];
    /// * otherwise the state becomes `to` atomically.
    ///
    /// # Examples
    ///
    /// ```
    /// use goose_rsi::proposal::{
    ///     FitnessExpectation, MutationProposal, MutationTarget, ProposalSource,
    /// };
    /// use goose_rsi::state::{ProposalState, ProposalStateMachine};
    ///
    /// let machine = ProposalStateMachine::new(MutationProposal::new(
    ///     "example",
    ///     ProposalSource::Session,
    ///     vec![MutationTarget::FilePath { path: "recipes/a.yaml".into() }],
    ///     FitnessExpectation { baseline_score: 0.5, minimum_score: 0.5 },
    /// ));
    ///
    /// // Illegal pair: Proposed may only move to Validating.
    /// assert!(machine
    ///     .transit(ProposalState::Proposed, ProposalState::Applied)
    ///     .is_err());
    ///
    /// // Legal pair, owner matches: succeeds.
    /// assert_eq!(
    ///     machine.transit(ProposalState::Proposed, ProposalState::Validating),
    ///     Ok(ProposalState::Validating)
    /// );
    ///
    /// // Same expectation again: the state already moved, CAS fails.
    /// assert!(machine
    ///     .transit(ProposalState::Proposed, ProposalState::Validating)
    ///     .is_err());
    /// ```
    pub fn transit(
        &self,
        from: ProposalState,
        to: ProposalState,
    ) -> Result<ProposalState, TransitionError> {
        if !from.can_transit_to(to) {
            return Err(TransitionError::Illegal { from, to });
        }
        self.compare_exchange(from, to)
    }

    /// Typed transition: compiles only for legal state pairs.
    ///
    /// # Examples
    ///
    /// Legal usage compiles and runs:
    ///
    /// ```
    /// use goose_rsi::state::{tag, ProposalStateMachine};
    /// # use goose_rsi::proposal::{
    /// #     FitnessExpectation, MutationProposal, MutationTarget, ProposalSource,
    /// # };
    /// # let machine = ProposalStateMachine::new(MutationProposal::new(
    /// #     "example",
    /// #     ProposalSource::Session,
    /// #     vec![MutationTarget::FilePath { path: "recipes/a.yaml".into() }],
    /// #     FitnessExpectation { baseline_score: 0.5, minimum_score: 0.5 },
    /// # ));
    /// assert_eq!(
    ///     machine.transit_typed::<tag::Proposed, tag::Validating>(),
    ///     Ok(goose_rsi::state::ProposalState::Validating)
    /// );
    /// ```
    ///
    /// Illegal transitions do not compile — `Solidified` is terminal, so
    /// `Proposed` does not implement `TransitionFrom<Solidified>`:
    ///
    /// ```compile_fail
    /// use goose_rsi::state::{tag, ProposalStateMachine};
    /// # use goose_rsi::proposal::{
    /// #     FitnessExpectation, MutationProposal, MutationTarget, ProposalSource,
    /// # };
    /// # let machine = ProposalStateMachine::new(MutationProposal::new(
    /// #     "example",
    /// #     ProposalSource::Session,
    /// #     vec![MutationTarget::FilePath { path: "recipes/a.yaml".into() }],
    /// #     FitnessExpectation { baseline_score: 0.5, minimum_score: 0.5 },
    /// # ));
    /// machine.transit_typed::<tag::Solidified, tag::Proposed>();
    /// ```
    pub fn transit_typed<From, To>(&self) -> Result<ProposalState, TransitionError>
    where
        From: tag::StateTag,
        To: tag::StateTag + tag::TransitionFrom<From>,
    {
        self.transit(From::STATE, To::STATE)
    }

    /// Fail-closed pipeline step used by the evolution loop: validate the
    /// proposal against the frozen whitelist and move `Proposed → Validating`
    /// atomically. A proposal that fails validation never leaves `Proposed`.
    pub fn begin_validation(&self) -> Result<ProposalState, BeginValidationError> {
        if let Err(error) = self.proposal.validate() {
            return Err(BeginValidationError::Whitelist(error));
        }
        self.transit(ProposalState::Proposed, ProposalState::Validating)
            .map_err(BeginValidationError::Transition)
    }
}

/// Error combining whitelist rejection and transition failure for
/// [`ProposalStateMachine::begin_validation`].
#[derive(Debug, thiserror::Error)]
pub enum BeginValidationError {
    #[error("whitelist rejected the proposal")]
    Whitelist(#[from] crate::whitelist::ValidationError),
    #[error("state transition failed")]
    Transition(#[from] TransitionError),
}

impl ProposalStateMachine {
    fn compare_exchange(
        &self,
        from: ProposalState,
        to: ProposalState,
    ) -> Result<ProposalState, TransitionError> {
        match self
            .state
            .compare_exchange(from as u8, to as u8, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => Ok(to),
            Err(current) => Err(TransitionError::ConcurrentModification {
                expected: from,
                current: ProposalState::from_u8(current)
                    .ok_or(TransitionError::Corrupted(current))?,
            }),
        }
    }
}
