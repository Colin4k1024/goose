//! # goose-rsi
//!
//! Core data model and frozen whitelist validator for the RSI
//! (Recursive Self-Improvement) agent engine.
//!
//! The crate guarantees that any self-modification first passes a
//! fail-closed whitelist check before the proposal can advance through the
//! state machine:
//!
//! ```text
//! Proposed → Validating → Applied → Solidified | RolledBack
//! ```
//!
//! Invariants (issue #1):
//!
//! 1. Illegal state transitions are rejected both at the type level (the
//!    [`state::tag`] typestate API only exposes legal transitions) and at
//!    runtime ([`state::ProposalStateMachine::transit`] returns `Err`).
//! 2. A proposal whose mutation targets hit the frozen deny list fails
//!    [`proposal::MutationProposal::validate`] with an error, and validation
//!    performs no filesystem I/O whatsoever (it is a pure function).
//! 3. The whitelist/deny list are frozen constants. There is no API in this
//!    crate that can mutate them at runtime.
//!
//! Later milestones build on this crate: the fitness executor (issue #2),
//! the atomic apply/rollback layer (issue #3), the scheduled evolution loop
//! (issue #4) and the evolution log + CLI (issue #5). Those layers are out of
//! scope here: this crate defines the contract they must respect.

pub mod apply;
pub mod fitness;
pub mod proposal;
pub mod state;
pub mod whitelist;

pub use apply::{
    rollback_to_version, Applied, ApplyConfig, ApplyError, Change, Failpoint, RecoveryReport,
    RolledBack,
};
pub use proposal::{FitnessExpectation, MutationProposal, MutationTarget, ProposalSource};
pub use state::{
    tag, BeginValidationError, ProposalState, ProposalStateMachine, TransitionError,
    TransitionLegality,
};
pub use whitelist::{ValidationError, WhitelistCheck};
