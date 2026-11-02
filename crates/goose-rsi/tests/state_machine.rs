//! State machine acceptance tests (issue #1).
//!
//! Legal chain, illegal pairs rejected by `transit()`, and the atomic-CAS
//! concurrency guarantee: two owners can never both move the same proposal.

use goose_rsi::proposal::{FitnessExpectation, MutationProposal, MutationTarget, ProposalSource};
use goose_rsi::state::{tag, ProposalState, ProposalStateMachine, TransitionError};
use goose_rsi::whitelist::WhitelistCheck;
use std::sync::{Arc, Barrier};
use std::thread;

fn machine() -> ProposalStateMachine {
    ProposalStateMachine::new(MutationProposal::new(
        "state machine test",
        ProposalSource::Scheduled,
        vec![MutationTarget::FilePath {
            path: "recipes/a.yaml".to_string(),
        }],
        FitnessExpectation {
            baseline_score: 0.5,
            minimum_score: 0.6,
        },
    ))
}

fn chain_to_applied(machine: &ProposalStateMachine) {
    machine
        .transit(ProposalState::Proposed, ProposalState::Validating)
        .expect("proposed -> validating");
    machine
        .transit(ProposalState::Validating, ProposalState::Applied)
        .expect("validating -> applied");
}

#[test]
fn legal_chain_reaches_solidified() {
    let machine = machine();
    chain_to_applied(&machine);
    machine
        .transit(ProposalState::Applied, ProposalState::Solidified)
        .expect("applied -> solidified");
    assert_eq!(machine.state(), Ok(ProposalState::Solidified));
    assert!(ProposalState::Solidified.is_terminal());
}

#[test]
fn legal_chain_reaches_rolled_back() {
    let machine = machine();
    chain_to_applied(&machine);
    machine
        .transit(ProposalState::Applied, ProposalState::RolledBack)
        .expect("applied -> rolled back");
    assert!(ProposalState::RolledBack.is_terminal());
}

#[test]
fn illegal_pairs_return_err_from_transit() {
    let illegal = [
        (ProposalState::Proposed, ProposalState::Applied),
        (ProposalState::Proposed, ProposalState::Solidified),
        (ProposalState::Proposed, ProposalState::RolledBack),
        (ProposalState::Validating, ProposalState::Solidified),
        (ProposalState::Validating, ProposalState::Proposed),
        (ProposalState::Applied, ProposalState::Proposed),
        (ProposalState::Applied, ProposalState::Validating),
        (ProposalState::Solidified, ProposalState::Proposed),
        (ProposalState::Solidified, ProposalState::Applied),
        (ProposalState::RolledBack, ProposalState::Applied),
        (ProposalState::Proposed, ProposalState::Proposed),
    ];
    for (from, to) in illegal {
        let error = machine()
            .transit(from, to)
            .expect_err("illegal pair must be rejected");
        assert!(
            matches!(error, TransitionError::Illegal { .. }),
            "expected Illegal for {from:?} -> {to:?}, got {error:?}"
        );
    }
}

#[test]
fn typed_transit_follows_the_same_chain() {
    let machine = machine();
    assert_eq!(
        machine.transit_typed::<tag::Proposed, tag::Validating>(),
        Ok(ProposalState::Validating)
    );
    assert_eq!(
        machine.transit_typed::<tag::Validating, tag::Applied>(),
        Ok(ProposalState::Applied)
    );
    assert_eq!(
        machine.transit_typed::<tag::Applied, tag::Solidified>(),
        Ok(ProposalState::Solidified)
    );
}

#[test]
fn stale_owner_observes_concurrent_modification() {
    let machine = Arc::new(machine());
    let late_owner = Arc::clone(&machine);

    machine
        .transit(ProposalState::Proposed, ProposalState::Validating)
        .expect("first owner moves the proposal");

    let error = late_owner
        .transit(ProposalState::Proposed, ProposalState::Validating)
        .expect_err("second owner must lose the CAS");
    assert_eq!(
        error,
        TransitionError::ConcurrentModification {
            expected: ProposalState::Proposed,
            current: ProposalState::Validating,
        }
    );
}

#[test]
fn racing_owners_produce_exactly_one_winner() {
    const OWNERS: usize = 8;
    let machine = Arc::new(machine());
    let barrier = Arc::new(Barrier::new(OWNERS));
    let winners = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let losers = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    thread::scope(|scope| {
        for _ in 0..OWNERS {
            let machine = Arc::clone(&machine);
            let barrier = Arc::clone(&barrier);
            let winners = Arc::clone(&winners);
            let losers = Arc::clone(&losers);
            scope.spawn(move || {
                barrier.wait();
                match machine.transit(ProposalState::Proposed, ProposalState::Validating) {
                    Ok(_) => {
                        winners.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    }
                    Err(TransitionError::ConcurrentModification { .. }) => {
                        losers.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    }
                    Err(other) => panic!("unexpected error: {other:?}"),
                }
            });
        }
    });

    assert_eq!(winners.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(losers.load(std::sync::atomic::Ordering::SeqCst), OWNERS - 1);
}

#[test]
fn begin_validation_moves_only_valid_proposals() {
    let machine = ProposalStateMachine::new(MutationProposal::new(
        "blocked",
        ProposalSource::Scheduled,
        vec![MutationTarget::ConfigKey {
            key: "GOOSE_MODE".to_string(),
        }],
        FitnessExpectation {
            baseline_score: 0.5,
            minimum_score: 0.5,
        },
    ));
    let error = machine
        .begin_validation()
        .expect_err("deny-listed proposal must not enter Validating");
    assert!(matches!(
        error,
        goose_rsi::state::BeginValidationError::Whitelist(ref validation)
            if matches!(validation.check, WhitelistCheck::Denied { .. })
    ));
    // The proposal stays in Proposed.
    assert_eq!(machine.state(), Ok(ProposalState::Proposed));
}
