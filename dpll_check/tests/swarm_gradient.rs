#![forbid(unsafe_code)]
#![deny(warnings)]

#[allow(dead_code)]
#[path = "../src/hybrid_gradient.rs"]
mod gradient;

use gradient::{GradientConfig, OracleWorkspace};
use std::sync::atomic::{AtomicBool, Ordering};

fn config(max_epochs: usize) -> GradientConfig {
    GradientConfig { max_epochs, restarts: 1, ..GradientConfig::default() }
}

#[test]
fn precancelled_workspace_performs_no_updates_and_clamps_the_current_trail() {
    let formula = vec![vec![1], vec![-2], vec![3, 4]];
    let mut workspace = OracleWorkspace::new(&formula, &[0.2, 0.7, 0.4, 0.8]);
    let pointer = workspace.values().as_ptr();
    let cancel = AtomicBool::new(true);
    assert_eq!(workspace.reignite_with_cancel(&[1, -1, 0, 0], config(100), &cancel), (0, true));
    assert_eq!(workspace.values(), &[1.0, 0.0, 0.4, 0.8]);
    assert_eq!(workspace.values().as_ptr(), pointer);
}

#[test]
fn a_false_cancel_flag_matches_legacy_tensor_and_exact_epoch_counts() {
    let formula = vec![vec![1, -2, 3], vec![-1, 2], vec![-3], vec![3, 3]];
    let mut legacy = OracleWorkspace::new(&formula, &[0.2, 0.7, 0.8]);
    let mut cancellable = OracleWorkspace::new(&formula, &[0.2, 0.7, 0.8]);
    let cancel = AtomicBool::new(false);
    let pointer = cancellable.values().as_ptr();
    for (assignments, epochs) in [
        ([0, 0, 0], 0),
        ([-1, 0, 0], 1),
        ([0, 1, 0], 7),
        ([0, 0, 0], 100),
        ([1, -1, 1], 5),
    ] {
        assert_eq!(legacy.reignite(&assignments, config(epochs)), epochs);
        assert_eq!(cancellable.reignite_with_cancel(&assignments, config(epochs), &cancel), (epochs, false));
        assert_eq!(legacy.values(), cancellable.values());
        assert_eq!(cancellable.values().as_ptr(), pointer);
    }
}

#[test]
fn zero_budget_still_reports_cancellation_and_preserves_clamped_values() {
    let mut workspace = OracleWorkspace::new(&[vec![1, 2]], &[0.2, 0.7]);
    let cancel = AtomicBool::new(true);
    assert_eq!(workspace.reignite_with_cancel(&[-1, 0], config(0), &cancel), (0, true));
    assert_eq!(workspace.values(), &[0.0, 0.7]);
    cancel.store(false, Ordering::Release);
    assert_eq!(workspace.reignite_with_cancel(&[0, 0], config(0), &cancel), (0, false));
    assert_eq!(workspace.values(), &[0.0, 0.7]);
}

#[test]
fn cancellation_does_not_poison_a_later_warm_start() {
    let formula = vec![vec![-1, 2], vec![1, 3], vec![-3]];
    let mut reused = OracleWorkspace::new(&formula, &[0.3, 0.4, 0.8]);
    let cancel = AtomicBool::new(true);
    assert_eq!(reused.reignite_with_cancel(&[1, 0, -1], config(5), &cancel), (0, true));
    let mut fresh = OracleWorkspace::new(&formula, reused.values());
    cancel.store(false, Ordering::Release);
    assert_eq!(reused.reignite_with_cancel(&[0, 0, 0], config(3), &cancel), (3, false));
    assert_eq!(fresh.reignite(&[0, 0, 0], config(3)), 3);
    assert_eq!(reused.values(), fresh.values());
}

#[test]
fn cancellation_of_an_empty_tensor_is_not_a_logical_answer() {
    let mut workspace = OracleWorkspace::new(&[Vec::new()], &[]);
    let cancel = AtomicBool::new(true);
    assert_eq!(workspace.reignite_with_cancel(&[], config(100), &cancel), (0, true));
    assert!(workspace.values().is_empty());
}
