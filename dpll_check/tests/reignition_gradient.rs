#![forbid(unsafe_code)]
#![deny(warnings)]

#[allow(dead_code)]
#[path = "../src/hybrid_gradient.rs"]
mod gradient;

use gradient::{CompiledFormula, GradientConfig, OracleWorkspace};

fn config(max_epochs: usize) -> GradientConfig {
    GradientConfig { max_epochs, restarts: 1, ..GradientConfig::default() }
}

#[test]
fn zero_updates_preserve_the_warm_tensor_and_apply_current_assignments() {
    let formula = vec![vec![1, 2, 3]];
    let mut workspace = OracleWorkspace::new(&formula, &[0.2, 0.3, 0.4]);
    let pointer = workspace.values().as_ptr();
    assert_eq!(workspace.reignite(&[0, -1, 1], config(0)), 0);
    assert_eq!(workspace.values(), &[0.2, 0.0, 1.0]);
    assert_eq!(workspace.values().as_ptr(), pointer);
    assert_eq!(workspace.reignite(&[0, 0, 0], config(0)), 0);
    assert_eq!(workspace.values(), &[0.2, 0.0, 1.0]);
}

#[test]
fn trail_variables_stay_fixed_even_when_their_units_prefer_the_opposite() {
    let formula = vec![vec![1], vec![-2], vec![3], vec![-4]];
    let mut workspace = OracleWorkspace::new(&formula, &[0.7, 0.3, 0.2, 0.8]);
    assert_eq!(workspace.reignite(&[-1, 1, 0, 0], config(64)), 64);
    assert_eq!(workspace.values()[0], 0.0);
    assert_eq!(workspace.values()[1], 1.0);
    assert!(workspace.values()[2] > 0.2);
    assert!(workspace.values()[3] < 0.8);
    assert!(workspace.values().iter().all(|value| value.is_finite() && (0.0..=1.0).contains(value)));
}

#[test]
fn clamped_variables_condition_the_gradient_of_free_variables() {
    let formula = vec![vec![1, 2]];
    let mut false_first = OracleWorkspace::new(&formula, &[0.5, 0.1]);
    let mut true_first = OracleWorkspace::new(&formula, &[0.5, 0.1]);
    false_first.reignite(&[-1, 0], config(1));
    true_first.reignite(&[1, 0], config(1));
    assert_eq!(false_first.values()[0], 0.0);
    assert_eq!(true_first.values()[0], 1.0);
    assert!(false_first.values()[1] > 0.1);
    assert_eq!(true_first.values()[1], 0.1);
}

#[test]
fn each_call_resets_adam_moments_and_bias_correction() {
    let formula = vec![vec![1, -2, 3], vec![-1, 2], vec![-3]];
    let mut reused = OracleWorkspace::new(&formula, &[0.2, 0.7, 0.8]);
    assert_eq!(reused.reignite(&[0, 0, 0], config(7)), 7);
    let previous_tensor = reused.values().to_vec();
    let mut fresh = OracleWorkspace::new(&formula, &previous_tensor);
    assert_eq!(reused.reignite(&[0, 0, 0], config(5)), 5);
    assert_eq!(fresh.reignite(&[0, 0, 0], config(5)), 5);
    assert_eq!(reused.values(), fresh.values());
}

#[test]
fn a_backtracked_variable_is_released_from_its_previous_fixed_value() {
    let formula = vec![vec![-1]];
    let mut workspace = OracleWorkspace::new(&formula, &[0.3]);
    assert_eq!(workspace.reignite(&[1], config(5)), 5);
    assert_eq!(workspace.values(), &[1.0]);
    assert_eq!(workspace.reignite(&[0], config(1)), 1);
    assert!(workspace.values()[0] < 1.0);
    assert!(workspace.values()[0] > 0.0);
    assert_eq!(workspace.reignite(&[-1], config(0)), 0);
    assert_eq!(workspace.values(), &[0.0]);
}

#[test]
fn sparse_ids_follow_original_compiled_order_and_reuse_the_tensor_buffer() {
    let formula = vec![vec![i32::MAX, -7], vec![7, 7], vec![17, -17]];
    let compiled = CompiledFormula::new(&formula);
    assert_eq!(compiled.variable_ids(), &[7, 17, i32::MAX as u32]);
    let mut workspace = OracleWorkspace::new(&formula, &[0.7, 0.2, 0.1]);
    let pointer = workspace.values().as_ptr();
    for updates in [0, 1, 10, 100] {
        assert_eq!(workspace.reignite(&[-1, 0, 1], config(updates)), updates);
        assert_eq!(workspace.values().as_ptr(), pointer);
        assert_eq!(workspace.values()[0], 0.0);
        assert_eq!(workspace.values()[2], 1.0);
        assert!(workspace.values().iter().all(|value| value.is_finite() && (0.0..=1.0).contains(value)));
    }
}

#[test]
fn fixed_inconsistent_branch_and_empty_clause_do_not_change_the_budget() {
    let mut workspace = OracleWorkspace::new(&[vec![1], vec![]], &[0.7]);
    assert_eq!(workspace.reignite(&[-1], config(100)), 100);
    assert_eq!(workspace.values(), &[0.0]);
    let mut empty = OracleWorkspace::new(&[], &[]);
    assert_eq!(empty.reignite(&[], config(3)), 3);
    assert!(empty.values().is_empty());
}

#[test]
#[should_panic(expected = "Initial tensor must match packed variables")]
fn constructor_rejects_an_incompatible_tensor_shape() {
    let _ = OracleWorkspace::new(&[vec![1, 2]], &[0.5]);
}

#[test]
#[should_panic]
fn constructor_rejects_nonfinite_tensor_values() {
    let _ = OracleWorkspace::new(&[vec![1]], &[f64::NAN]);
}

#[test]
#[should_panic(expected = "Trail must match packed variables")]
fn reignition_rejects_an_incompatible_assignment_shape() {
    let mut workspace = OracleWorkspace::new(&[vec![1]], &[0.5]);
    workspace.reignite(&[], config(1));
}

#[test]
#[should_panic]
fn reignition_rejects_nonboolean_trail_values() {
    let mut workspace = OracleWorkspace::new(&[vec![1]], &[0.5]);
    workspace.reignite(&[2], config(1));
}

#[test]
#[should_panic(expected = "Reignition is one warm-start trajectory")]
fn reignition_requires_one_trajectory() {
    let mut workspace = OracleWorkspace::new(&[vec![1]], &[0.5]);
    workspace.reignite(&[0], GradientConfig { restarts: 2, ..config(1) });
}
