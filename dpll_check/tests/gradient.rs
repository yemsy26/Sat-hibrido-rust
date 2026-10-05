#![forbid(unsafe_code)]
#![deny(warnings)]

#[allow(dead_code)]
#[path = "../src/gradient.rs"]
mod gradient;

use gradient::{solve_gradient, CompiledFormula, Formula, GradientConfig, Outcome};
use std::collections::BTreeMap;

fn config(max_epochs: usize) -> GradientConfig {
    GradientConfig {
        max_epochs,
        restarts: 2,
        check_every: 1,
        seed: 0x1234_5678_9abc_def0,
        ..GradientConfig::default()
    }
}

fn check_model(formula: &[Vec<i32>], model: &[i32]) {
    let mut values = BTreeMap::new();
    for &literal in model {
        assert_ne!(literal, 0);
        if let Some(previous) = values.insert(literal.unsigned_abs(), literal > 0) {
            assert_eq!(previous, literal > 0, "Conflicting model assignments");
        }
    }
    assert!(formula.iter().all(|clause| clause.iter().any(|&literal| {
        values.get(&literal.unsigned_abs()) == Some(&(literal > 0))
    })), "Returned model does not satisfy the original formula");

    let compiled = CompiledFormula::new(formula);
    let packed: Vec<bool> = compiled
        .variable_ids()
        .iter()
        .map(|variable| *values.get(variable).expect("Model includes every variable"))
        .collect();
    assert!(compiled.verify(&packed));
}

fn loss(compiled: &CompiledFormula, values: &[f64]) -> f64 {
    compiled.loss_and_gradient(values, &mut vec![0.0; values.len()])
}

fn check_finite_difference(formula: &[Vec<i32>], points: &[Vec<f64>]) {
    let compiled = CompiledFormula::new(formula);
    for values in points {
        assert_eq!(values.len(), compiled.variable_ids().len());
        let mut analytical = vec![f64::NAN; values.len()];
        let base = compiled.loss_and_gradient(values, &mut analytical);
        assert!(base.is_finite());
        for variable in 0..values.len() {
            assert!(analytical[variable].is_finite());
            let h = 1.0e-6;
            let mut lower = values.clone();
            let mut upper = values.clone();
            lower[variable] = (lower[variable] - h).max(0.0);
            upper[variable] = (upper[variable] + h).min(1.0);
            let numerical = (loss(&compiled, &upper) - loss(&compiled, &lower))
                / (upper[variable] - lower[variable]);
            let tolerance = 2.0e-5 * (1.0 + analytical[variable].abs());
            assert!(
                (analytical[variable] - numerical).abs() <= tolerance,
                "variable={variable}, point={values:?}, analytical={}, numerical={numerical}",
                analytical[variable]
            );
        }
    }
}

#[test]
fn mixed_sign_loss_and_gradient_match_the_clause_polynomial() {
    let compiled = CompiledFormula::new(&[vec![1, -2, 3]]);
    let mut gradient = vec![0.0; 3];
    let actual = compiled.loss_and_gradient(&[0.2, 0.3, 0.4], &mut gradient);
    assert!((actual - 0.8 * 0.3 * 0.6).abs() < 1.0e-14);
    let expected = [-0.3 * 0.6, 0.8 * 0.6, -0.8 * 0.3];
    for (actual, expected) in gradient.iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-14);
    }
}

#[test]
fn gradients_match_finite_differences_at_interior_and_boundary_points() {
    let formula = vec![
        vec![1, -2, 3],
        vec![-1, 2],
        vec![2, -3],
        vec![-1],
        vec![3],
    ];
    check_finite_difference(
        &formula,
        &[
            vec![0.2, 0.3, 0.4],
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
        ],
    );
}

#[test]
fn zero_factors_do_not_produce_nan_or_drop_nonzero_derivatives() {
    let compiled = CompiledFormula::new(&[vec![1, 2, -3]]);
    let mut gradient = vec![f64::NAN; 3];
    assert_eq!(compiled.loss_and_gradient(&[1.0, 0.0, 1.0], &mut gradient), 0.0);
    assert_eq!(gradient, vec![-1.0, 0.0, 0.0]);
    assert_eq!(compiled.loss_and_gradient(&[1.0, 1.0, 1.0], &mut gradient), 0.0);
    assert_eq!(gradient, vec![0.0, 0.0, 0.0]);
}

#[test]
fn duplicate_literals_and_tautologies_have_consistent_gradients() {
    let formula = vec![
        vec![1, 1, -2],
        vec![-1, 2, 2],
        vec![3, -3],
        vec![4],
        vec![-4],
    ];
    check_finite_difference(
        &formula,
        &[
            vec![0.2, 0.3, 0.4, 0.7],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0, 0.0],
        ],
    );
}

#[test]
fn zero_loss_at_boolean_corners_matches_original_clause_semantics() {
    let literals = [-3_i32, -2, -1, 1, 2, 3];
    for &a in &literals {
        for &b in &literals {
            for &c in &literals {
                let formula = vec![vec![a, b, c]];
                let compiled = CompiledFormula::new(&formula);
                for bits in 0..8 {
                    let expected = formula[0].iter().any(|&literal| {
                        let value = bits & (1 << (literal.unsigned_abs() - 1)) != 0;
                        value == (literal > 0)
                    });
                    let boolean: Vec<bool> = compiled
                        .variable_ids()
                        .iter()
                        .map(|&variable| bits & (1 << (variable - 1)) != 0)
                        .collect();
                    let continuous: Vec<f64> = boolean
                        .iter()
                        .map(|&value| f64::from(u8::from(value)))
                        .collect();
                    assert_eq!(compiled.verify(&boolean), expected);
                    assert_eq!(loss(&compiled, &continuous) == 0.0, expected);
                }
            }
        }
    }
}

#[test]
fn empty_clause_has_constant_loss_and_returns_unknown() {
    let formula = vec![Vec::new()];
    let compiled = CompiledFormula::new(&formula);
    assert!(compiled.variable_ids().is_empty());
    assert_eq!(compiled.loss_and_gradient(&[], &mut []), 1.0);
    assert!(!compiled.verify(&[]));
    let report = solve_gradient(&formula, config(100));
    assert!(matches!(report.outcome, Outcome::Unknown));
    assert!(report.loss.is_finite());
    assert!(report.gradient_l2.is_finite());
}

#[test]
fn empty_formula_and_simple_units_produce_verified_models() {
    for formula in [
        Formula::new(),
        vec![vec![1]],
        vec![vec![-1]],
        vec![vec![1], vec![-2], vec![3]],
    ] {
        let report = solve_gradient(&formula, config(1000));
        match report.outcome {
            Outcome::Sat(model) => check_model(&formula, &model),
            Outcome::Unknown => panic!("Simple formula was not solved: {formula:?}"),
        }
    }
}

#[test]
fn contradictory_units_never_report_sat() {
    let formula = vec![vec![1], vec![-1]];
    let compiled = CompiledFormula::new(&formula);
    let mut gradient = [f64::NAN];
    for value in [0.0, 0.2, 0.5, 0.9, 1.0] {
        assert_eq!(compiled.loss_and_gradient(&[value], &mut gradient), 1.0);
        assert_eq!(gradient, [0.0]);
    }
    for seed in 1..=8 {
        let report = solve_gradient(&formula, GradientConfig { seed, ..config(100) });
        assert!(matches!(report.outcome, Outcome::Unknown));
        assert!(report.violated_clauses > 0);
    }
}

#[test]
fn unsatisfiable_pigeonhole_returns_unknown() {
    let mut formula = Formula::new();
    for pigeon in 0..4 {
        formula.push((0..3).map(|hole| pigeon * 3 + hole + 1).collect());
    }
    for hole in 0..3 {
        for first in 0..4 {
            for second in first + 1..4 {
                formula.push(vec![-(first * 3 + hole + 1), -(second * 3 + hole + 1)]);
            }
        }
    }
    let report = solve_gradient(&formula, config(500));
    assert!(matches!(report.outcome, Outcome::Unknown));
    assert!(report.violated_clauses > 0);
}

#[test]
fn an_epoch_limit_is_not_an_unsatisfiability_certificate() {
    let formula = vec![vec![1], vec![-1]];
    for max_epochs in [0, 1] {
        let report = solve_gradient(&formula, GradientConfig {
            max_epochs,
            restarts: 1,
            ..config(max_epochs)
        });
        assert!(matches!(report.outcome, Outcome::Unknown));
    }
}

#[test]
fn identical_seed_and_configuration_reproduce_reports() {
    let formula = vec![vec![1, -2, 3], vec![-1, 2], vec![-3, -2], vec![2, 3]];
    let a = solve_gradient(&formula, config(100));
    let b = solve_gradient(&formula, config(100));
    assert_eq!(a.epochs, b.epochs);
    assert_eq!(a.restarts, b.restarts);
    assert_eq!(a.loss.to_bits(), b.loss.to_bits());
    assert_eq!(a.gradient_l2.to_bits(), b.gradient_l2.to_bits());
    assert_eq!(a.violated_clauses, b.violated_clauses);
    assert_eq!(a.learning_rate.to_bits(), b.learning_rate.to_bits());
    match (a.outcome, b.outcome) {
        (Outcome::Sat(first), Outcome::Sat(second)) => {
            assert_eq!(first, second);
            check_model(&formula, &first);
        }
        (Outcome::Unknown, Outcome::Unknown) => {}
        _ => panic!("Identical deterministic executions returned different outcomes"),
    }
}

#[test]
fn sparse_maximum_variable_ids_are_packed_without_large_allocations() {
    let formula = vec![vec![i32::MAX], vec![-7]];
    let compiled = CompiledFormula::new(&formula);
    assert_eq!(compiled.variable_ids(), &[7, i32::MAX as u32]);
    assert!(compiled.verify(&[false, true]));
    assert!(!compiled.verify(&[true, true]));
    let report = solve_gradient(&formula, config(1000));
    if let Outcome::Sat(model) = report.outcome {
        check_model(&formula, &model);
    }
}

#[test]
fn every_sat_output_satisfies_the_original_formula_even_with_a_loose_loss_threshold() {
    let mut random = 0x6543_210f_abcd_9876_u64;
    for _ in 0..256 {
        let mut formula = Formula::new();
        for _ in 0..12 {
            let mut clause = Vec::new();
            for _ in 0..3 {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                let variable = (random % 5 + 1) as i32;
                clause.push(if random & 32 == 0 { variable } else { -variable });
            }
            formula.push(clause);
        }
        let report = solve_gradient(&formula, GradientConfig {
            loss_threshold: 1000.0,
            restarts: 1,
            ..config(20)
        });
        assert!(report.loss.is_finite());
        assert!(report.gradient_l2.is_finite());
        if let Outcome::Sat(model) = report.outcome {
            check_model(&formula, &model);
        }
    }
}
