#![forbid(unsafe_code)]
#![deny(warnings)]

#[allow(dead_code)]
#[path = "../src/hybrid.rs"]
mod hybrid;

use hybrid::gradient::{solve_gradient, CompiledFormula, GradientConfig};
use hybrid::{solve_hybrid, HybridConfig, HybridReport};
use std::collections::{BTreeMap, BTreeSet};

fn config(max_epochs: usize, activity_boost: f64) -> HybridConfig {
    HybridConfig {
        oracle: GradientConfig {
            max_epochs,
            restarts: 1,
            check_every: 1,
            seed: 0x1234_5678_9abc_def0,
            ..GradientConfig::default()
        },
        activity_boost,
        ..HybridConfig::default()
    }
}

fn check_model(formula: &[Vec<i32>], model: &[i32]) {
    let mut values = BTreeMap::new();
    for &literal in model {
        assert!(literal != 0 && literal != i32::MIN);
        if let Some(previous) = values.insert(literal.unsigned_abs(), literal > 0) {
            assert_eq!(previous, literal > 0, "Inconsistent returned model");
        }
    }
    assert!(
        formula.iter().all(|clause| clause
            .iter()
            .any(|&literal| { values.get(&literal.unsigned_abs()) == Some(&(literal > 0)) })),
        "Returned model does not satisfy the original clauses"
    );
}

fn brute_force(formula: &[Vec<i32>]) -> bool {
    let variables: Vec<u32> = formula
        .iter()
        .flatten()
        .map(|literal| literal.unsigned_abs())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    assert!(variables.len() <= 20);
    (0..1_u64 << variables.len()).any(|bits| {
        formula.iter().all(|clause| {
            clause.iter().any(|&literal| {
                let index = variables.binary_search(&literal.unsigned_abs()).unwrap();
                (bits & (1_u64 << index) != 0) == (literal > 0)
            })
        })
    })
}

fn check_report(formula: &[Vec<i32>], report: &HybridReport, budget: usize) {
    assert_eq!(report.model.is_some(), brute_force(formula), "{formula:?}");
    if let Some(model) = &report.model {
        check_model(formula, model);
    }
    assert!(report.oracle.epochs <= budget);
    assert!(report.oracle_ms.is_finite() && report.oracle_ms >= 0.0);
    assert!(report.cdcl_ms.is_finite() && report.cdcl_ms >= 0.0);
    assert!(report
        .oracle
        .continuous_values
        .iter()
        .all(|value| { value.is_finite() && (0.0..=1.0).contains(value) }));
    if !report.used_cdcl {
        assert!(report.oracle.model.is_some());
        assert!(report.model.is_some());
        assert_eq!(report.cdcl.decisions, 0);
        assert_eq!(report.cdcl.conflicts, 0);
    }
}

#[test]
fn empty_formula_uses_verified_oracle_fast_path() {
    let formula = vec![];
    let report = solve_hybrid(&formula, config(0, 4.0));
    check_report(&formula, &report, 0);
    assert!(!report.used_cdcl);
    assert!(report.model.as_ref().unwrap().is_empty());
}

#[test]
fn unknown_oracle_is_never_treated_as_unsat_without_cdcl() {
    let formula = vec![vec![1], vec![-1]];
    let report = solve_hybrid(&formula, config(0, 4.0));
    check_report(&formula, &report, 0);
    assert!(report.used_cdcl);
    assert!(report.oracle.model.is_none());
    assert!(report.model.is_none());
}

#[test]
fn zero_epoch_oracle_falls_back_and_finds_a_model() {
    let parameters = config(0, 4.0);
    let scout_formula: Vec<Vec<i32>> = (1..=6).map(|v| vec![v, -v]).collect();
    let scout = solve_gradient(&scout_formula, parameters.oracle);
    // Same variable IDs and seed, but every unit opposes the initial tensor.
    let formula: Vec<Vec<i32>> = scout
        .continuous_values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let variable = (index + 1) as i32;
            vec![if value >= 0.5 { -variable } else { variable }]
        })
        .collect();
    let report = solve_hybrid(&formula, parameters);
    check_report(&formula, &report, 0);
    assert!(report.used_cdcl);
    assert!(report.oracle.model.is_none());
    assert!(report.model.is_some());
    assert_eq!(report.oracle.continuous_values, scout.continuous_values);
}

#[test]
fn empty_clause_tautologies_and_repeated_literals_remain_exact() {
    let cases = [
        vec![vec![]],
        vec![vec![1, -1], vec![2, 2, 2]],
        vec![vec![1, 1], vec![-1, -1, -1]],
        vec![vec![1, -1], vec![2, 2], vec![-2, 3], vec![-3, 4], vec![-4]],
        vec![vec![1, -1, 1], vec![2, -2, 2], vec![-3, -3, 4], vec![3, 4]],
    ];
    for formula in cases {
        for epochs in [0, 7] {
            let report = solve_hybrid(&formula, config(epochs, 4.0));
            check_report(&formula, &report, epochs);
        }
    }
}

#[test]
fn sparse_identifiers_roundtrip_without_dense_allocation() {
    let high = i32::MAX;
    let cases = [
        vec![vec![high], vec![-19, high], vec![19, 19], vec![7, -7]],
        vec![
            vec![high, 19],
            vec![-high, 19],
            vec![high, -19],
            vec![-high, -19],
        ],
    ];
    for formula in cases {
        let report = solve_hybrid(&formula, config(0, 32.0));
        check_report(&formula, &report, 0);
        assert_eq!(
            report.oracle.continuous_values.len(),
            formula
                .iter()
                .flatten()
                .map(|v| v.unsigned_abs())
                .collect::<BTreeSet<_>>()
                .len()
        );
    }
}

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

#[test]
fn deterministic_generated_formulas_match_exhaustive_truth_tables() {
    let mut seed = 0x8a5c_920e_7612_43d9;
    for n in 1..=7 {
        for case in 0..36 {
            let clauses = (next(&mut seed) % 29) as usize;
            let formula: Vec<Vec<i32>> = (0..clauses)
                .map(|_| {
                    let width = if case % 19 == 0 {
                        (next(&mut seed) % 5) as usize
                    } else {
                        (next(&mut seed) % 4 + 1) as usize
                    };
                    (0..width)
                        .map(|_| {
                            let variable = (next(&mut seed) % n + 1) as i32;
                            if next(&mut seed) & 1 == 0 {
                                variable
                            } else {
                                -variable
                            }
                        })
                        .collect()
                })
                .collect();
            let epochs = [0, 3, 7][case % 3];
            let boost = [0.0, 4.0, 1000.0][case % 3];
            let report = solve_hybrid(&formula, config(epochs, boost));
            check_report(&formula, &report, epochs);
        }
    }
}

#[test]
fn every_assignment_excluded_is_proved_unsat_after_oracle_budget() {
    for n in 1..=5 {
        let formula: Vec<Vec<i32>> = (0..1_usize << n)
            .map(|bits| {
                (0..n)
                    .map(|index| {
                        let variable = (index + 1) as i32;
                        if bits & (1 << index) == 0 {
                            variable
                        } else {
                            -variable
                        }
                    })
                    .collect()
            })
            .collect();
        let report = solve_hybrid(&formula, config(5, 4.0));
        check_report(&formula, &report, 5);
        assert!(report.used_cdcl);
        assert!(report.model.is_none());
        assert!(report.cdcl.conflicts > 0);
    }
}

fn pigeonhole(pigeons: usize, holes: usize) -> Vec<Vec<i32>> {
    let mut formula: Vec<Vec<i32>> = (0..pigeons)
        .map(|pigeon| {
            (0..holes)
                .map(|hole| (pigeon * holes + hole + 1) as i32)
                .collect()
        })
        .collect();
    for hole in 0..holes {
        for first in 0..pigeons {
            for second in first + 1..pigeons {
                formula.push(vec![
                    -((first * holes + hole + 1) as i32),
                    -((second * holes + hole + 1) as i32),
                ]);
            }
        }
    }
    formula
}

#[test]
fn small_pigeonhole_is_exact_with_and_without_symmetry_recognition() {
    for (pigeons, holes) in [(2, 2), (3, 2), (3, 3), (4, 3)] {
        for asymmetric in [false, true] {
            let mut formula = pigeonhole(pigeons, holes);
            if asymmetric {
                // Raw extra tautology disables the strict recognized-family preprocessor.
                formula.push(vec![1, -1, 1]);
            }
            let report = solve_hybrid(&formula, config(5, 4.0));
            check_report(&formula, &report, 5);
            assert_eq!(report.model.is_some(), pigeons <= holes);
            if pigeons > holes {
                assert!(report.used_cdcl);
            }
        }
    }
}

#[test]
fn hybrid_search_model_and_search_counts_are_deterministic() {
    let mut formula = pigeonhole(4, 3);
    formula.push(vec![1, -1, 1]);
    let parameters = config(17, 11.0);
    let first = solve_hybrid(&formula, parameters);
    let second = solve_hybrid(&formula, parameters);
    check_report(&formula, &first, 17);
    check_report(&formula, &second, 17);
    assert_eq!(first.model, second.model);
    assert_eq!(first.oracle, second.oracle);
    assert_eq!(first.used_cdcl, second.used_cdcl);
    assert_eq!(first.cdcl.decisions, second.cdcl.decisions);
    assert_eq!(first.cdcl.conflicts, second.cdcl.conflicts);
    assert_eq!(first.cdcl.propagations, second.cdcl.propagations);
    assert_eq!(first.cdcl.learned_clauses, second.cdcl.learned_clauses);
    assert_eq!(first.cdcl.backjumps, second.cdcl.backjumps);
    assert_eq!(
        first.cdcl.nonchronological_backjumps,
        second.cdcl.nonchronological_backjumps
    );
}

#[test]
fn oracle_statistics_describe_the_returned_tensor() {
    let formula = vec![vec![1], vec![1], vec![-1], vec![2], vec![-2]];
    let mut parameters = config(11, 4.0);
    parameters.oracle.restarts = 3;
    let report = solve_hybrid(&formula, parameters);
    check_report(&formula, &report, 11);
    assert!(report.used_cdcl);
    assert_eq!(report.oracle.epochs, 11);
    let compiled = CompiledFormula::new(&formula);
    assert_eq!(compiled.variable_ids(), &report.oracle.variable_ids);
    let mut gradient = vec![0.0; report.oracle.continuous_values.len()];
    let loss = compiled.loss_and_gradient(&report.oracle.continuous_values, &mut gradient);
    let norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
    assert!((loss - report.oracle.loss).abs() < 1.0e-12);
    assert!((norm - report.oracle.gradient_l2).abs() < 1.0e-12);
    let rounded: Vec<bool> = report
        .oracle
        .continuous_values
        .iter()
        .map(|&v| v >= 0.5)
        .collect();
    assert_eq!(
        compiled.violations(&rounded),
        report.oracle.violated_clauses
    );
}

#[test]
fn loose_numerical_threshold_cannot_create_an_unsound_sat_answer() {
    let formula = vec![vec![1], vec![-1]];
    let mut parameters = config(2, 4.0);
    parameters.oracle.loss_threshold = 10.0;
    let report = solve_hybrid(&formula, parameters);
    check_report(&formula, &report, 2);
    assert!(report.used_cdcl);
    assert!(report.model.is_none());
}

#[test]
fn maximal_finite_activity_boost_preserves_exactness() {
    let cases = [
        vec![vec![1, 2]],
        vec![vec![-1, -2]],
        vec![vec![1, -1]],
        vec![vec![1, 2], vec![-1, 2], vec![1, -2], vec![-1, -2]],
    ];
    for formula in cases {
        for value in [0.0, 0.5, 1.0] {
            let guidance = [(1, value), (2, value)];
            let (model, _) = hybrid::cdcl::solve_cdcl(&formula, Some(&guidance), f64::MAX);
            assert_eq!(model.is_some(), brute_force(&formula));
            if let Some(model) = model {
                check_model(&formula, &model);
            }
        }
    }

    let parameters = config(0, f64::MAX);
    let scaffold: Vec<Vec<i32>> = (1..=6).map(|v| vec![v, -v]).collect();
    let tensor = solve_gradient(&scaffold, parameters.oracle).continuous_values;
    let units: Vec<Vec<i32>> = tensor.iter().enumerate().map(|(index, &value)| {
        let id = (index + 1) as i32;
        vec![if value >= 0.5 { -id } else { id }]
    }).collect();
    let report = solve_hybrid(&units, parameters);
    check_report(&units, &report, 0);
    assert!(report.used_cdcl);
}

#[test]
fn confidence_and_polarity_priors_control_the_initial_branch() {
    let formula = vec![vec![1, 2]];
    for (value, signed_variable) in [(0.85, 1), (0.15, -1)] {
        let guidance = [(1, value), (2, 0.5)];
        let (model, _) = hybrid::cdcl::solve_cdcl(&formula, Some(&guidance), 1000.0);
        let model = model.unwrap();
        check_model(&formula, &model);
        assert!(model.contains(&signed_variable));
    }

    // Choosing confident x2=false first makes x1 a unit: exactly one decision.
    // Choosing the earlier variable x1=true first requires two decisions.
    let guidance = [(1, 0.5), (2, 0.01)];
    let (model, stats) = hybrid::cdcl::solve_cdcl(&formula, Some(&guidance), 1000.0);
    let model = model.unwrap();
    check_model(&formula, &model);
    assert!(model.contains(&-2));
    assert_eq!(stats.decisions, 1);
}
