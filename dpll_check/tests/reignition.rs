#![forbid(unsafe_code)]
#![deny(warnings)]

#[allow(dead_code)]
#[path = "../src/hybrid.rs"]
mod hybrid;

use hybrid::cdcl::{solve_cdcl, solve_cdcl_with_reignition, ReignitionConfig, SolveStats};
use hybrid::gradient::{GradientConfig, OracleWorkspace};
use hybrid::{solve_hybrid, HybridConfig};
use std::collections::{BTreeMap, BTreeSet};

fn oracle(epochs: usize) -> GradientConfig {
    GradientConfig {
        max_epochs: epochs,
        restarts: 1,
        check_every: 1,
        seed: 0x6174_b3e9_029c_85d1,
        ..GradientConfig::default()
    }
}

fn reignition(interval: usize, epochs: usize) -> ReignitionConfig {
    ReignitionConfig {
        interval,
        oracle: oracle(epochs),
    }
}

fn check_model(formula: &[Vec<i32>], model: &[i32]) {
    let mut values = BTreeMap::new();
    for &literal in model {
        assert!(literal != 0 && literal != i32::MIN);
        if let Some(previous) = values.insert(literal.unsigned_abs(), literal > 0) {
            assert_eq!(previous, literal > 0);
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
        .map(|v| v.unsigned_abs())
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

fn assert_budgets(stats: &SolveStats, epochs: usize) {
    assert!(stats.reignition_epochs <= stats.reignitions * epochs);
    assert!(stats.reignitions <= stats.conflicts);
    assert!(stats.reignition_ms.is_finite() && stats.reignition_ms >= 0.0);
}

fn counts(stats: &SolveStats) -> [usize; 13] {
    [
        stats.decisions,
        stats.propagations,
        stats.conflicts,
        stats.learned_clauses,
        stats.backjumps,
        stats.nonchronological_backjumps,
        stats.deleted_clauses,
        stats.database_reductions,
        stats.symmetry_rows,
        stats.symmetry_columns,
        stats.symmetry_clauses,
        stats.reignitions,
        stats.reignition_epochs,
    ]
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
    // Extra raw tautology avoids the strict recognized-family preprocessor.
    formula.push(vec![1, -1, 1]);
    formula
}

#[test]
fn conflict_feedback_reignites_without_compromising_unsat_proof() {
    let formula = pigeonhole(4, 3);
    let (model, stats) = solve_cdcl_with_reignition(&formula, None, 4.0, reignition(1, 2));
    assert!(model.is_none());
    assert!(stats.reignitions > 0);
    assert!(stats.learned_clauses > 0);
    assert!(stats.backjumps > 0);
    assert_eq!(stats.symmetry_clauses, 0);
    assert_budgets(&stats, 2);
}

#[test]
fn root_conflicts_do_not_request_a_continuous_reignition() {
    for formula in [vec![vec![]], vec![vec![1], vec![-1]]] {
        let (model, stats) = solve_cdcl_with_reignition(&formula, None, 4.0, reignition(1, 2));
        assert!(model.is_none());
        assert_eq!(stats.reignitions, 0);
        assert_eq!(stats.reignition_epochs, 0);
        assert_eq!(stats.reignition_ms, 0.0);
    }
}

#[test]
fn zero_interval_preserves_legacy_search_exactly() {
    let formulas = [
        pigeonhole(4, 3),
        vec![vec![1, 2], vec![-1, 3], vec![-2, -3]],
        vec![vec![1], vec![-1]],
        vec![],
    ];
    for formula in formulas {
        let (legacy_model, legacy_stats) = solve_cdcl(&formula, None, 4.0);
        let (model, stats) = solve_cdcl_with_reignition(&formula, None, 4.0, reignition(0, 2));
        assert_eq!(model, legacy_model);
        assert_eq!(counts(&stats), counts(&legacy_stats));
        assert_eq!(stats.reignitions, 0);
        assert_eq!(stats.reignition_ms, 0.0);
    }
}

#[test]
fn conflict_feedback_is_deterministic_in_models_and_counts() {
    let formula = pigeonhole(4, 3);
    let (first_model, first_stats) =
        solve_cdcl_with_reignition(&formula, None, 4.0, reignition(1, 3));
    let (second_model, second_stats) =
        solve_cdcl_with_reignition(&formula, None, 4.0, reignition(1, 3));
    assert_eq!(first_model, second_model);
    assert_eq!(counts(&first_stats), counts(&second_stats));
    assert!(first_stats.reignitions > 0);
    assert_budgets(&first_stats, 3);
    assert_budgets(&second_stats, 3);
}

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

#[test]
fn feedback_solver_matches_small_exhaustive_truth_tables() {
    let mut seed = 0x704b_8d52_13af_962e;
    for n in 1..=7 {
        for case in 0..24 {
            let clauses = (next(&mut seed) % 30) as usize;
            let formula: Vec<Vec<i32>> = (0..clauses)
                .map(|_| {
                    let width = if case % 17 == 0 {
                        next(&mut seed) % 5
                    } else {
                        next(&mut seed) % 3 + 1
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
            let ids: Vec<u32> = formula
                .iter()
                .flatten()
                .map(|v| v.unsigned_abs())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let guidance: Vec<(u32, f64)> = ids
                .into_iter()
                .enumerate()
                .map(|(index, id)| (id, [0.1, 0.5, 0.9][index % 3]))
                .collect();
            let (model, stats) =
                solve_cdcl_with_reignition(&formula, Some(&guidance), 4.0, reignition(1, 2));
            assert_eq!(model.is_some(), brute_force(&formula), "{formula:?}");
            if let Some(model) = model {
                check_model(&formula, &model);
            }
            assert_budgets(&stats, 2);
        }
    }
}

#[test]
fn assigned_coordinates_are_frozen_and_free_coordinates_can_move() {
    let formula = vec![vec![1, 3], vec![-2, 3], vec![-3], vec![3], vec![3]];
    let mut workspace = OracleWorkspace::new(&formula, &[0.2, 0.8, 0.4]);
    let epochs = workspace.reignite(&[1, -1, 0], oracle(2));
    assert!(epochs <= 2);
    assert_eq!(workspace.values()[0], 1.0);
    assert_eq!(workspace.values()[1], 0.0);
    assert!(workspace.values()[2] > 0.4);
    assert!(workspace
        .values()
        .iter()
        .all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
}

#[test]
fn backjumped_coordinates_are_not_permanently_clamped() {
    let formula = vec![vec![1], vec![-1], vec![-1]];
    let mut workspace = OracleWorkspace::new(&formula, &[0.8]);
    assert!(workspace.reignite(&[1], oracle(2)) <= 2);
    assert_eq!(workspace.values(), &[1.0]);
    assert!(workspace.reignite(&[0], oracle(2)) <= 2);
    assert!(workspace.values()[0] < 1.0);
}

#[test]
fn sparse_identifiers_and_repeated_literals_survive_feedback() {
    let high = i32::MAX;
    let cases = [
        vec![
            vec![high, 19],
            vec![-high, 19],
            vec![high, -19],
            vec![-high, -19],
        ],
        vec![vec![high, high], vec![-19, high], vec![19, -19, 19]],
    ];
    for formula in cases {
        let (model, stats) = solve_cdcl_with_reignition(&formula, None, 4.0, reignition(1, 2));
        assert_eq!(model.is_some(), brute_force(&formula));
        if let Some(model) = model {
            check_model(&formula, &model);
        }
        assert_budgets(&stats, 2);
    }
}

#[test]
fn hybrid_pipeline_exposes_feedback_counters_after_oracle_fallback() {
    let formula = pigeonhole(4, 3);
    let report = solve_hybrid(
        &formula,
        HybridConfig {
            oracle: oracle(0),
            activity_boost: 4.0,
            reignition_interval: 1,
            reignition_epochs: 2,
            ..HybridConfig::default()
        },
    );
    assert!(report.used_cdcl);
    assert!(report.model.is_none());
    assert!(report.cdcl.reignitions > 0);
    assert_budgets(&report.cdcl, 2);
}

#[test]
fn larger_conflict_run_preserves_learning_backjumps_and_clause_cleanup() {
    let formula = pigeonhole(6, 5);
    let (model, stats) = solve_cdcl_with_reignition(&formula, None, 4.0, reignition(4, 1));
    assert!(model.is_none());
    assert!(stats.reignitions > 0);
    assert!(stats.learned_clauses > 0 && stats.backjumps > 0);
    assert!(stats.database_reductions > 0 && stats.deleted_clauses > 0);
    assert_budgets(&stats, 1);
}
