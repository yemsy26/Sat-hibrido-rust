#![forbid(unsafe_code)]
#![deny(warnings)]

#[allow(dead_code)]
#[path = "../src/hybrid.rs"]
mod hybrid;

use hybrid::cdcl::{
    solve_cdcl, solve_cdcl_with_reignition, OracleConfig, ReignitionConfig, SolveStats,
};
use hybrid::{solve_hybrid, HybridConfig};
use std::collections::{BTreeMap, BTreeSet};

fn reignition(interval: usize, flips_per_variable: usize) -> ReignitionConfig {
    ReignitionConfig {
        interval,
        oracle: OracleConfig {
            flips_per_variable,
            seed: 0x6174_b3e9_029c_85d1,
        },
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

fn variable_ids(formula: &[Vec<i32>]) -> Vec<u32> {
    formula
        .iter()
        .flatten()
        .map(|v| v.unsigned_abs())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn brute_force(formula: &[Vec<i32>]) -> bool {
    let variables = variable_ids(formula);
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

fn assert_budgets(formula: &[Vec<i32>], stats: &SolveStats, flips_per_variable: usize) {
    let budget = flips_per_variable * variable_ids(formula).len();
    assert!(stats.reignition_flips <= stats.reignitions * budget);
    // One ignition at the start, then never more often than restarts happen.
    assert!(stats.reignitions <= stats.restarts + 1);
    assert!(stats.reignition_ms.is_finite() && stats.reignition_ms >= 0.0);
}

fn counts(stats: &SolveStats) -> [usize; 11] {
    [
        stats.decisions,
        stats.propagations,
        stats.conflicts,
        stats.learned_clauses,
        stats.backjumps,
        stats.nonchronological_backjumps,
        stats.restarts,
        stats.deleted_clauses,
        stats.database_reductions,
        stats.reignitions,
        stats.reignition_flips,
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
    // The CDCL entry points break no symmetries: pigeonhole stays a hard instance here.
    formula
}

#[test]
fn the_oracle_ignites_without_compromising_the_unsat_proof() {
    let formula = pigeonhole(4, 3);
    let (model, stats) = solve_cdcl_with_reignition(&formula, reignition(1, 2));
    assert!(model.is_none());
    assert!(stats.reignitions > 0);
    assert!(stats.learned_clauses > 0);
    assert!(stats.backjumps > 0);
    assert_budgets(&formula, &stats, 2);
}

#[test]
fn reignition_happens_only_at_the_start_and_at_restarts() {
    let formula = pigeonhole(7, 6);
    let (model, stats) = solve_cdcl_with_reignition(&formula, reignition(1, 1));
    assert!(model.is_none());
    assert!(stats.restarts > 0);
    assert!(stats.reignitions > 1, "Restarts must reignite the oracle");
    assert_budgets(&formula, &stats, 1);

    // A long interval leaves only the initial ignition.
    let (model, stats) = solve_cdcl_with_reignition(&formula, reignition(usize::MAX, 1));
    assert!(model.is_none());
    assert_eq!(stats.reignitions, 1);
}

#[test]
fn root_conflicts_do_not_request_a_reignition() {
    for formula in [vec![vec![]], vec![vec![1], vec![-1]]] {
        let (model, stats) = solve_cdcl_with_reignition(&formula, reignition(1, 2));
        assert!(model.is_none());
        assert_eq!(stats.reignitions, 0);
        assert_eq!(stats.reignition_flips, 0);
        assert_eq!(stats.reignition_ms, 0.0);
    }
}

#[test]
fn zero_interval_or_zero_flips_preserve_the_plain_search_exactly() {
    let formulas = [
        pigeonhole(4, 3),
        vec![vec![1, 2], vec![-1, 3], vec![-2, -3]],
        vec![vec![1], vec![-1]],
        vec![],
    ];
    for formula in formulas {
        let (plain_model, plain_stats) = solve_cdcl(&formula);
        for disabled in [reignition(0, 2), reignition(1, 0)] {
            let (model, stats) = solve_cdcl_with_reignition(&formula, disabled);
            assert_eq!(model, plain_model);
            assert_eq!(counts(&stats), counts(&plain_stats));
            assert_eq!(stats.reignitions, 0);
            assert_eq!(stats.reignition_ms, 0.0);
        }
    }
}

#[test]
fn reignited_search_is_deterministic_in_models_and_counts() {
    let formula = pigeonhole(6, 5);
    let (first_model, first_stats) = solve_cdcl_with_reignition(&formula, reignition(1, 3));
    let (second_model, second_stats) = solve_cdcl_with_reignition(&formula, reignition(1, 3));
    assert_eq!(first_model, second_model);
    assert_eq!(counts(&first_stats), counts(&second_stats));
    assert!(first_stats.reignitions > 0);
    assert_budgets(&formula, &first_stats, 3);
    assert_budgets(&formula, &second_stats, 3);
}

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

#[test]
fn reignited_solver_matches_small_exhaustive_truth_tables() {
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
            let flips = [1, 2, 9][case % 3];
            let (model, stats) = solve_cdcl_with_reignition(&formula, reignition(1, flips));
            assert_eq!(model.is_some(), brute_force(&formula), "{formula:?}");
            if let Some(model) = model {
                check_model(&formula, &model);
            }
            assert_budgets(&formula, &stats, flips);
        }
    }
}

#[test]
fn sparse_identifiers_and_repeated_literals_survive_reignition() {
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
        let (model, stats) = solve_cdcl_with_reignition(&formula, reignition(1, 2));
        assert_eq!(model.is_some(), brute_force(&formula));
        if let Some(model) = model {
            check_model(&formula, &model);
        }
        assert_budgets(&formula, &stats, 2);
    }
}

#[test]
fn hybrid_pipeline_exposes_the_oracle_counters() {
    let formula = pigeonhole(4, 3);
    let report = solve_hybrid(
        &formula,
        HybridConfig {
            oracle: reignition(1, 2).oracle,
            reignition_interval: 1,
            symmetry: false,
            parity: false,
        },
    );
    assert!(report.model.is_none());
    assert!(report.cdcl.reignitions > 0);
    assert_budgets(&formula, &report.cdcl, 2);
}

#[test]
fn larger_conflict_run_preserves_learning_backjumps_and_clause_cleanup() {
    let formula = pigeonhole(7, 6);
    let (model, stats) = solve_cdcl_with_reignition(&formula, reignition(4, 1));
    assert!(model.is_none());
    assert!(stats.reignitions > 0);
    assert!(stats.learned_clauses > 0 && stats.backjumps > 0);
    assert!(stats.database_reductions > 0 && stats.deleted_clauses > 0);
    assert_budgets(&formula, &stats, 1);
}

#[test]
fn satisfiable_instances_are_solved_by_the_oracle_or_by_the_search_it_seeds() {
    // Equivalence chain with a single model: a short walk rarely lands on it.
    let literals: Vec<i32> = (1..=14).map(|v| if v % 3 == 0 { -v } else { v }).collect();
    let mut formula = vec![literals.clone()];
    for pair in literals.windows(2) {
        formula.push(vec![-pair[0], pair[1]]);
        formula.push(vec![pair[0], -pair[1]]);
    }
    for flips in [1, 4, 256] {
        let (model, stats) = solve_cdcl_with_reignition(&formula, reignition(1, flips));
        let model = model.expect("The chain has exactly one model");
        assert_eq!(model, literals);
        assert_budgets(&formula, &stats, flips);
    }
}
