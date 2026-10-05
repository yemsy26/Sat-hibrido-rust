#![forbid(unsafe_code)]
#![deny(warnings)]

#[allow(dead_code)]
#[path = "../src/hybrid.rs"]
mod hybrid;

use hybrid::cdcl::{solve_cdcl, OracleConfig};
use hybrid::{solve_hybrid, HybridConfig, HybridReport};
use std::collections::{BTreeMap, BTreeSet};

fn config(flips_per_variable: usize, reignition_interval: usize) -> HybridConfig {
    HybridConfig {
        oracle: OracleConfig {
            flips_per_variable,
            seed: 0x1234_5678_9abc_def0,
        },
        reignition_interval,
        symmetry: true,
        parity: true,
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

fn variable_ids(formula: &[Vec<i32>]) -> Vec<u32> {
    formula
        .iter()
        .flatten()
        .map(|literal| literal.unsigned_abs())
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

fn check_report(formula: &[Vec<i32>], report: &HybridReport, parameters: HybridConfig) {
    assert_eq!(report.model.is_some(), brute_force(formula), "{formula:?}");
    if let Some(model) = &report.model {
        check_model(formula, model);
    }
    assert!(report.cdcl_ms.is_finite() && report.cdcl_ms >= 0.0);
    assert!(report.cdcl.reignition_ms.is_finite() && report.cdcl.reignition_ms >= 0.0);
    // Every oracle run stays within its flip budget.
    let budget = parameters.oracle.flips_per_variable * variable_ids(formula).len();
    assert!(report.cdcl.reignition_flips <= report.cdcl.reignitions * budget);
    if parameters.oracle.flips_per_variable == 0 || parameters.reignition_interval == 0 {
        assert_eq!(report.cdcl.reignitions, 0);
        assert_eq!(report.cdcl.reignition_ms, 0.0);
    }
}

#[test]
fn empty_formula_is_sat_without_search() {
    let formula = vec![];
    for parameters in [config(0, 500), config(4, 500)] {
        let report = solve_hybrid(&formula, parameters);
        check_report(&formula, &report, parameters);
        assert!(report.model.as_ref().unwrap().is_empty());
        assert_eq!(report.cdcl.decisions, 0);
        assert_eq!(report.cdcl.conflicts, 0);
    }
}

#[test]
fn the_oracle_never_decides_unsat_and_the_search_still_proves_it() {
    let formula = vec![vec![1], vec![-1]];
    let parameters = config(64, 1);
    let report = solve_hybrid(&formula, parameters);
    check_report(&formula, &report, parameters);
    assert!(report.model.is_none());
    assert!(report.cdcl.conflicts > 0);
    // A root contradiction is settled before the oracle is ever consulted.
    assert_eq!(report.cdcl.reignitions, 0);
}

#[test]
fn a_disabled_oracle_leaves_the_plain_search_untouched() {
    let formula = vec![
        vec![1, 2, 3],
        vec![-1, 2],
        vec![-2, 3],
        vec![-3, 4, 5],
        vec![-4, -5],
        vec![5, 6],
    ];
    let (plain_model, plain) = solve_cdcl(&formula);
    for mut parameters in [config(0, 500), config(8, 0)] {
        // The bare engine runs no inprocessing, so the comparison must not either.
        parameters.symmetry = false;
        parameters.parity = false;
        let report = solve_hybrid(&formula, parameters);
        check_report(&formula, &report, parameters);
        assert_eq!(report.model, plain_model);
        assert_eq!(report.cdcl.decisions, plain.decisions);
        assert_eq!(report.cdcl.conflicts, plain.conflicts);
        assert_eq!(report.cdcl.propagations, plain.propagations);
    }
}

#[test]
fn an_oracle_model_is_verified_and_ends_the_search() {
    // Disjoint clauses: local search satisfies them long before its budget runs out.
    let formula: Vec<Vec<i32>> = (1..=18).step_by(3).map(|v| vec![v, v + 1, v + 2]).collect();
    let mut parameters = config(16, 500);
    parameters.symmetry = false;
    let report = solve_hybrid(&formula, parameters);
    check_report(&formula, &report, parameters);
    assert!(report.model.is_some());
    assert_eq!(report.cdcl.reignitions, 1);
    assert_eq!(report.cdcl.decisions, 0);
    assert_eq!(report.cdcl.conflicts, 0);
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
        for flips in [0, 7] {
            let parameters = config(flips, 1);
            let report = solve_hybrid(&formula, parameters);
            check_report(&formula, &report, parameters);
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
        for flips in [0, 3] {
            let parameters = config(flips, 1);
            let report = solve_hybrid(&formula, parameters);
            check_report(&formula, &report, parameters);
            if let Some(model) = &report.model {
                let assigned: Vec<u32> = model.iter().map(|literal| literal.unsigned_abs()).collect();
                assert_eq!(assigned, variable_ids(&formula));
            }
        }
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
            let parameters = config([0, 1, 5][case % 3], [1, 3, 500][(case / 3) % 3]);
            let report = solve_hybrid(&formula, parameters);
            check_report(&formula, &report, parameters);
        }
    }
}

#[test]
fn every_assignment_excluded_is_proved_unsat_with_the_oracle_running() {
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
        let parameters = config(5, 1);
        let report = solve_hybrid(&formula, parameters);
        check_report(&formula, &report, parameters);
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
fn small_pigeonhole_is_exact_with_and_without_symmetry_breaking() {
    for (pigeons, holes) in [(2, 2), (3, 2), (3, 3), (4, 3)] {
        for (symmetry, padded) in [(false, false), (true, false), (true, true)] {
            let mut formula = pigeonhole(pigeons, holes);
            if padded {
                // A tautology is not part of the structure: it must change nothing.
                formula.push(vec![1, -1, 1]);
            }
            let mut parameters = config(5, 1);
            parameters.symmetry = symmetry;
            let report = solve_hybrid(&formula, parameters);
            check_report(&formula, &report, parameters);
            assert_eq!(report.model.is_some(), pigeons <= holes);
            assert_eq!(report.symmetry.clauses > 0, symmetry);
        }
    }
}

#[test]
fn hybrid_search_model_and_search_counts_are_deterministic() {
    let mut formula = pigeonhole(5, 4);
    formula.push(vec![1, -1, 1]);
    let parameters = config(3, 2);
    let first = solve_hybrid(&formula, parameters);
    let second = solve_hybrid(&formula, parameters);
    check_report(&formula, &first, parameters);
    check_report(&formula, &second, parameters);
    assert_eq!(first.model, second.model);
    assert_eq!(first.cdcl.decisions, second.cdcl.decisions);
    assert_eq!(first.cdcl.conflicts, second.cdcl.conflicts);
    assert_eq!(first.cdcl.propagations, second.cdcl.propagations);
    assert_eq!(first.cdcl.learned_clauses, second.cdcl.learned_clauses);
    assert_eq!(first.cdcl.backjumps, second.cdcl.backjumps);
    assert_eq!(
        first.cdcl.nonchronological_backjumps,
        second.cdcl.nonchronological_backjumps
    );
    assert_eq!(first.cdcl.restarts, second.cdcl.restarts);
    assert_eq!(first.cdcl.reignitions, second.cdcl.reignitions);
    assert_eq!(first.cdcl.reignition_flips, second.cdcl.reignition_flips);
}

#[test]
fn the_oracle_seed_changes_the_walk_but_never_the_answer() {
    let mut asymmetric = pigeonhole(4, 3);
    asymmetric.push(vec![1, -1, 1]);
    let cases = [
        asymmetric,
        pigeonhole(3, 3),
        vec![vec![1, 2], vec![-1, 3], vec![-2, -3], vec![2, 3, 4], vec![-4, 1]],
        vec![vec![1, 2], vec![-1, 2], vec![1, -2], vec![-1, -2]],
    ];
    for formula in cases {
        let expected = brute_force(&formula);
        for seed in 0..16 {
            let parameters = HybridConfig {
                oracle: OracleConfig { flips_per_variable: 2, seed },
                reignition_interval: 1,
                symmetry: seed % 2 == 0,
                parity: seed % 4 < 2,
            };
            let report = solve_hybrid(&formula, parameters);
            check_report(&formula, &report, parameters);
            assert_eq!(report.model.is_some(), expected);
        }
    }
}
