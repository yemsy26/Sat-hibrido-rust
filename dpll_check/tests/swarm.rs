#![forbid(unsafe_code)]
#![deny(warnings)]

#[allow(dead_code)]
#[path = "../src/hybrid.rs"]
mod hybrid;

use hybrid::cdcl::{solve_swarm, ReignitionConfig, SwarmConfig, SwarmStats};
use hybrid::gradient::GradientConfig;
use std::collections::{BTreeMap, BTreeSet};

fn config(workers: usize, sharing: bool) -> SwarmConfig {
    SwarmConfig {
        workers,
        cubes_per_worker: 4,
        share_capacity: 8,
        share_interval: 1,
        share_enabled: sharing,
        reignition: ReignitionConfig {
            interval: 1,
            oracle: GradientConfig {
                max_epochs: 1,
                restarts: 1,
                check_every: 1,
                ..GradientConfig::default()
            },
        },
        activity_boost: 4.0,
        ..SwarmConfig::default()
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
        "Returned model does not satisfy original clauses"
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

fn assert_shared_clauses_are_global_consequences(formula: &[Vec<i32>], stats: &SwarmStats) {
    let variables: Vec<u32> = formula
        .iter()
        .flatten()
        .map(|v| v.unsigned_abs())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    assert!(variables.len() <= 20);
    assert!(
        brute_force(formula),
        "Consequence checks must have a nonvacuous SAT premise"
    );
    for clause in &stats.shared_clauses {
        assert!(!clause.is_empty() && clause.len() <= 3);
        assert!(clause.iter().all(|literal| literal.unsigned_abs() != 0
            && variables.binary_search(&literal.unsigned_abs()).is_ok()));
        for bits in 0..1_u64 << variables.len() {
            let literal_true = |literal: i32| {
                let index = variables.binary_search(&literal.unsigned_abs()).unwrap();
                (bits & (1_u64 << index) != 0) == (literal > 0)
            };
            if formula
                .iter()
                .all(|original| original.iter().copied().any(literal_true))
            {
                assert!(clause.iter().copied().any(literal_true),
                    "Shared clause {clause:?} is only valid in one cube, not globally implied by {formula:?}");
            }
        }
    }
}

fn check_counters(stats: &SwarmStats, expected_sat: bool, requested_workers: usize) {
    assert!(stats.workers <= requested_workers);
    assert!(stats.cubes_started <= stats.cubes_generated);
    assert!(stats.cubes_unsat + stats.cubes_cancelled <= stats.cubes_started);
    assert!(stats.total.reignition_ms.is_finite() && stats.total.reignition_ms >= 0.0);
    assert!(stats.total.reignition_epochs <= stats.total.reignitions);
    if stats.workers == 0 {
        assert_eq!(stats.cubes_started, 0);
        assert_eq!(stats.cubes_cancelled, 0);
    } else if expected_sat {
        assert!(
            stats.cubes_started > stats.cubes_unsat + stats.cubes_cancelled,
            "At least one started cube must return SAT, not UNSAT or cancellation"
        );
    } else {
        assert_eq!(
            stats.cubes_cancelled, 0,
            "UNSAT requires full cube coverage"
        );
        assert_eq!(stats.cubes_started, stats.cubes_generated);
        assert_eq!(stats.cubes_unsat, stats.cubes_generated);
    }
}

fn check(formula: &[Vec<i32>], workers: usize, sharing: bool) -> SwarmStats {
    let expected = brute_force(formula);
    let (model, stats) = solve_swarm(formula, None, config(workers, sharing));
    assert_eq!(
        model.is_some(),
        expected,
        "workers={workers}, sharing={sharing}, formula={formula:?}"
    );
    if let Some(model) = model {
        check_model(formula, &model);
    }
    check_counters(&stats, expected, workers);
    if !sharing {
        assert_eq!(stats.total.exported_clauses, 0);
        assert_eq!(stats.total.imported_clauses, 0);
        assert_eq!(stats.total.dropped_exports, 0);
        assert!(stats.shared_clauses.is_empty());
    }
    stats
}

#[test]
fn root_sat_and_root_unsat_do_not_spawn_workers() {
    for (formula, expected_sat) in [
        (vec![], true),
        (vec![vec![1], vec![-1, 2]], true),
        (vec![vec![]], false),
        (vec![vec![1], vec![-1]], false),
    ] {
        let (model, stats) = solve_swarm(&formula, None, config(4, true));
        assert_eq!(model.is_some(), expected_sat);
        if let Some(model) = model {
            check_model(&formula, &model);
        }
        assert_eq!(stats.workers, 0);
        assert_eq!(stats.cubes_started, 0);
        assert_eq!(stats.cubes_cancelled, 0);
    }
}

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

#[test]
fn swarm_matches_small_exhaustive_truth_tables_with_sharing_on_and_off() {
    let mut seed = 0x825e_a73d_4061_9bc2;
    for n in 1..=6 {
        for case in 0..12 {
            let clauses = (next(&mut seed) % 24) as usize;
            let formula: Vec<Vec<i32>> = (0..clauses)
                .map(|_| {
                    let width = if case % 11 == 0 {
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
            for workers in [1, 2, 4] {
                for sharing in [false, true] {
                    check(&formula, workers, sharing);
                }
            }
        }
    }
}

fn pigeonhole(pigeons: usize, holes: usize, disable_recognition: bool) -> Vec<Vec<i32>> {
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
    if disable_recognition {
        formula.push(vec![1, -1, 1]);
    }
    formula
}

#[test]
fn unsat_requires_every_generated_cube_to_be_processed() {
    let formula = pigeonhole(4, 3, true);
    for workers in [1, 2, 4] {
        for sharing in [false, true] {
            let stats = check(&formula, workers, sharing);
            assert!(stats.workers > 0);
            assert!(stats.cubes_generated > 1);
            assert_eq!(stats.total.symmetry_clauses, 0);
        }
    }
}

#[test]
fn certified_symmetry_is_applied_before_cubing() {
    let formula = pigeonhole(4, 3, false);
    let (model, stats) = solve_swarm(&formula, None, config(4, true));
    assert!(model.is_none());
    assert_eq!(stats.workers, 0);
    assert_eq!(stats.cubes_started, 0);
    assert!(stats.total.symmetry_clauses > 0);
}

// Equivalence chain plus one non-unit disjunction has exactly one model.
fn unique_model_without_units(n: usize, bits: usize) -> Vec<Vec<i32>> {
    let literals: Vec<i32> = (0..n)
        .map(|index| {
            let variable = (index + 1) as i32;
            if bits & (1 << index) != 0 {
                variable
            } else {
                -variable
            }
        })
        .collect();
    let mut formula = vec![literals.clone()];
    for pair in literals.windows(2) {
        formula.push(vec![-pair[0], pair[1]]);
        formula.push(vec![pair[0], -pair[1]]);
    }
    formula
}

#[test]
fn unsatisfiable_cubes_cannot_poison_the_unique_satisfiable_cube() {
    for bits in [0b000000, 0b101010, 0b111111] {
        let formula = unique_model_without_units(6, bits);
        let guidance: Vec<(u32, f64)> = (0..6)
            .map(|index| {
                // Deliberately oppose every bit of the unique satisfying model.
                (
                    (index + 1) as u32,
                    if bits & (1 << index) != 0 { 0.01 } else { 0.99 },
                )
            })
            .collect();
        for sharing in [false, true] {
            let mut parameters = config(4, sharing);
            parameters.cubes_per_worker = 8;
            parameters.share_capacity = 1;
            let (model, stats) = solve_swarm(&formula, Some(&guidance), parameters);
            let model = model.expect(
                "Sharing a cube-specific contradiction must not make the base SAT formula UNSAT",
            );
            check_model(&formula, &model);
            check_counters(&stats, true, 4);
            if sharing {
                assert!(!stats.shared_clauses.is_empty(),
                    "The first cube opposes the unique model and must publish a global short clause");
            }
            assert_shared_clauses_are_global_consequences(&formula, &stats);
        }
    }
}

#[test]
fn empty_and_single_slot_sharing_pools_finish_without_deadlock() {
    let formulas = [
        pigeonhole(4, 3, true),
        unique_model_without_units(6, 0b101101),
    ];
    for formula in formulas {
        let expected = brute_force(&formula);
        for capacity in [0, 1] {
            let mut parameters = config(4, true);
            parameters.share_capacity = capacity;
            let (model, stats) = solve_swarm(&formula, None, parameters);
            assert_eq!(model.is_some(), expected);
            if let Some(model) = model {
                check_model(&formula, &model);
            }
            check_counters(&stats, expected, 4);
            assert!(stats.shared_clauses.len() <= capacity);
            if capacity == 0 {
                assert_eq!(stats.total.exported_clauses, 0);
                assert_eq!(stats.total.imported_clauses, 0);
            }
        }
    }
}

#[test]
fn sparse_ids_tautologies_repeated_literals_and_bad_priors_remain_exact() {
    let high = i32::MAX;
    let cases = [
        vec![
            vec![high, 19],
            vec![-high, 19],
            vec![high, -19],
            vec![-high, -19],
        ],
        vec![vec![high, high], vec![-19, high], vec![19, -19, 19]],
        vec![vec![high, 19], vec![-high, -19], vec![7, -7]],
    ];
    for formula in cases {
        let guidance = [(7, 0.01), (19, 0.99), (high as u32, 0.01)];
        for sharing in [false, true] {
            let (model, stats) = solve_swarm(&formula, Some(&guidance), config(4, sharing));
            let expected = brute_force(&formula);
            assert_eq!(model.is_some(), expected);
            if let Some(model) = model {
                check_model(&formula, &model);
            }
            check_counters(&stats, expected, 4);
        }
    }
}

#[test]
fn published_clauses_are_entailed_by_all_models_of_a_satisfiable_formula() {
    let formula = vec![
        vec![1, 2],
        vec![-1, -2],
        vec![3, 4],
        vec![-3, -4],
        vec![-1, 3],
        vec![2, 3, 5],
        vec![5, -5],
    ];
    for workers in [2, 4] {
        let mut parameters = config(workers, true);
        parameters.share_capacity = 64;
        parameters.cubes_per_worker = 8;
        let (model, stats) = solve_swarm(&formula, None, parameters);
        check_model(&formula, &model.unwrap());
        check_counters(&stats, true, workers);
        assert_shared_clauses_are_global_consequences(&formula, &stats);
    }
}

#[test]
fn conflict_search_publishes_short_clauses_and_reignites() {
    let formula = pigeonhole(4, 3, true);
    let mut parameters = config(2, true);
    parameters.cubes_per_worker = 1;
    parameters.share_capacity = 64;
    let (model, stats) = solve_swarm(&formula, None, parameters);
    assert!(model.is_none());
    check_counters(&stats, false, 2);
    assert!(stats.total.exported_clauses > 0);
    assert!(!stats.shared_clauses.is_empty());
    assert!(stats
        .shared_clauses
        .iter()
        .all(|clause| !clause.is_empty() && clause.len() <= 3));
    assert!(stats.total.reignitions > 0);
}

#[test]
fn shared_conflict_pool_preserves_unsat_with_persistent_workers() {
    let formula = pigeonhole(6, 5, true);
    let mut parameters = config(2, true);
    parameters.cubes_per_worker = 1;
    parameters.share_capacity = 128;
    let (model, stats) = solve_swarm(&formula, None, parameters);
    assert!(model.is_none());
    check_counters(&stats, false, 2);
    assert!(stats.total.exported_clauses > 0);
}

#[test]
fn persistent_single_worker_keeps_learning_sound_across_multiple_cubes() {
    let satisfiable = unique_model_without_units(6, 0b101010);
    let guidance: Vec<(u32, f64)> = (0..6).map(|index| {
        ((index + 1) as u32, if 0b101010 & (1 << index) != 0 { 0.01 } else { 0.99 })
    }).collect();
    let mut parameters = config(1, true);
    parameters.cubes_per_worker = 8;
    let (model, stats) = solve_swarm(&satisfiable, Some(&guidance), parameters);
    check_model(&satisfiable, &model.unwrap());
    assert_eq!(stats.workers, 1);
    assert!(stats.cubes_generated >= 4 && stats.cubes_started >= 4);
    check_counters(&stats, true, 1);
    assert_shared_clauses_are_global_consequences(&satisfiable, &stats);

    let unsatisfiable = pigeonhole(4, 3, true);
    let (model, stats) = solve_swarm(&unsatisfiable, None, parameters);
    assert!(model.is_none());
    assert_eq!(stats.workers, 1);
    assert!(stats.cubes_generated >= 4);
    assert_eq!(stats.cubes_started, stats.cubes_generated);
    assert_eq!(stats.cubes_unsat, stats.cubes_generated);
    check_counters(&stats, false, 1);
}
