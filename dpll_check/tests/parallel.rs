#[allow(dead_code)]
#[path = "../src/solver.rs"]
mod solver;

use solver::{solve_cdcl, solve_parallel, solve_parallel_with_threads, Formula};

fn brute_satisfiable(formula: &Formula, variables: usize) -> bool {
    assert!(variables <= 9);
    (0..1_usize << variables).any(|bits| {
        formula.iter().all(|clause| {
            clause.iter().any(|&literal| {
                let variable = literal.unsigned_abs() as usize - 1;
                ((bits >> variable) & 1 != 0) == (literal > 0)
            })
        })
    })
}

fn verify_model(formula: &Formula, model: &[i32]) {
    let mut values = std::collections::BTreeMap::new();
    for &literal in model {
        assert_ne!(literal, 0);
        let variable = literal.unsigned_abs();
        let value = literal > 0;
        if let Some(previous) = values.insert(variable, value) {
            assert_eq!(previous, value, "model assigns a variable both ways");
        }
    }
    assert!(formula.iter().all(|clause| {
        clause.iter().any(|&literal| {
            values.get(&literal.unsigned_abs()) == Some(&(literal > 0))
        })
    }), "returned model does not satisfy original formula: {formula:?}");
}

fn check(formula: &Formula, expected: bool, threads: usize) {
    let (result, _) = solve_parallel_with_threads(formula.clone(), Vec::new(), threads);
    assert_eq!(result.is_some(), expected, "threads={threads}, formula={formula:?}");
    if let Some(model) = result {
        verify_model(formula, &model);
    }
}

fn next_random(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn pigeonhole(pigeons: usize, holes: usize, encode_3cnf: bool) -> Formula {
    let mut variables = pigeons * holes;
    let mut formula = Formula::new();
    for pigeon in 0..pigeons {
        let row: Vec<i32> = (0..holes)
            .map(|hole| (pigeon * holes + hole + 1) as i32)
            .collect();
        if !encode_3cnf || row.len() <= 3 {
            formula.push(row);
            continue;
        }
        variables += 1;
        let mut auxiliary = variables as i32;
        formula.push(vec![row[0], row[1], auxiliary]);
        for &literal in &row[2..row.len() - 2] {
            variables += 1;
            let next = variables as i32;
            formula.push(vec![-auxiliary, literal, next]);
            auxiliary = next;
        }
        formula.push(vec![-auxiliary, row[row.len() - 2], row[row.len() - 1]]);
    }
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
fn empty_conflicting_units_and_output_prefix_contract() {
    for threads in [1, 2, 4] {
        let prefix = vec![-900_001, 900_002];
        assert_eq!(solve_parallel_with_threads(Vec::new(), prefix.clone(), threads).0, Some(prefix.clone()));
        assert!(solve_parallel_with_threads(vec![Vec::new()], prefix.clone(), threads).0.is_none());
        check(&vec![vec![1], vec![-1]], false, threads);

        // The historical assignment argument is an output prefix, not assumptions.
        let formula = vec![vec![1, 2], vec![-1, 2], vec![1, -2]];
        let model = solve_parallel_with_threads(formula.clone(), prefix.clone(), threads)
            .0.expect("formula is SAT");
        assert_eq!(&model[..prefix.len()], prefix.as_slice());
        verify_model(&formula, &model[prefix.len()..]);
    }
    assert_eq!(solve_parallel(Vec::new(), vec![99]), Some(vec![99]));
    let formula = vec![vec![1, -2], vec![2, 3], vec![-3, -1]];
    verify_model(&formula, &solve_parallel(formula.clone(), Vec::new()).expect("SAT"));
}

#[test]
fn all_two_variable_formulas_agree_with_brute_force() {
    let clauses = [
        vec![1], vec![-1], vec![2], vec![-2],
        vec![1, 2], vec![1, -2], vec![-1, 2], vec![-1, -2],
    ];
    for subset in 0..1_usize << clauses.len() {
        let formula: Formula = clauses.iter().enumerate()
            .filter(|(index, _)| subset & (1 << index) != 0)
            .map(|(_, clause)| clause.clone())
            .collect();
        let expected = brute_satisfiable(&formula, 2);
        for threads in [1, 2, 4] {
            check(&formula, expected, threads);
        }
    }
}

#[test]
fn every_cube_can_contain_the_only_satisfying_assignment() {
    let n = 4;
    let blocked_assignment = |bits: usize| {
        (0..n).map(|variable| {
            let literal = (variable + 1) as i32;
            if bits & (1 << variable) != 0 { -literal } else { literal }
        }).collect::<Vec<i32>>()
    };

    // No units: each full-width clause excludes exactly one assignment.
    // Testing every survivor catches lost cubes and incorrect assumption handling.
    for survivor in 0..1_usize << n {
        let formula: Formula = (0..1_usize << n)
            .filter(|&bits| bits != survivor)
            .map(blocked_assignment)
            .collect();
        for threads in [1, 2, 4] {
            check(&formula, true, threads);
        }
    }
    let unsat: Formula = (0..1_usize << n).map(blocked_assignment).collect();
    for threads in [1, 2, 4] {
        let (result, stats) = solve_parallel_with_threads(unsat.clone(), Vec::new(), threads);
        assert!(result.is_none());
        assert!(stats.cubes_generated > 0);
        assert_eq!(stats.cubes_started, stats.cubes_generated);
        assert_eq!(stats.cubes_unsat, stats.cubes_generated);
        assert_eq!(stats.cubes_cancelled, 0);
        assert!((1..=threads).contains(&stats.workers));
    }
}

#[test]
fn deterministic_random_cnfs_match_serial_and_brute_force() {
    let mut random = 0xa7_14_3d_62_88_90_ef_u64;
    for case in 0..384 {
        let pure_3cnf = case % 2 == 0;
        let variables = if pure_3cnf {
            3 + next_random(&mut random) as usize % 7
        } else {
            1 + next_random(&mut random) as usize % 9
        };
        let count = if pure_3cnf {
            variables * (3 + next_random(&mut random) as usize % 4)
        } else {
            next_random(&mut random) as usize % (variables * 7 + 1)
        };
        let mut formula = Vec::with_capacity(count);
        for _ in 0..count {
            let length = if pure_3cnf { 3 } else { 1 + next_random(&mut random) as usize % 3 };
            let mut clause = Vec::with_capacity(length);
            while clause.len() < length {
                let variable = (1 + next_random(&mut random) as usize % variables) as i32;
                if pure_3cnf && clause.iter().any(|literal: &i32| literal.unsigned_abs() == variable as u32) {
                    continue;
                }
                clause.push(if next_random(&mut random) & 1 == 0 { variable } else { -variable });
            }
            formula.push(clause);
        }
        let expected = brute_satisfiable(&formula, variables);
        assert_eq!(solve_cdcl(formula.clone(), Vec::new()).0.is_some(), expected);
        check(&formula, expected, [1, 2, 4][case % 3]);
    }
}

#[test]
fn sparse_ids_duplicates_and_tautologies_preserve_models() {
    for threads in [1, 2, 4] {
        check(&vec![
            vec![i32::MAX, -i32::MAX],
            vec![7, 7, -31],
            vec![-7, -7, 31],
            vec![31, i32::MAX],
        ], true, threads);
        check(&vec![
            vec![i32::MAX, i32::MAX],
            vec![-i32::MAX, -i32::MAX],
            vec![7, -7, 31],
        ], false, threads);
    }
}

#[test]
fn certified_large_php_and_asymmetric_deleted_edge_remain_exact() {
    let large = pigeonhole(15, 14, true);
    for threads in [1, 2, 4] {
        let (result, stats) = solve_parallel_with_threads(large.clone(), Vec::new(), threads);
        assert!(result.is_none());
        assert_eq!(stats.total.symmetry_rows, 15);
        assert_eq!(stats.total.symmetry_columns, 14);
        assert!(stats.total.symmetry_clauses > 0);
        assert_eq!(stats.total.decisions, 0);
        assert_eq!(stats.total.conflicts, 1);
        assert_eq!(stats.workers, 0, "root proof must precede cube generation");
        assert_eq!(stats.cubes_started, 0);
    }

    let square = pigeonhole(4, 4, true);
    for threads in [1, 2, 4] {
        check(&square, true, threads);
    }

    let mut asymmetric = pigeonhole(5, 4, true);
    let edge = asymmetric.iter().position(|clause| clause.as_slice() == [-1, -5])
        .expect("column exclusion exists");
    asymmetric.remove(edge);
    for threads in [1, 2, 4] {
        let (result, stats) = solve_parallel_with_threads(asymmetric.clone(), Vec::new(), threads);
        verify_model(&asymmetric, &result.expect("a missing PHP exclusion makes it SAT"));
        assert_eq!(stats.total.symmetry_clauses, 0);
    }
}

#[test]
fn sat_winner_is_valid_while_other_cubes_may_cancel() {
    let mut formula = pigeonhole(5, 4, false);
    let guard = 21;
    for clause in &mut formula {
        clause.push(guard);
    }
    formula.push(vec![-guard, 22]);

    // guard=false contains an UNSAT PHP; guard=true has a short SAT model.
    // Repetition checks result handoff without asserting schedule-dependent counts.
    for _ in 0..16 {
        let model = solve_parallel_with_threads(formula.clone(), Vec::new(), 4)
            .0.expect("a SAT worker must win over cancelled workers");
        verify_model(&formula, &model);
        assert!(model.contains(&guard));
    }
}
