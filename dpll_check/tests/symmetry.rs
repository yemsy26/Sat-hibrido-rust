#[allow(dead_code)]
#[path = "../src/solver.rs"]
mod solver;

use solver::{solve_cdcl, solve_dpll, Formula, SolveStats};

fn pigeonhole(pigeons: usize, holes: usize, encode_3cnf: bool) -> Formula {
    let mut variables = pigeons * holes;
    let mut formula = Formula::new();
    for pigeon in 0..pigeons {
        let positive: Vec<i32> = (0..holes)
            .map(|hole| (pigeon * holes + hole + 1) as i32)
            .collect();
        if !encode_3cnf || positive.len() <= 3 {
            formula.push(positive);
        } else {
            variables += 1;
            let mut auxiliary = variables as i32;
            formula.push(vec![positive[0], positive[1], auxiliary]);
            for &literal in &positive[2..positive.len() - 2] {
                variables += 1;
                let next = variables as i32;
                formula.push(vec![-auxiliary, literal, next]);
                auxiliary = next;
            }
            formula.push(vec![
                -auxiliary,
                positive[positive.len() - 2],
                positive[positive.len() - 1],
            ]);
        }
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

fn check(formula: &Formula, expected_sat: bool) -> SolveStats {
    let (result, stats) = solve_cdcl(formula.clone(), Vec::new());
    assert_eq!(result.is_some(), expected_sat, "formula: {formula:?}");
    if let Some(model) = result {
        let variables = formula
            .iter()
            .flatten()
            .map(|literal| literal.unsigned_abs() as usize)
            .max()
            .unwrap_or(0);
        let mut values = vec![0_i8; variables + 1];
        for literal in model {
            let variable = literal.unsigned_abs() as usize;
            assert!(variable != 0 && variable <= variables);
            let value = if literal > 0 { 1 } else { -1 };
            assert!(values[variable] == 0 || values[variable] == value);
            values[variable] = value;
        }
        assert!(formula.iter().all(|clause| clause.iter().any(|&literal| {
            values[literal.unsigned_abs() as usize] == if literal > 0 { 1 } else { -1 }
        })), "returned model does not satisfy the original formula");
    }
    stats
}

fn next_random(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn shuffled(formula: &Formula, seed: u64) -> Formula {
    let variables = formula
        .iter()
        .flatten()
        .map(|literal| literal.unsigned_abs() as usize)
        .max()
        .unwrap_or(0);
    let mut permutation: Vec<usize> = (1..=variables).collect();
    let mut random = seed;
    for index in (1..permutation.len()).rev() {
        let other = next_random(&mut random) as usize % (index + 1);
        permutation.swap(index, other);
    }
    let mut result: Formula = formula
        .iter()
        .map(|clause| {
            clause
                .iter()
                .map(|&literal| {
                    let variable = permutation[literal.unsigned_abs() as usize - 1] as i32;
                    if literal > 0 { variable } else { -variable }
                })
                .collect()
        })
        .collect();
    for clause in &mut result {
        for index in (1..clause.len()).rev() {
            let other = next_random(&mut random) as usize % (index + 1);
            clause.swap(index, other);
        }
    }
    for index in (1..result.len()).rev() {
        let other = next_random(&mut random) as usize % (index + 1);
        result.swap(index, other);
    }
    result
}

#[test]
fn canonical_pigeonholes_sat_and_unsat() {
    for (pigeons, holes, expected) in [
        (0, 0, true),
        (1, 0, false),
        (1, 1, true),
        (2, 1, false),
        (2, 2, true),
        (3, 3, true),
        (4, 3, false),
        (4, 5, true),
        (5, 4, false),
    ] {
        let stats = check(&pigeonhole(pigeons, holes, false), expected);
        if pigeons >= 2 && holes >= 2 {
            assert!(stats.symmetry_clauses > 0);
            assert_eq!(stats.symmetry_rows, pigeons);
            assert_eq!(stats.symmetry_columns, holes);
        } else {
            assert_eq!(stats.symmetry_clauses, 0);
        }
    }
    assert!(solve_dpll(Vec::new(), Vec::new()).is_some());
    assert!(solve_dpll(vec![vec![1], vec![-1]], Vec::new()).is_none());
}

#[test]
fn private_chain_auxiliaries_preserve_original_models() {
    for (pigeons, holes, expected) in [
        (3, 4, true),
        (4, 4, true),
        (5, 4, false),
        (5, 5, true),
        (6, 5, false),
        (6, 6, true),
    ] {
        let formula = pigeonhole(pigeons, holes, true);
        assert!(formula.iter().all(|clause| clause.len() <= 3));
        let stats = check(&formula, expected);
        assert!(stats.symmetry_clauses > 0);
        assert_eq!(stats.symmetry_rows, pigeons);
        assert_eq!(stats.symmetry_columns, holes);
    }
}

#[test]
fn variable_clause_and_literal_permutations_preserve_results() {
    for (pigeons, holes, encoded, expected) in [
        (3, 3, false, true),
        (4, 3, false, false),
        (4, 4, true, true),
        (5, 4, true, false),
    ] {
        let original = pigeonhole(pigeons, holes, encoded);
        for seed in 1..=12 {
            let stats = check(&shuffled(&original, seed), expected);
            assert!(stats.symmetry_clauses > 0);
            assert_eq!(stats.symmetry_rows, pigeons);
            assert_eq!(stats.symmetry_columns, holes);
        }
    }
}

#[test]
fn deleting_one_exclusion_makes_overfull_pigeonhole_sat() {
    for (pigeons, holes, encoded) in [(4, 3, false), (5, 4, true)] {
        let mut formula = pigeonhole(pigeons, holes, encoded);
        let exclusion = vec![-1, -((holes + 1) as i32)];
        let position = formula.iter().position(|clause| *clause == exclusion).unwrap();
        formula.remove(position);
        assert_eq!(check(&formula, true).symmetry_clauses, 0);
        for seed in 1..=8 {
            assert_eq!(check(&shuffled(&formula, seed), true).symmetry_clauses, 0);
        }
    }
}

#[test]
fn extra_units_break_symmetry_without_losing_sat_models() {
    for (pigeons, holes, encoded) in [(2, 2, false), (4, 4, true)] {
        for reversed in [false, true] {
            let mut formula = pigeonhole(pigeons, holes, encoded);
            for pigeon in 0..pigeons {
                let selected = if reversed { holes - 1 - pigeon } else { pigeon };
                for hole in 0..holes {
                    let literal = (pigeon * holes + hole + 1) as i32;
                    formula.push(vec![if hole == selected { literal } else { -literal }]);
                }
            }
            assert_eq!(check(&formula, true).symmetry_clauses, 0);
            assert_eq!(check(&shuffled(&formula, 17), true).symmetry_clauses, 0);
        }
    }
}

#[test]
fn externally_constrained_auxiliary_is_not_private() {
    // In the width-four encoding, an all-false first pair forces its auxiliary true.
    let mut satisfiable = pigeonhole(1, 4, true);
    satisfiable.extend([vec![-1], vec![-2], vec![3], vec![5]]);
    assert_eq!(check(&satisfiable, true).symmetry_clauses, 0);
    let mut unsatisfiable = pigeonhole(1, 4, true);
    unsatisfiable.extend([vec![-1], vec![-2], vec![3], vec![-5]]);
    assert_eq!(check(&unsatisfiable, false).symmetry_clauses, 0);
    // The same restriction must disable recognition for a nontrivial square PHP.
    let mut square = pigeonhole(4, 4, true);
    square.extend([vec![-1], vec![-2], vec![3], vec![17]]);
    assert_eq!(check(&square, true).symmetry_clauses, 0);
    let mut square_conflict = pigeonhole(4, 4, true);
    square_conflict.extend([vec![-1], vec![-2], vec![3], vec![-17]]);
    assert_eq!(check(&square_conflict, false).symmetry_clauses, 0);
}

fn brute_satisfiable(formula: &Formula, variables: usize) -> bool {
    assert!(variables <= 9);
    (0..1_usize << variables).any(|bits| {
        formula.iter().all(|clause| clause.iter().any(|&literal| {
            let variable = literal.unsigned_abs() as usize - 1;
            ((bits >> variable) & 1 == 1) == (literal > 0)
        }))
    })
}

#[test]
fn deterministic_near_pigeonholes_agree_with_brute_force() {
    let mut random = 0x61_d1_82_34_a3_c9_27_u64;
    for (pigeons, holes) in [(3, 2), (2, 3), (3, 3)] {
        let variables = pigeons * holes;
        let original = pigeonhole(pigeons, holes, false);
        for _ in 0..96 {
            let mut formula = original.clone();
            let removed = next_random(&mut random) as usize % 4;
            for _ in 0..removed {
                if !formula.is_empty() {
                    let position = next_random(&mut random) as usize % formula.len();
                    formula.remove(position);
                }
            }
            let added = next_random(&mut random) as usize % 4;
            for _ in 0..added {
                let length = 1 + next_random(&mut random) as usize % 3;
                let mut clause = Vec::with_capacity(length);
                for _ in 0..length {
                    let variable = (1 + next_random(&mut random) as usize % variables) as i32;
                    let literal = if next_random(&mut random) & 1 == 0 { variable } else { -variable };
                    clause.push(literal);
                }
                formula.push(clause);
            }
            let expected = brute_satisfiable(&formula, variables);
            check(&formula, expected);
        }
    }
}

#[test]
fn pigeonhole_15_14_is_root_conflict() {
    let formula = pigeonhole(15, 14, true);
    let stats = check(&formula, false);
    assert!(stats.symmetry_clauses > 0);
    assert_eq!(stats.symmetry_rows, 15);
    assert_eq!(stats.symmetry_columns, 14);
    assert_eq!(stats.decisions, 0, "certified pigeonhole preprocessing must close at root");
}
