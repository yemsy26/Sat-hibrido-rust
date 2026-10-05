#![forbid(unsafe_code)]
#![deny(warnings)]
#[allow(dead_code)]
#[path = "../src/solver.rs"]
mod solver;
use solver::solve_dpll;

fn baseline(formula: Vec<Vec<i32>>, mut assignment: Vec<i32>) -> Option<Vec<i32>> {
    if formula.is_empty() {
        return Some(assignment);
    }
    if formula.iter().any(Vec::is_empty) {
        return None;
    }
    if let Some(unit) = formula.iter().find(|c| c.len() == 1) {
        let literal = unit[0];
        assignment.push(literal);
        return baseline(simplify(&formula, literal), assignment);
    }
    let literal = formula[0][0];
    let mut yes = assignment.clone();
    yes.push(literal);
    if let Some(model) = baseline(simplify(&formula, literal), yes) {
        return Some(model);
    }
    assignment.push(-literal);
    baseline(simplify(&formula, -literal), assignment)
}

fn simplify(formula: &[Vec<i32>], literal: i32) -> Vec<Vec<i32>> {
    let mut result = Vec::with_capacity(formula.len());
    for clause in formula {
        if !clause.contains(&literal) {
            result.push(clause.iter().copied().filter(|&x| x != -literal).collect());
        }
    }
    result
}

fn chain(n: i32, reverse: bool) -> Vec<Vec<i32>> {
    let mut result: Vec<Vec<i32>> = (1..n).map(|v| vec![-v, v + 1]).collect();
    if reverse {
        result.reverse();
    }
    result.push(vec![1]);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn brute_force(formula: &[Vec<i32>], n: u32) -> bool {
        (0..1_u32 << n).any(|bits| formula.iter().all(|clause| {
            clause.iter().any(|&literal| {
                let value = bits & (1 << (literal.unsigned_abs() - 1)) != 0;
                value == (literal > 0)
            })
        }))
    }

    fn check_model(formula: &[Vec<i32>], model: &[i32]) {
        let mut values = BTreeMap::new();
        for &literal in model {
            if let Some(previous) = values.insert(literal.unsigned_abs(), literal > 0) {
                assert_eq!(previous, literal > 0);
            }
        }
        assert!(formula.iter().all(|clause| clause.iter().any(|literal| {
            values.get(&literal.unsigned_abs()) == Some(&(literal > &0))
        })));
    }

    fn check(formula: Vec<Vec<i32>>, n: u32) {
        let expected = brute_force(&formula, n);
        assert_eq!(baseline(formula.clone(), Vec::new()).is_some(), expected);
        let result = solve_dpll(formula.clone(), Vec::new());
        assert_eq!(result.is_some(), expected, "{formula:?}");
        if let Some(model) = result {
            check_model(&formula, &model);
        }
    }

    #[test]
    fn exhaustive_two_variable_formulas() {
        let literals = [-2, -1, 1, 2];
        let mut universe: Vec<Vec<i32>> = literals.into_iter().map(|lit| vec![lit]).collect();
        for i in 0..literals.len() {
            for j in i..literals.len() {
                universe.push(vec![literals[i], literals[j]]);
            }
        }
        for mask in 0..1_usize << universe.len() {
            let formula = universe.iter().enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, clause)| clause.clone()).collect();
            check(formula, 2);
        }
    }

    #[test]
    fn exhaustive_three_variable_sign_formulas() {
        let universe: Vec<Vec<i32>> = (0..8).map(|signs| {
            (1..=3).map(|v| if signs & (1 << (v - 1)) == 0 { v } else { -v }).collect()
        }).collect();
        for mask in 0..256 {
            let formula = universe.iter().enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, c)| c.clone()).collect();
            check(formula, 3);
        }
    }

    #[test]
    fn generated_formulas_match_exhaustive_search() {
        let mut seed = 0x4d595df4d0f33173_u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 32) as u32
        };
        for _ in 0..12_000 {
            let n = 1 + next() % 8;
            let m = next() % 35;
            let formula = (0..m).map(|_| {
                let len = 1 + next() % 5;
                (0..len).map(|_| {
                    let v = (1 + next() % n) as i32;
                    if next() & 1 == 0 { v } else { -v }
                }).collect()
            }).collect();
            check(formula, n);
        }
    }

    #[test]
    fn edge_cases_and_prefix_contract() {
        assert_eq!(solve_dpll(vec![], vec![7, -7]), Some(vec![7, -7]));
        assert_eq!(solve_dpll(vec![vec![]], vec![]), None);
        check(vec![vec![1, 2], vec![-1, 3], vec![-1, -3]], 3);
        check(vec![vec![1, 1], vec![-1, -1]], 1);
        check(vec![vec![1, -1]], 1);
        let prefix = solve_dpll(vec![vec![1]], vec![-1]).expect("SAT");
        assert_eq!(prefix, vec![-1, 1]);
        let sparse = vec![vec![i32::MAX, -1000000], vec![-i32::MAX], vec![-1000000]];
        let model = solve_dpll(sparse.clone(), vec![]).expect("SAT");
        check_model(&sparse, &model);
        assert_eq!(solve_dpll(vec![vec![i32::MAX], vec![-i32::MAX]], vec![]), None);
    }

    #[test]
    fn vsids_prioritizes_the_highest_occurrence_variable() {
        let formula = vec![vec![1, 2, 3], vec![2, 4, 5], vec![2, 6, 7]];
        let model = solve_dpll(formula.clone(), vec![]).expect("SAT");
        assert_eq!(model[0], 2);
        check_model(&formula, &model);
    }

    #[test]
    fn watched_duplicates_tautologies_and_moves() {
        check(vec![vec![1, 1, 2], vec![-1]], 2);
        check(vec![vec![1, 1, 2], vec![-1], vec![-2]], 2);
        check(vec![vec![1, 1, -1], vec![-1]], 1);
        check(vec![vec![1, 2, 3], vec![-1], vec![-2]], 3);
        check(vec![vec![1, 2, 3], vec![-1], vec![-2], vec![-3]], 3);
        check(vec![vec![-1, -2, 3], vec![-1, -3, 4], vec![-1, -4, -3]], 4);
    }

    #[test]
    fn cdcl_learns_a_root_unit() {
        let formula = vec![vec![-1, 2], vec![-1, -2]];
        let (result, stats) = solver::solve_cdcl(formula.clone(), vec![]);
        let model = result.expect("SAT after learning -1");
        check_model(&formula, &model);
        assert!(model.contains(&-1));
        assert!(stats.conflicts > 0, "{stats:?}");
        assert!(stats.learned_clauses > 0, "{stats:?}");
        assert!(stats.backjumps > 0, "{stats:?}");
    }

    #[test]
    fn cdcl_resolves_multiple_reasons_and_skips_a_decision_level() {
        let mut formula = vec![
            vec![-1, -3, 4],
            vec![-3, 5],
            vec![-4, -5, 6],
            vec![-6, 7],
            vec![-4, -7],
        ];
        formula.extend((8..=18).map(|v| vec![1, v]));
        formula.extend((19..=28).map(|v| vec![2, v]));
        formula.extend((29..=34).map(|v| vec![3, v]));
        let (result, stats) = solver::solve_cdcl(formula.clone(), vec![]);
        check_model(&formula, &result.expect("SAT after learning -3 or -1"));
        assert!(stats.conflicts > 0, "{stats:?}");
        assert!(stats.learned_clauses > 0, "{stats:?}");
        assert!(stats.nonchronological_backjumps > 0, "{stats:?}");
    }

    #[test]
    fn cdcl_random_three_cnf_matches_exhaustive_search() {
        let mut seed = 0x38c5492a271bc697_u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 32) as u32
        };
        for sample in 0..1_000 {
            let n = 9 + next() % 4;
            let m = 3 * n + next() % (3 * n + 1);
            let formula: Vec<Vec<i32>> = (0..m).map(|_| {
                let mut clause = Vec::with_capacity(3);
                while clause.len() < 3 {
                    let variable = (1 + next() % n) as i32;
                    if clause.iter().any(|literal: &i32| literal.abs() == variable) {
                        continue;
                    }
                    clause.push(if next() & 1 == 0 { variable } else { -variable });
                }
                clause
            }).collect();
            let expected = brute_force(&formula, n);
            let (result, stats) = solver::solve_cdcl(formula.clone(), vec![]);
            assert_eq!(result.is_some(), expected,
                "sample={sample}, stats={stats:?}, formula={formula:?}");
            if let Some(model) = result {
                check_model(&formula, &model);
            }
        }
    }

    #[test]
    fn cdcl_pigeonhole_unsat_reclaims_learned_clauses() {
        let holes = 6_i32;
        let pigeons = holes + 1;
        let mut formula = Vec::new();
        for pigeon in 0..pigeons {
            formula.push((0..holes).map(|hole| pigeon * holes + hole + 1).collect());
        }
        for hole in 0..holes {
            for first in 0..pigeons {
                for second in first + 1..pigeons {
                    formula.push(vec![
                        -(first * holes + hole + 1),
                        -(second * holes + hole + 1),
                    ]);
                }
            }
        }
        // Fuerza el camino CDCL general para seguir comprobando su limpieza.
        formula.push(vec![1, -1]);
        let (result, stats) = solver::solve_cdcl(formula, vec![]);
        assert!(result.is_none(), "Seven pigeons cannot occupy six holes");
        assert!(stats.database_reductions > 0, "{stats:?}");
        assert!(stats.deleted_clauses > 0, "{stats:?}");
    }

    #[test]
    fn long_unit_chain_avoids_recursive_stack() {
        let formula = chain(30_000, true);
        let model = solve_dpll(formula.clone(), vec![]).expect("SAT");
        assert_eq!(model.len(), 30_000);
        check_model(&formula, &model);
    }
}
