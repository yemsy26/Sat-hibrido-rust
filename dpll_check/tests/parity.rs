#![forbid(unsafe_code)]
#![deny(warnings)]

#[allow(dead_code)]
#[path = "../src/hybrid.rs"]
mod hybrid;

use hybrid::parity::{reduce_parity, xor_equations};
use hybrid::{solve_hybrid, verify_model, HybridConfig};
use std::collections::BTreeSet;

fn variable_ids(formula: &[Vec<i32>]) -> Vec<u32> {
    formula
        .iter()
        .flatten()
        .map(|literal| literal.unsigned_abs())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn satisfies(formula: &[Vec<i32>], variables: &[u32], bits: u64) -> bool {
    formula.iter().all(|clause| {
        clause.iter().any(|&literal| {
            let index = variables.binary_search(&literal.unsigned_abs()).unwrap();
            (bits & (1_u64 << index) != 0) == (literal > 0)
        })
    })
}

// CNF of x1 xor ... xor xk = value: one clause per sign pattern of the opposite parity.
fn xor_clauses(variables: &[i32], value: bool, formula: &mut Vec<Vec<i32>>) {
    for pattern in 0..1_u32 << variables.len() {
        if (pattern.count_ones() & 1 == 1) != value {
            formula.push(
                variables
                    .iter()
                    .enumerate()
                    .map(|(bit, &variable)| if pattern >> bit & 1 == 1 { -variable } else { variable })
                    .collect(),
            );
        }
    }
}

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

// The injected clauses must not change the models over the original variables: every model
// of the formula extends to the auxiliaries, and nothing else satisfies the extension.
fn assert_models_are_preserved(formula: &[Vec<i32>]) -> usize {
    let reduction = reduce_parity(formula);
    let original = variable_ids(formula);
    let mut extended = formula.to_vec();
    extended.extend(reduction.clauses.iter().cloned());
    let all = variable_ids(&extended);
    assert_eq!(all.len(), original.len() + reduction.auxiliaries);
    assert!(all.len() <= 22, "Too many variables for an exhaustive check");
    assert!(all[..original.len()] == original[..], "Auxiliaries must sit above every original id");
    let extra = all.len() - original.len();
    for bits in 0..1_u64 << original.len() {
        let extends = (0..1_u64 << extra).any(|aux| satisfies(&extended, &all, bits | aux << original.len()));
        assert_eq!(
            extends,
            satisfies(formula, &original, bits),
            "Injected {:?} changed the models of {formula:?}",
            reduction.clauses
        );
    }
    assert_eq!(reduction.refuted, reduction.clauses.iter().any(Vec::is_empty));
    reduction.clauses.len()
}

#[test]
fn the_four_clause_pattern_is_one_equation() {
    // a xor b xor c = 1, exactly as four clauses.
    let formula = vec![vec![1, 2, 3], vec![-1, -2, 3], vec![-1, 2, -3], vec![1, -2, -3]];
    let reduction = reduce_parity(&formula);
    assert_eq!(reduction.equations, 1);
    assert_eq!(reduction.rank, 1);
    assert!(!reduction.refuted);
    // The only reduced row is the equation the formula already states.
    assert!(reduction.clauses.is_empty());

    // Literal order, clause order, repeats and an unrelated clause change nothing.
    let disguised = vec![vec![-3, 1, -2], vec![4, 5], vec![3, -2, -1], vec![2, 1, 3], vec![2, -1, -3], vec![1, 2, 3]];
    assert_eq!(reduce_parity(&disguised).equations, 1);
    // Three of the four clauses are not a parity constraint.
    assert_eq!(reduce_parity(&formula[..3]).equations, 0);
}

#[test]
fn elimination_reveals_units_and_equivalences() {
    // x1 xor x2 = 1 and x1 xor x2 xor x3 = 0 force x3 = 1.
    let mut formula = Vec::new();
    xor_clauses(&[1, 2], true, &mut formula);
    xor_clauses(&[1, 2, 3], false, &mut formula);
    let reduction = reduce_parity(&formula);
    assert_eq!((reduction.equations, reduction.rank), (2, 2));
    assert_eq!(reduction.units, 1);
    assert!(reduction.clauses.contains(&vec![3]));
    assert_models_are_preserved(&formula);

    // x1 xor x2 xor x3 = 1 and x1 xor x2 xor x4 = 0 force x3 xor x4 = 1.
    let mut formula = Vec::new();
    xor_clauses(&[1, 2, 3], true, &mut formula);
    xor_clauses(&[1, 2, 4], false, &mut formula);
    let reduction = reduce_parity(&formula);
    assert_eq!(reduction.equivalences, 1);
    assert!(reduction.clauses.contains(&vec![3, 4]) && reduction.clauses.contains(&vec![-3, -4]));
    assert_models_are_preserved(&formula);
}

#[test]
fn an_inconsistent_system_is_refuted_without_search() {
    // Tseitin on K4: every edge joins two vertices, so the charges cannot sum to one.
    let edges = [[1, 2, 3], [1, 4, 5], [2, 4, 6], [3, 5, 6]];
    let mut formula = Vec::new();
    for (vertex, incident) in edges.iter().enumerate() {
        xor_clauses(incident, vertex == 0, &mut formula);
    }
    let reduction = reduce_parity(&formula);
    assert!(reduction.refuted);
    assert_eq!(reduction.clauses, vec![Vec::<i32>::new()]);
    assert_models_are_preserved(&formula);

    let report = solve_hybrid(&formula, HybridConfig::default());
    assert!(report.model.is_none());
    assert!(report.parity.refuted);
    assert_eq!(report.cdcl.decisions, 0);
    // The symmetry layer is not consulted once 0 = 1 is known.
    assert_eq!(report.symmetry.generators, 0);

    let report = solve_hybrid(&formula, HybridConfig { parity: false, ..HybridConfig::default() });
    assert!(report.model.is_none());
    assert!(!report.parity.refuted);
    assert_eq!(report.parity.equations, 0);
}

#[test]
fn injected_clauses_preserve_every_model_on_random_systems() {
    let mut seed = 0x6a09_e667_f3bc_c908;
    let (mut formulas, mut reduced) = (0, 0);
    for case in 0..400 {
        let variables = 3 + (next(&mut seed) % 8) as i32;
        let mut formula = Vec::new();
        for _ in 0..1 + next(&mut seed) % variables as u64 {
            let width = 1 + (next(&mut seed) % 4) as usize;
            let mut row: Vec<i32> = Vec::new();
            while row.len() < width.min(variables as usize) {
                let variable = 1 + (next(&mut seed) % variables as u64) as i32;
                if !row.contains(&variable) {
                    row.push(variable);
                }
            }
            xor_clauses(&row, next(&mut seed) & 1 == 1, &mut formula);
        }
        // Ordinary clauses around the linear part: they must not be mistaken for parity.
        for _ in 0..case % 4 {
            let clause: Vec<i32> = (0..1 + next(&mut seed) % 3)
                .map(|_| {
                    let variable = 1 + (next(&mut seed) % variables as u64) as i32;
                    if next(&mut seed) & 1 == 0 { variable } else { -variable }
                })
                .collect();
            formula.push(clause);
        }
        formulas += 1;
        reduced += usize::from(assert_models_are_preserved(&formula) > 0);
    }
    assert!(reduced * 3 > formulas, "Only {reduced} of {formulas} systems were reduced");
}

#[test]
fn wide_rows_are_chained_through_auxiliaries_that_stay_out_of_the_model() {
    // Two overlapping equations over twelve variables: their sum has ten variables.
    let mut formula = Vec::new();
    xor_clauses(&[1, 2, 3, 4, 5, 6, 7], true, &mut formula);
    xor_clauses(&[1, 2, 8, 9, 10, 11, 12], false, &mut formula);
    let reduction = reduce_parity(&formula);
    assert_eq!((reduction.equations, reduction.rank), (2, 2));
    assert!(reduction.auxiliaries > 0);
    // One definition of three literals per link, and a unit on the last link.
    assert!(reduction.clauses.iter().all(|clause| clause.len() == 3 || clause.len() == 1));
    assert_eq!(reduction.clauses.iter().filter(|clause| clause.len() == 1).count(), 1);
    assert_models_are_preserved(&formula);

    let report = solve_hybrid(&formula, HybridConfig::default());
    let model = report.model.expect("The system is consistent");
    assert!(verify_model(&formula, &model));
    let assigned: Vec<u32> = model.iter().map(|literal| literal.unsigned_abs()).collect();
    assert_eq!(assigned, variable_ids(&formula));
    assert_eq!(report.parity.auxiliaries, reduction.auxiliaries);
}

#[test]
fn the_largest_identifier_leaves_no_room_for_auxiliaries_and_stays_exact() {
    let high = i32::MAX;
    let mut formula = Vec::new();
    xor_clauses(&[1, 2, 3, 4, 5, 6, high], true, &mut formula);
    xor_clauses(&[1, 2, 8, 9, 10, 11, 12], false, &mut formula);
    let reduction = reduce_parity(&formula);
    assert_eq!(reduction.auxiliaries, 0);
    let report = solve_hybrid(&formula, HybridConfig::default());
    assert!(verify_model(&formula, &report.model.expect("The system is consistent")));
}

fn planted(variables: usize, equations: usize, mut seed: u64) -> Vec<Vec<i32>> {
    let secret: Vec<bool> = (0..variables).map(|_| next(&mut seed) & 1 == 1).collect();
    let mut formula = Vec::new();
    for _ in 0..equations {
        let mut row: Vec<i32> = Vec::new();
        while row.len() < 3 {
            let variable = 1 + (next(&mut seed) % variables as u64) as i32;
            if !row.contains(&variable) {
                row.push(variable);
            }
        }
        let value = row.iter().fold(false, |parity, &variable| parity ^ secret[variable as usize - 1]);
        xor_clauses(&row, value, &mut formula);
    }
    formula
}

#[test]
fn planted_linear_systems_are_solved_without_a_conflict_explosion() {
    for (variables, equations) in [(60, 72), (120, 150), (120, 90), (200, 160)] {
        let formula = planted(variables, equations, 0x5be0_cd19_137e_2179);
        let report = solve_hybrid(&formula, HybridConfig::default());
        let model = report.model.expect("A planted system is satisfiable");
        assert!(verify_model(&formula, &model));
        assert!(report.parity.equations > 0 && report.parity.rank > 0);
        // The reduced rows propagate: the search never needs more conflicts than variables.
        assert!(
            report.cdcl.conflicts <= variables,
            "{} conflicts on {variables} variables",
            report.cdcl.conflicts
        );
    }
}

#[test]
fn formulas_without_parity_are_left_alone() {
    let cases = [
        vec![],
        vec![vec![]],
        vec![vec![1, 2, 3], vec![-1, 2], vec![-2, 3]],
        vec![vec![1, -1], vec![2, 3, 4, 5, 6, 7, 8, 9, 10]],
    ];
    for formula in cases {
        let reduction = reduce_parity(&formula);
        assert!(reduction.clauses.is_empty());
        assert_eq!(reduction.auxiliaries, 0);
        assert!(!reduction.refuted);
    }
    // A unit is an equation of one variable, but its reduced row is the unit itself.
    let reduction = reduce_parity(&[vec![1], vec![-2], vec![3, 4]]);
    assert_eq!(reduction.equations, 2);
    assert!(reduction.clauses.is_empty());
}

#[test]
fn the_extractor_reports_the_equations_over_the_original_identifiers() {
    // Two equations over sparse identifiers, a unit and an ordinary clause.
    let mut formula = vec![vec![7, 900, 13]];
    xor_clauses(&[900, 7, 4000], true, &mut formula);
    xor_clauses(&[13, 7], false, &mut formula);
    formula.push(vec![-52]);
    let mut equations = xor_equations(&formula);
    equations.sort();
    assert_eq!(equations, vec![(vec![7, 13], false), (vec![7, 900, 4000], true), (vec![52], false)]);
    // The same ones the reduction counts, and asking does not disturb it.
    assert_eq!(reduce_parity(&formula).equations, 3);
    // Three of the four clauses are not an equation, and an empty clause voids everything.
    let partial = vec![vec![1, 2, 3], vec![-1, -2, 3], vec![-1, 2, -3]];
    assert!(xor_equations(&partial).is_empty());
    formula.push(vec![]);
    assert!(xor_equations(&formula).is_empty());
}
