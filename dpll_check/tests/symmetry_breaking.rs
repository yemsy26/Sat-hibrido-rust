#![forbid(unsafe_code)]
#![deny(warnings)]

#[allow(dead_code)]
#[path = "../src/hybrid.rs"]
mod hybrid;

use hybrid::symmetry::break_symmetries;
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

// The formula as a set of clauses: literal order, repeats and tautologies do not matter.
fn normalized(formula: &[Vec<i32>]) -> BTreeSet<Vec<i32>> {
    formula
        .iter()
        .filter_map(|clause| {
            let literals: BTreeSet<i32> = clause.iter().copied().collect();
            let tautology = literals.iter().any(|literal| literals.contains(&-literal));
            (!tautology).then(|| literals.into_iter().collect())
        })
        .collect()
}

fn renamed(formula: &[Vec<i32>], generator: &[(u32, u32)]) -> Vec<Vec<i32>> {
    formula
        .iter()
        .map(|clause| {
            clause
                .iter()
                .map(|&literal| {
                    let image = generator
                        .iter()
                        .find(|&&(variable, _)| variable == literal.unsigned_abs())
                        .map_or(literal.unsigned_abs(), |&(_, image)| image) as i32;
                    if literal > 0 {
                        image
                    } else {
                        -image
                    }
                })
                .collect()
        })
        .collect()
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

// Proper colouring: every vertex takes a colour and adjacent vertices never share one.
fn colouring(vertices: usize, edges: &[(usize, usize)], colours: usize) -> Vec<Vec<i32>> {
    let variable = |vertex: usize, colour: usize| (vertex * colours + colour + 1) as i32;
    let mut formula: Vec<Vec<i32>> = (0..vertices)
        .map(|vertex| (0..colours).map(|colour| variable(vertex, colour)).collect())
        .collect();
    for &(first, second) in edges {
        for colour in 0..colours {
            formula.push(vec![-variable(first, colour), -variable(second, colour)]);
        }
    }
    formula
}

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn structured_formulas() -> Vec<Vec<Vec<i32>>> {
    let cycle: Vec<(usize, usize)> = (0..5).map(|vertex| (vertex, (vertex + 1) % 5)).collect();
    let complete: Vec<(usize, usize)> = (0..4)
        .flat_map(|first| (first + 1..4).map(move |second| (first, second)))
        .collect();
    let excluded: Vec<Vec<i32>> = (0..16_i32)
        .map(|bits| (0..4).map(|index| if bits & (1 << index) == 0 { index + 1 } else { -(index + 1) }).collect())
        .collect();
    vec![
        pigeonhole(2, 2),
        pigeonhole(3, 2),
        pigeonhole(3, 3),
        pigeonhole(4, 3),
        pigeonhole(4, 4),
        pigeonhole(5, 4),
        pigeonhole(4, 5),
        colouring(5, &cycle, 3),
        colouring(5, &cycle, 2),
        colouring(4, &complete, 3),
        colouring(4, &complete, 4),
        excluded,
        // Interchangeable variables inside clauses, with mixed polarities.
        vec![vec![1, 2, 3], vec![-1, -2], vec![-1, -3], vec![-2, -3], vec![4, 5], vec![-4, -5]],
        vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![-1, -3], vec![-3, -5], vec![-1, -5]],
        vec![vec![1, -2], vec![2, -3], vec![3, -1], vec![4, -5], vec![5, -6], vec![6, -4]],
    ]
}

#[test]
fn every_generator_is_an_automorphism_of_the_formula() {
    let mut generators = 0;
    for formula in structured_formulas() {
        let breaking = break_symmetries(&formula);
        let ids = variable_ids(&formula);
        for generator in &breaking.generators {
            assert!(!generator.is_empty());
            assert!(generator.iter().all(|&(variable, image)| variable != image
                && ids.binary_search(&variable).is_ok()
                && ids.binary_search(&image).is_ok()));
            assert_eq!(
                normalized(&renamed(&formula, generator)),
                normalized(&formula),
                "{generator:?} is not a symmetry of {formula:?}"
            );
            generators += 1;
        }
    }
    assert!(generators > 0, "The structured suite must exercise the layer");
}

fn assert_predicates_are_sound(formula: &[Vec<i32>]) -> usize {
    let breaking = break_symmetries(formula);
    let ids = variable_ids(formula);
    // No new variables: the predicates only mention what the formula already has.
    assert!(breaking
        .clauses
        .iter()
        .flatten()
        .all(|literal| ids.binary_search(&literal.unsigned_abs()).is_ok()));
    assert!(breaking.clauses.iter().all(|clause| !clause.is_empty()));
    let mut extended = formula.to_vec();
    extended.extend(breaking.clauses.iter().cloned());
    assert_eq!(
        brute_force(&extended),
        brute_force(formula),
        "Predicates {:?} changed the answer of {formula:?}",
        breaking.clauses
    );
    breaking.clauses.len()
}

#[test]
fn predicates_preserve_satisfiability_on_structured_formulas() {
    let mut predicates = 0;
    for formula in structured_formulas() {
        if variable_ids(&formula).len() <= 20 {
            predicates += assert_predicates_are_sound(&formula);
        }
    }
    assert!(predicates > 0);
}

#[test]
fn predicates_preserve_satisfiability_on_exhaustive_random_formulas() {
    // Few variables and many clauses: accidental symmetries are the rule, not the exception.
    let mut seed = 0x3c6e_f372_fe94_f82b;
    let (mut formulas, mut broken) = (0, 0);
    for n in 2..=9_u64 {
        for case in 0..160 {
            let clauses = (next(&mut seed) % (3 * n) + 1) as usize;
            let width = [2, 3, 3, 4][case % 4];
            let formula: Vec<Vec<i32>> = (0..clauses)
                .map(|_| {
                    (0..next(&mut seed) % width + 1)
                        .map(|_| {
                            let variable = (next(&mut seed) % n + 1) as i32;
                            // Mostly one polarity per case keeps more variables interchangeable.
                            if case % 3 != 0 || next(&mut seed) & 1 == 0 {
                                variable
                            } else {
                                -variable
                            }
                        })
                        .collect()
                })
                .collect();
            formulas += 1;
            broken += usize::from(assert_predicates_are_sound(&formula) > 0);
        }
    }
    assert!(broken * 4 > formulas, "Only {broken} of {formulas} formulas had symmetries");
}

#[test]
fn pigeonhole_is_refuted_at_the_root_without_built_in_knowledge() {
    for holes in [3, 5, 8, 12, 16] {
        let formula = pigeonhole(holes + 1, holes);
        let report = solve_hybrid(&formula, HybridConfig::default());
        assert!(report.model.is_none());
        assert!(report.symmetry.generators > 0 && report.symmetry.clauses > 0);
        // Unit propagation over the predicates is enough: no decision, one root conflict.
        assert_eq!(report.cdcl.decisions, 0);
        assert_eq!(report.cdcl.conflicts, 1);

        // Clause order, repeated clauses and tautologies are not structure.
        let mut disguised: Vec<Vec<i32>> = formula.iter().rev().cloned().collect();
        disguised.push(vec![1, -1]);
        disguised.push(formula[0].clone());
        let report = solve_hybrid(&disguised, HybridConfig::default());
        assert!(report.model.is_none());
        assert_eq!(report.cdcl.decisions, 0);
    }
}

#[test]
fn satisfiable_symmetric_instances_keep_a_model() {
    let cycle: Vec<(usize, usize)> = (0..9).map(|vertex| (vertex, (vertex + 1) % 9)).collect();
    let cases = [
        pigeonhole(5, 5),
        pigeonhole(6, 6),
        pigeonhole(5, 9),
        pigeonhole(8, 8),
        colouring(9, &cycle, 3),
        colouring(9, &cycle, 4),
    ];
    for formula in cases {
        for flips_per_variable in [0, 16] {
            let mut parameters = HybridConfig::default();
            parameters.oracle.flips_per_variable = flips_per_variable;
            let report = solve_hybrid(&formula, parameters);
            let model = report.model.expect("Symmetry breaking must keep one model per orbit");
            assert!(verify_model(&formula, &model));
            assert!(report.symmetry.clauses > 0);
        }
    }
}

#[test]
fn renamed_variables_keep_the_answer_exact() {
    // A scrambled numbering hides the row order the predicates rely on, never the answer.
    let mut seed = 0x1f83_d9ab_fb41_bd6b;
    for (pigeons, holes) in [(5, 4), (6, 5), (5, 5), (4, 6)] {
        let formula = pigeonhole(pigeons, holes);
        let mut ids: Vec<i32> = (1..=(pigeons * holes) as i32).map(|id| id * 7 + 3).collect();
        for index in (1..ids.len()).rev() {
            ids.swap(index, (next(&mut seed) % (index as u64 + 1)) as usize);
        }
        let scrambled: Vec<Vec<i32>> = formula
            .iter()
            .map(|clause| {
                clause
                    .iter()
                    .map(|&literal| {
                        let id = ids[literal.unsigned_abs() as usize - 1];
                        if literal > 0 { id } else { -id }
                    })
                    .collect()
            })
            .collect();
        let report = solve_hybrid(&scrambled, HybridConfig::default());
        assert_eq!(report.model.is_some(), pigeons <= holes);
        if let Some(model) = &report.model {
            assert!(verify_model(&scrambled, model));
        }
    }
}

#[test]
fn rigid_formulas_get_no_predicates() {
    // Every variable has its own occurrence profile: the only symmetry is the identity.
    let formula = vec![
        vec![1, 2, 3],
        vec![-1, 2],
        vec![-2, 4],
        vec![3, -4, 5],
        vec![-5, 1],
        vec![5, 6, -3],
        vec![6],
    ];
    let breaking = break_symmetries(&formula);
    assert!(breaking.generators.is_empty());
    assert!(breaking.clauses.is_empty());
    for formula in [vec![], vec![vec![]], vec![vec![1]], vec![vec![1, -1]]] {
        let breaking = break_symmetries(&formula);
        assert!(breaking.generators.is_empty() && breaking.clauses.is_empty());
    }
}

#[test]
fn disabling_the_layer_leaves_the_formula_alone() {
    let formula = pigeonhole(5, 4);
    let report = solve_hybrid(&formula, HybridConfig { symmetry: false, ..HybridConfig::default() });
    assert!(report.model.is_none());
    assert_eq!(report.symmetry.generators, 0);
    assert_eq!(report.symmetry.clauses, 0);
    assert_eq!(report.symmetry.ms, 0.0);
    // Without the predicates the refutation needs genuine search.
    assert!(report.cdcl.decisions > 0 && report.cdcl.conflicts > 1);
}
