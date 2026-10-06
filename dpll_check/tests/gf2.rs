#![forbid(unsafe_code)]
#![deny(warnings)]

#[allow(dead_code)]
#[path = "../src/hybrid.rs"]
mod hybrid;

use hybrid::cdcl::SwarmConfig;
use hybrid::gf2::Gf2Outcome;
use hybrid::{
    load_equations, solve_hybrid, solve_hybrid_gf2, solve_hybrid_parallel_gf2, verify_model, HybridConfig, OracleConfig,
};
use std::collections::BTreeSet;

// CNF of x1 xor ... xor xk = value: one clause per sign pattern of the opposite parity.
fn xor_clauses(variables: &[i32], value: bool, formula: &mut Vec<Vec<i32>>) {
    for pattern in 0..1_u32 << variables.len() {
        if (pattern.count_ones() & 1 == 1) != value {
            formula.push(variables.iter().enumerate()
                .map(|(bit, &variable)| if pattern >> bit & 1 == 1 { -variable } else { variable }).collect());
        }
    }
}

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn satisfiable(formula: &[Vec<i32>]) -> bool {
    let ids: Vec<u32> = formula.iter().flatten().map(|literal| literal.unsigned_abs())
        .collect::<BTreeSet<_>>().into_iter().collect();
    assert!(ids.len() <= 20, "Too many variables for an exhaustive check");
    (0..1_u64 << ids.len()).any(|bits| formula.iter().all(|clause| clause.iter().any(|&literal| {
        let index = ids.binary_search(&literal.unsigned_abs()).unwrap();
        (bits >> index & 1 == 1) == (literal > 0)
    })))
}

// A linear system hidden in CNF next to ordinary clauses.
fn mixed(variables: i32, equations: u64, clauses: u64, seed: &mut u64) -> Vec<Vec<i32>> {
    let mut formula = Vec::new();
    for _ in 0..equations {
        let width = 2 + (next(seed) % 3) as usize;
        let mut row: Vec<i32> = Vec::new();
        while row.len() < width.min(variables as usize) {
            let variable = 1 + (next(seed) % variables as u64) as i32;
            if !row.contains(&variable) {
                row.push(variable);
            }
        }
        xor_clauses(&row, next(seed) & 1 == 1, &mut formula);
    }
    for _ in 0..clauses {
        formula.push((0..2 + next(seed) % 2).map(|_| {
            let variable = 1 + (next(seed) % variables as u64) as i32;
            if next(seed) & 1 == 0 { variable } else { -variable }
        }).collect());
    }
    formula
}

fn config(seed: u64, symmetry: bool, parity: bool) -> HybridConfig {
    HybridConfig { oracle: OracleConfig { flips_per_variable: 2, seed }, reignition_interval: 4, symmetry, parity }
}

// The same without the oracle: every model has to come out of the search itself.
fn exact(seed: u64, symmetry: bool, parity: bool) -> HybridConfig {
    HybridConfig { reignition_interval: 0, ..config(seed, symmetry, parity) }
}

#[test]
fn the_search_with_equations_agrees_with_exhaustive_search() {
    let mut seed = 0x6a09_e667_bb67_ae85;
    let (mut refuted, mut forced, mut conflicts, mut searched) = (0, 0, 0, 0);
    for case in 0..500_u64 {
        let variables = 6 + (next(&mut seed) % 11) as i32;
        let formula = mixed(variables, 1 + next(&mut seed) % variables as u64, next(&mut seed) % (3 * variables as u64), &mut seed);
        let expected = satisfiable(&formula);
        // Every combination of symmetry, parity and oracle, with the equations always on.
        for layers in 0..8 {
            let build = if layers & 4 == 4 { exact } else { config };
            let parameters = build(case, layers & 1 == 1, layers & 2 == 2);
            let report = solve_hybrid_gf2(&formula, parameters);
            assert_eq!(report.model.is_some(), expected, "{formula:?} with {parameters:?}");
            if let Some(model) = &report.model {
                assert!(verify_model(&formula, model));
            }
            forced += report.cdcl.gf2_propagations;
            conflicts += report.cdcl.gf2_conflicts;
            searched += usize::from(report.cdcl.decisions > 0);
        }
        refuted += usize::from(!expected);
    }
    assert!(refuted > 50, "{refuted} refutations");
    // The equations really took part in searches that needed decisions.
    assert!(forced > 1500 && searched > 1000, "{forced} forced, {conflicts} conflicts, {searched} searches");
}

#[test]
fn rows_falsified_by_clause_propagation_become_ordinary_conflicts() {
    // Implication chains assign several variables of a row inside one clause fixpoint, so
    // the row goes from two free variables to none: the equations must then hand the
    // search a conflict clause it can analyse like any other.
    let mut seed = 0xbb67_ae85_84ca_a73b;
    let (mut conflicts, mut forced, mut refuted) = (0, 0, 0);
    for case in 0..400_u64 {
        let variables = 12 + (next(&mut seed) % 7) as i32;
        let mut formula = Vec::new();
        for _ in 0..variables / 2 {
            let width = 3 + (next(&mut seed) % 3) as usize;
            let mut row: Vec<i32> = Vec::new();
            while row.len() < width {
                let variable = 1 + (next(&mut seed) % variables as u64) as i32;
                if !row.contains(&variable) {
                    row.push(variable);
                }
            }
            xor_clauses(&row, next(&mut seed) & 1 == 1, &mut formula);
        }
        for _ in 0..3 * variables / 2 {
            let (a, b) = (1 + (next(&mut seed) % variables as u64) as i32, 1 + (next(&mut seed) % variables as u64) as i32);
            if a != b {
                formula.push(vec![if next(&mut seed) & 1 == 0 { a } else { -a }, if next(&mut seed) & 1 == 0 { b } else { -b }]);
            }
        }
        let expected = satisfiable(&formula);
        for symmetry in [false, true] {
            let report = solve_hybrid_gf2(&formula, exact(case, symmetry, false));
            assert_eq!(report.model.is_some(), expected, "{formula:?}");
            if let Some(model) = &report.model {
                assert!(verify_model(&formula, model));
            }
            conflicts += report.cdcl.gf2_conflicts;
            forced += report.cdcl.gf2_propagations;
        }
        refuted += usize::from(!expected);
    }
    assert!(conflicts > 20 && forced > 300 && refuted > 50, "{conflicts} conflicts, {forced} forced, {refuted} refutations");
}

#[test]
fn equations_known_only_to_the_propagator_are_enforced_exactly() {
    // The hardest check of the hook: the linear system is NOT in the clauses, so every
    // deduction and every conflict about it has to come from the in-search equations.
    // The answer must be that of clauses and equations together. Without the oracle,
    // whose model check only looks at clauses.
    use hybrid::cdcl::{solve_cdcl_with_equations, solve_swarm_with_equations, OracleConfig, ReignitionConfig};
    use hybrid::gf2::Gf2Propagator;
    let mut seed = 0xa54f_f53a_510e_527f;
    let exact = ReignitionConfig { interval: 0, oracle: OracleConfig::default() };
    let (mut conflicts, mut forced, mut refuted, mut models) = (0, 0, 0, 0);
    for case in 0..1500_u64 {
        let variables = 5 + (next(&mut seed) % 12) as i32;
        // Every variable occurs in the formula, in a tautology the engine drops.
        let mut clauses: Vec<Vec<i32>> = (1..=variables).map(|variable| vec![variable, -variable]).collect();
        for _ in 0..next(&mut seed) % (2 * variables as u64) {
            clauses.push((0..2 + next(&mut seed) % 2).map(|_| {
                let variable = 1 + (next(&mut seed) % variables as u64) as i32;
                if next(&mut seed) & 1 == 0 { variable } else { -variable }
            }).collect());
        }
        let mut system: Vec<(Vec<i32>, bool)> = Vec::new();
        for _ in 0..1 + next(&mut seed) % (variables as u64 / 2) {
            let width = 2 + (next(&mut seed) % 5) as usize;
            let mut row: Vec<i32> = Vec::new();
            while row.len() < width.min(variables as usize) {
                let variable = 1 + (next(&mut seed) % variables as u64) as i32;
                if !row.contains(&variable) {
                    row.push(variable);
                }
            }
            system.push((row, next(&mut seed) & 1 == 1));
        }
        // What the answer has to be: the clauses with the system written out as clauses.
        let mut whole = clauses.clone();
        for (row, constant) in &system {
            xor_clauses(row, *constant, &mut whole);
        }
        let expected = satisfiable(&whole);
        let build = || {
            let mut gf2 = Gf2Propagator::new(true);
            for (row, constant) in &system {
                gf2.add_xor(&row.iter().map(|&variable| variable as u32).collect::<Vec<u32>>(), *constant);
            }
            let outcome = gf2.eliminate();
            (gf2, outcome)
        };
        let (gf2, outcome) = build();
        if outcome == Gf2Outcome::Contradiction {
            assert!(!expected);
            refuted += 1;
            continue;
        }
        let (model, stats) = if case % 5 == 0 {
            let swarm = SwarmConfig { workers: 3, cubes_per_worker: 2, reignition: exact, ..SwarmConfig::default() };
            let (model, stats) = solve_swarm_with_equations(&clauses, swarm, gf2);
            (model, stats.total)
        } else {
            solve_cdcl_with_equations(&clauses, exact, gf2)
        };
        assert_eq!(model.is_some(), expected, "clauses {clauses:?} with system {system:?}");
        if let Some(model) = &model {
            assert!(verify_model(&whole, model), "The model breaks an equation: {system:?}");
            models += 1;
        }
        conflicts += stats.gf2_conflicts;
        forced += stats.gf2_propagations;
        refuted += usize::from(!expected);
    }
    assert!(
        conflicts > 300 && forced > 2000 && refuted > 100 && models > 1000,
        "{conflicts} conflicts, {forced} forced, {refuted} refutations, {models} models"
    );
}

#[test]
fn the_swarm_with_equations_agrees_with_exhaustive_search() {
    let mut seed = 0x3c6e_f372_a54f_f53a;
    for case in 0..120_u64 {
        let variables = 8 + (next(&mut seed) % 9) as i32;
        let formula = mixed(variables, 2 + next(&mut seed) % variables as u64, next(&mut seed) % (2 * variables as u64), &mut seed);
        let swarm = SwarmConfig { workers: 4, cubes_per_worker: 2, ..SwarmConfig::default() };
        let report = solve_hybrid_parallel_gf2(&formula, config(case, case % 2 == 0, false), swarm);
        assert_eq!(report.model.is_some(), satisfiable(&formula), "{formula:?}");
        if let Some(model) = &report.model {
            assert!(verify_model(&formula, model));
        }
    }
}

#[test]
fn a_contradictory_linear_system_is_answered_without_search() {
    // Tseitin on K4: the four equations sum to 0 = 1.
    let mut formula = Vec::new();
    for (vertex, incident) in [[1, 2, 3], [1, 4, 5], [2, 4, 6], [3, 5, 6]].iter().enumerate() {
        xor_clauses(incident, vertex == 0, &mut formula);
    }
    assert_eq!(load_equations(&formula, true).1, Gf2Outcome::Contradiction);
    // Even with the parity layer off, nothing is decided and nothing is preprocessed.
    let report = solve_hybrid_gf2(&formula, config(1, true, false));
    assert!(report.model.is_none());
    assert_eq!((report.cdcl.decisions, report.cdcl.conflicts, report.symmetry.generators), (0, 1, 0));
    let swarm = solve_hybrid_parallel_gf2(&formula, config(1, false, false), SwarmConfig::default());
    assert!(swarm.model.is_none());
    assert_eq!((swarm.swarm.workers, swarm.swarm.cubes_generated), (0, 0));
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
fn reduced_rows_propagate_where_the_clauses_alone_do_not() {
    // Planted systems with the parity layer off: only the in-search equations know the
    // reduced rows, and a fully determined system is solved by propagation at the root.
    for (variables, equations) in [(60, 72), (120, 150), (200, 240)] {
        let formula = planted(variables, equations, 0x5be0_cd19_137e_2179);
        let report = solve_hybrid_gf2(&formula, config(7, false, false));
        let model = report.model.expect("A planted system is satisfiable");
        assert!(verify_model(&formula, &model));
        assert!(report.cdcl.gf2_propagations > 0);
        assert!(report.cdcl.conflicts <= variables, "{} conflicts on {variables} variables", report.cdcl.conflicts);
    }
}

#[test]
fn the_classic_entry_point_never_touches_the_equations() {
    let mut seed = 0x1f83_d9ab_5be0_cd19;
    for case in 0..40_u64 {
        let formula = mixed(12, 5, 20, &mut seed);
        let report = solve_hybrid(&formula, config(case, true, true));
        assert_eq!((report.cdcl.gf2_propagations, report.cdcl.gf2_conflicts), (0, 0));
        assert_eq!(report.model.is_some(), satisfiable(&formula));
    }
    // Asking with the module off extracts nothing at all.
    let (equations, outcome) = load_equations(&mixed(12, 5, 20, &mut seed), false);
    assert_eq!((equations.equations(), outcome), (0, Gf2Outcome::Disabled));
}
