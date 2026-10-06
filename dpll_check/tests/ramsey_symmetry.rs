#![forbid(unsafe_code)]
#![deny(warnings)]

#[allow(dead_code)]
#[path = "../src/hybrid.rs"]
mod hybrid;
#[allow(dead_code)]
#[path = "../src/symmetry_breaker.rs"]
mod symmetry_breaker;
#[allow(dead_code)]
#[path = "../src/ramsey_symmetry.rs"]
mod ramsey_symmetry;

use hybrid::{solve_hybrid, verify_model, HybridConfig};
use ramsey_symmetry::{edge_count, inject_ramsey_symmetries, is_vertex_symmetric};
use symmetry_breaker::SymmetryBreaker;

// The generator's numbering: edges {i, j}, 1 <= i < j <= n, in lexicographic order from 1.
fn edge(i: u32, j: u32, n: u32) -> i32 {
    ((i - 1) * n - (i - 1) * i / 2 + (j - i)) as i32
}

// No clique of `first` vertices in colour 1 and none of `second` vertices in colour 0.
fn ramsey(n: u32, first: u32, second: u32) -> Vec<Vec<i32>> {
    let mut formula = Vec::new();
    for (size, sign) in [(first, -1), (second, 1)] {
        for subset in 0..1_u32 << n {
            if subset.count_ones() != size {
                continue;
            }
            let clique: Vec<u32> = (1..=n).filter(|vertex| subset >> (vertex - 1) & 1 == 1).collect();
            let mut clause = Vec::new();
            for (position, &i) in clique.iter().enumerate() {
                for &j in &clique[position + 1..] {
                    clause.push(sign * edge(i, j, n));
                }
            }
            formula.push(clause);
        }
    }
    formula
}

#[test]
fn known_ramsey_numbers_survive_the_injected_ordering() {
    // R(3,3) = 6, R(3,4) = 9, R(3,5) = 14 and R(4,4) = 18: satisfiable strictly below.
    let cases = [
        (5, 3, 3, true), (6, 3, 3, false), (8, 3, 4, true), (9, 3, 4, false),
        (13, 3, 5, true), (14, 3, 5, false), (12, 4, 4, true),
    ];
    for (n, first, second, satisfiable) in cases {
        let original = ramsey(n, first, second);
        assert!(is_vertex_symmetric(n, &original));
        let edges = edge_count(n).unwrap();
        let mut breaker = SymmetryBreaker::new(edges + 1);
        let mut ordered = original.clone();
        inject_ramsey_symmetries(n, &mut breaker, &mut ordered);
        // Por cada par de vertices consecutivos, n - 2 posiciones: 3(n - 2) - 2 clausulas
        // y n - 3 variables auxiliares.
        assert_eq!(ordered.len() - original.len(), ((n - 1) * (3 * n - 8)) as usize);
        assert_eq!(breaker.next_free() - 1 - edges, (n - 1) * (n - 3));
        // With and without the engine's own symmetry layer on top of the injected clauses.
        for symmetry in [true, false] {
            let report = solve_hybrid(&ordered, HybridConfig { symmetry, ..HybridConfig::default() });
            assert_eq!(report.model.is_some(), satisfiable, "R({first},{second}) on {n} vertices");
            if let Some(model) = &report.model {
                // The colouring is a model of the formula nobody touched.
                assert!(verify_model(&original, model));
                assert!(verify_model(&ordered, model));
            }
        }
        // The answer of the untouched formula is the same.
        assert_eq!(solve_hybrid(&original, HybridConfig::default()).model.is_some(), satisfiable);
    }
}

#[test]
fn the_ordering_is_refused_on_a_formula_that_is_not_symmetric() {
    // Forcing one edge to a colour singles out two vertices: ordering all of them could
    // then remove every solution, so the launcher checks before injecting.
    let mut marked = ramsey(8, 3, 4);
    marked.push(vec![edge(3, 7, 8)]);
    assert!(!is_vertex_symmetric(8, &marked));
    assert!(is_vertex_symmetric(8, &ramsey(8, 3, 4)));
    // The wrong number of vertices is caught by the variables it uses or by the clauses.
    assert!(!is_vertex_symmetric(7, &ramsey(8, 3, 4)));
    assert!(!is_vertex_symmetric(9, &ramsey(8, 3, 4)));
}
