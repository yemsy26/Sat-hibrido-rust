//! Números de Ramsey conocidos con el DFS de referencia (espejo CPU del kernel K43).
//! Cada testigo SAT se comprueba con subconjuntos explícitos, sin el oráculo del DFS.
//! Los casos pesados llevan #[ignore]: `cargo test --release -- --include-ignored`.
use motor_core::ramsey_ref::{
    allowed_colours, edge_id, knuth_estimate, known_ramsey, paley, solve, solve_stars, Colouring,
    Config, Heuristic, Outcome, Policy, Problem, BLUE, RED,
};

const BUDGET: u64 = 200_000_000;

fn policy() -> Policy {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/neuro_ramsey_policy.onnx");
    Policy::from_onnx(&std::fs::read(path).unwrap()).unwrap()
}

const LEX: (Heuristic, bool) = (Heuristic::Lex, false);
const LEX_LA: (Heuristic, bool) = (Heuristic::Lex, true);
const FAIL_FIRST: (Heuristic, bool) = (Heuristic::FailFirst, true);
const NEURO_32: (Heuristic, bool) = (Heuristic::Neural { interval: 32 }, true);
const NEURO_1: (Heuristic, bool) = (Heuristic::Neural { interval: 1 }, true);
const RANDOM_32: (Heuristic, bool) = (Heuristic::Random { interval: 32 }, true);
const RANDOM_1: (Heuristic, bool) = (Heuristic::Random { interval: 1 }, true);
const WITH_LOOKAHEAD: [(Heuristic, bool); 6] =
    [LEX_LA, FAIL_FIRST, NEURO_32, NEURO_1, RANDOM_32, RANDOM_1];

/// ¿K_size monocromático? Recorre subconjuntos con la matriz de colores, sin bitsets.
fn brute_clique(graph: &Colouring, colour: u32, size: usize) -> bool {
    fn extend(graph: &Colouring, colour: u32, chosen: &mut Vec<usize>, size: usize) -> bool {
        if chosen.len() == size {
            return true;
        }
        let next = chosen.last().map_or(0, |&v| v + 1);
        for v in next..graph.n() {
            if chosen.iter().all(|&u| graph.colour(u, v) == Some(colour)) {
                chosen.push(v);
                if extend(graph, colour, chosen, size) {
                    return true;
                }
                chosen.pop();
            }
        }
        false
    }
    extend(graph, colour, &mut Vec::new(), size)
}

fn assert_witness(problem: Problem, graph: &Colouring) {
    assert_eq!(graph.free_edges(), 0, "testigo incompleto");
    assert!(!brute_clique(graph, RED, problem.s as usize), "K{} rojo", problem.s);
    assert!(!brute_clique(graph, BLUE, problem.t as usize), "K{} azul", problem.t);
}

/// Ejecuta cada estrategia y exige el veredicto conocido: K_{R-1} SAT y K_R UNSAT.
fn check(s: u32, t: u32, n: usize, stars: bool, strategies: &[(Heuristic, bool)]) {
    let r = known_ramsey(s, t).unwrap() as usize;
    assert!(n == r || n == r - 1);
    let problem = Problem::new(s, t, n).unwrap();
    let policy = policy();
    for &(heuristic, lookahead) in strategies {
        let config = Config::new(heuristic, lookahead, BUDGET);
        let report = if stars {
            solve_stars(problem, config, Some(&policy))
        } else {
            solve(problem, &Colouring::empty(n), config, Some(&policy))
        }
        .unwrap();
        let label = format!("R({s},{t}) K{n} {heuristic:?} lookahead={lookahead} estrellas={stars}");
        match report.outcome {
            Outcome::Sat(graph) => {
                assert_eq!(n, r - 1, "{label}: SAT en K_R");
                assert_witness(problem, &graph);
            }
            Outcome::Unsat => assert_eq!(n, r, "{label}: UNSAT en K_(R-1)"),
            Outcome::Unknown => panic!("{label}: presupuesto agotado ({BUDGET})"),
        }
    }
}

#[test]
fn k43_parameters_match_kernel_constants() {
    let k43 = Problem::new(5, 5, 43).unwrap();
    assert_eq!(k43.edges(), 903);
    assert_eq!((k43.red_cap(), k43.blue_cap()), (24, 24));
    assert_eq!(k43.star_degrees(), 18..=21);
    for a in 0..43 {
        for b in a + 1..43 {
            assert_eq!(edge_id(43, a, b), a * (85 - a) / 2 + b - a - 1);
        }
    }
}

#[test]
fn canonical_stars_match_k43_star_files() {
    let k43 = Problem::new(5, 5, 43).unwrap();
    for degree in k43.star_degrees() {
        let path = format!("{}/results/k43_star_{degree}.sat", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(path).unwrap();
        let mut from_file = Colouring::empty(43);
        for literal in text.lines().filter(|l| l.starts_with("v ")).flat_map(|l| l[2..].split_whitespace()) {
            let literal: i64 = literal.parse().unwrap();
            if literal == 0 {
                continue;
            }
            let id = literal.unsigned_abs() as usize - 1;
            let (a, b) = (0..43)
                .flat_map(|a| (a + 1..43).map(move |b| (a, b)))
                .find(|&(a, b)| edge_id(43, a, b) == id)
                .unwrap();
            from_file.set(a, b, Some(if literal > 0 { RED } else { BLUE }));
        }
        assert_eq!(from_file, k43.canonical_star(degree));
    }
}

/// Las cotas R(s-1,t)+R(s,t-1) ya refutan K_R en tres de los cinco casos: allí el
/// UNSAT por estrellas no ejercita el DFS, y por eso también se resuelve desde vacío.
#[test]
fn degree_caps_alone_refute_some_known_cases() {
    for (s, t, trivial) in [(3, 3, true), (3, 4, false), (3, 5, true), (4, 4, true), (3, 6, false)] {
        let n = known_ramsey(s, t).unwrap() as usize;
        let problem = Problem::new(s, t, n).unwrap();
        assert_eq!(problem.star_degrees().is_empty(), trivial, "R({s},{t})");
        assert!(!Problem::new(s, t, n - 1).unwrap().star_degrees().is_empty());
    }
    assert!(!Problem::new(5, 5, 43).unwrap().star_degrees().is_empty());
}

/// El oráculo de lookahead frente a la definición: grado lleno o K nuevo con (a,b).
#[test]
fn allowed_colours_match_brute_force() {
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for (s, t, n) in [(3, 3, 5), (3, 4, 8), (3, 5, 11), (4, 4, 10), (3, 6, 12)] {
        let problem = Problem::new(s, t, n).unwrap();
        let mut checked = 0;
        while checked < 400 {
            let mut graph = Colouring::empty(n);
            for a in 0..n {
                for b in a + 1..n {
                    match next() % 4 {
                        0 => graph.set(a, b, Some(RED)),
                        1 | 2 => graph.set(a, b, Some(BLUE)),
                        _ => {}
                    }
                }
            }
            // El oráculo exige un padre sin K monocromático.
            if brute_clique(&graph, RED, s as usize) || brute_clique(&graph, BLUE, t as usize) {
                continue;
            }
            let free: Vec<_> = (0..n)
                .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
                .filter(|&(a, b)| graph.colour(a, b).is_none())
                .collect();
            for (a, b) in free {
                let mut expected = 0;
                for (colour, size, cap) in [(RED, s, problem.red_cap()), (BLUE, t, problem.blue_cap())] {
                    let mut child = graph.clone();
                    child.set(a, b, Some(colour));
                    let full = child.degree(colour, a) > cap || child.degree(colour, b) > cap;
                    if !full && !brute_clique(&child, colour, size as usize) {
                        expected |= 1 << colour;
                    }
                }
                assert_eq!(allowed_colours(&problem, &graph, a, b), expected, "R({s},{t}) K{n} ({a},{b})");
                checked += 1;
            }
        }
    }
}

#[test]
fn paley17_is_the_r44_witness_and_is_recovered() {
    let problem = Problem::new(4, 4, 17).unwrap();
    let graph = paley(17);
    assert_witness(problem, &graph);
    // srg(17,8,3,4): caracteriza al único (4,4)-grafo de 17 vértices.
    for v in 0..17 {
        assert_eq!(graph.degree(RED, v), 8);
    }
    let full = solve(problem, &graph, Config::new(Heuristic::Lex, true, 10), None).unwrap();
    assert_eq!(full.outcome, Outcome::Sat(graph.clone()));
    // Sin las aristas de un vértice el DFS debe completar de nuevo la base.
    let mut partial = graph.clone();
    for v in 0..16 {
        partial.set(v, 16, None);
    }
    for (heuristic, lookahead) in [LEX, LEX_LA, FAIL_FIRST] {
        let report = solve(problem, &partial, Config::new(heuristic, lookahead, BUDGET), None).unwrap();
        let Outcome::Sat(found) = report.outcome else { panic!("{heuristic:?}") };
        assert_witness(problem, &found);
        for a in 0..16 {
            for b in a + 1..16 {
                assert_eq!(found.colour(a, b), partial.colour(a, b));
            }
        }
    }
}

/// Testigo de K17 para R(4,4) encontrado por el DFS: debe ser srg(17,8,3,4), es decir Paley.
#[test]
fn r44_k17_witness_is_strongly_regular() {
    let problem = Problem::new(4, 4, 17).unwrap();
    let report = solve_stars(problem, Config::new(Heuristic::FailFirst, true, BUDGET), None).unwrap();
    let Outcome::Sat(graph) = report.outcome else {
        panic!("R(4,4) K17 debería ser SAT: {:?}", report.outcome)
    };
    assert_witness(problem, &graph);
    for a in 0..17 {
        assert_eq!(graph.degree(RED, a), 8);
        for b in a + 1..17 {
            let common = (graph.row(RED, a) & graph.row(RED, b)).count_ones();
            let expected = if graph.colour(a, b) == Some(RED) { 3 } else { 4 };
            assert_eq!(common, expected, "({a},{b})");
        }
    }
}

/// Agotar un árbol de N nodos (raíz + asignaciones) cuesta 2N transiciones, y la media de
/// Knuth queda a menos de 4 errores estándar de ese tamaño exacto.
#[test]
fn knuth_estimate_matches_exhausted_trees() {
    for (s, t, n, stars, heuristic) in [
        (3, 4, 9, true, Heuristic::Lex),
        (3, 4, 9, true, Heuristic::FailFirst),
        (3, 4, 9, false, Heuristic::Lex),
        (3, 4, 9, false, Heuristic::FailFirst),
        (4, 4, 18, false, Heuristic::Lex),
    ] {
        let problem = Problem::new(s, t, n).unwrap();
        let bases: Vec<_> = if stars {
            problem.star_degrees().map(|d| problem.canonical_star(d)).collect()
        } else {
            vec![Colouring::empty(n)]
        };
        let (mut exact, mut nodes, mut variance) = (0, 0.0, 0.0);
        for (class, base) in bases.iter().enumerate() {
            let report = solve(problem, base, Config::new(heuristic, true, BUDGET), None).unwrap();
            assert_eq!(report.outcome, Outcome::Unsat);
            let c = report.counters;
            assert_eq!(c.transitions, 2 * (c.decisions + c.forced + 1));
            exact += c.transitions;
            let estimate = knuth_estimate(problem, base, heuristic, true, 20_000, 7 + class as u64).unwrap();
            nodes += estimate.nodes;
            variance += estimate.std_error * estimate.std_error;
        }
        let label = format!("R({s},{t}) K{n} estrellas={stars} {heuristic:?}");
        assert!((2.0 * nodes - exact as f64).abs() <= 8.0 * variance.sqrt(), "{label}: {nodes} vs {exact}");
    }
}

#[test]
fn exhausted_budget_is_unknown_not_unsat() {
    let problem = Problem::new(3, 4, 9).unwrap();
    let report = solve(problem, &Colouring::empty(9), Config::new(Heuristic::Lex, false, 1000), None).unwrap();
    assert_eq!(report.outcome, Outcome::Unknown);
    assert_eq!(report.counters.transitions, 1000);
}

#[test]
fn policy_loads_and_scores_free_edges() {
    let policy = policy();
    let problem = Problem::new(4, 4, 17).unwrap();
    let base = problem.canonical_star(8);
    let (a, b, colour) = policy.choose(&problem, &base, 0).expect("costes finitos");
    assert!(base.colour(a, b).is_none() && colour <= BLUE);
    assert_eq!(policy.choose(&problem, &base, 0), Some((a, b, colour)));
    assert!(policy.costs(&[0.0; 32]).iter().all(|c| c.is_finite() && *c >= 0.0));
}

#[test]
fn r33() {
    let all = [LEX, LEX_LA, FAIL_FIRST, NEURO_32, NEURO_1, RANDOM_32, RANDOM_1];
    for n in [5, 6] {
        check(3, 3, n, false, &all);
        check(3, 3, n, true, &all);
    }
}

#[test]
fn r34() {
    let all = [LEX, LEX_LA, FAIL_FIRST, NEURO_32, NEURO_1, RANDOM_32, RANDOM_1];
    check(3, 4, 8, false, &all);
    check(3, 4, 8, true, &all);
    check(3, 4, 9, false, &WITH_LOOKAHEAD);
    check(3, 4, 9, true, &all);
}

#[test]
fn r35() {
    let quick = [LEX_LA, FAIL_FIRST, NEURO_32];
    check(3, 5, 13, false, &quick);
    check(3, 5, 13, true, &WITH_LOOKAHEAD);
    check(3, 5, 14, false, &quick);
    check(3, 5, 14, true, &[LEX, LEX_LA, FAIL_FIRST, NEURO_32, NEURO_1]);
}

/// K17 se cubre con Paley y con el testigo srg; aquí K18 = R(4,4).
#[test]
fn r44() {
    check(4, 4, 18, false, &[LEX_LA, NEURO_32]);
    check(4, 4, 18, true, &[LEX, LEX_LA, FAIL_FIRST, NEURO_32, NEURO_1]);
}

#[test]
#[ignore = "pesado: cargo test --release -- --ignored"]
fn r44_k18_fail_first_from_empty() {
    check(4, 4, 18, false, &[FAIL_FIRST]);
}

/// K17 sí: fail-first con la semilla 0 encuentra un (3,6)-grafo. K18 = R(3,6) queda fuera
/// de alcance: Knuth estima ≥ 4·10^16 transiciones con la mejor estrategia (fail-first por
/// estrellas) y ~6·10^19 con lex+LA; ver results/validacion_ramsey.md.
#[test]
fn r36_k17() {
    check(3, 6, 17, false, &[FAIL_FIRST]);
}

#[test]
#[ignore = "pesado: cargo test --release -- --ignored"]
fn r36_k17_lex_lookahead() {
    check(3, 6, 17, false, &[LEX_LA]);
}
