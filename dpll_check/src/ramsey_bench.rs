//! Banco de validación: el DFS de referencia (espejo CPU del kernel K43) sobre números de
//! Ramsey conocidos, con tamaño de árbol por heurística. No usa GPU ni toca el asedio K43.
#![forbid(unsafe_code)]
#![deny(warnings)]

use crate::ramsey_ref::{
    knuth_estimate, known_ramsey, solve, solve_stars, Colouring, Config, Heuristic, Outcome, Policy, Problem,
    Report,
};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const HELP: &str = "ramsey_bench [--budget N] [--seeds K] [--jobs J] [--model red.onnx] [--out tabla.md]\n\
             [--only s,t[,n]] [--root vacía|estrellas] [--strategy lex,lex+LA,fail-first+LA,neuro/32+LA,azar/32+LA,neuro/1+LA,azar/1+LA]\n\
Para cada R(s,t) conocido ejecuta K_{R-1} (debe ser SAT) y K_R (debe ser UNSAT) con\n\
lex sin lookahead, lex+lookahead, fail-first, la política neuronal (intervalos 32 y 1)\n\
y su control azar (mismos 32 candidatos y cadencia, elección uniforme),\n\
desde la raíz vacía y por clases de estrella del vértice 0 (como --canonical-stars).\n\
--budget: transiciones por ejecución (y por clase); agotarlo da UNKNOWN. Defecto 50000000.\n\
--seeds: semillas 0..K del primer color; se informa la mediana y el rango. Defecto 3.\n\
En K_R (UNSAT) lex y fail-first usan sólo la semilla 0: su árbol no depende de ella.\n\
--jobs: ejecuciones en paralelo, una por hilo; el tiempo es por ejecución. Defecto: núcleos.
--knuth P: en lugar de ejecutar, estima con P sondas de Knuth el árbol completo de K_R
(lex y fail-first; transiciones = 2 x nodos). Útil donde el presupuesto no alcanza.";

const CASES: [(u32, u32); 5] = [(3, 3), (3, 4), (3, 5), (4, 4), (3, 6)];

// El banco funciona desde cualquier carpeta: --model solo hace falta para probar otra red.
const EMBEDDED_POLICY: &[u8] = include_bytes!("../neuro_ramsey_policy.onnx");

fn strategies() -> [(&'static str, Heuristic, bool); 7] {
    [
        ("lex", Heuristic::Lex, false),
        ("lex+LA", Heuristic::Lex, true),
        ("fail-first+LA", Heuristic::FailFirst, true),
        ("neuro/32+LA", Heuristic::Neural { interval: 32 }, true),
        ("azar/32+LA", Heuristic::Random { interval: 32 }, true),
        ("neuro/1+LA", Heuristic::Neural { interval: 1 }, true),
        ("azar/1+LA", Heuristic::Random { interval: 1 }, true),
    ]
}

/// Políticas cuya elección depende de la historia (contador de decisiones o azar).
fn history_dependent(heuristic: Heuristic) -> bool {
    matches!(heuristic, Heuristic::Neural { .. } | Heuristic::Random { .. })
}

fn verdict(outcome: &Outcome) -> &'static str {
    match outcome {
        Outcome::Sat(_) => "SAT",
        Outcome::Unsat => "UNSAT",
        Outcome::Unknown => "UNKNOWN",
    }
}

fn median(mut values: Vec<u64>) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

pub fn run(arguments: &[String]) {
    let mut budget = 50_000_000_u64;
    let mut seeds = 3_u32;
    let mut jobs = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut model: Option<String> = None;
    let mut out = None;
    let mut only: Option<(u32, u32, Option<usize>)> = None;
    let mut strategy: Option<String> = None;
    let mut root_filter: Option<String> = None;
    let mut knuth = 0_u64;
    let mut args = arguments.iter().cloned();
    while let Some(flag) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| fail(&format!("falta valor de {flag}")));
        match flag.as_str() {
            "--budget" => budget = value().parse().unwrap_or_else(|_| fail("--budget inválido")),
            "--seeds" => {
                seeds = value().parse().unwrap_or_else(|_| fail("--seeds inválido"));
                if seeds == 0 {
                    fail("--seeds >= 1");
                }
            }
            "--jobs" => jobs = value().parse::<usize>().unwrap_or_else(|_| fail("--jobs inválido")).max(1),
            "--model" => model = Some(value()),
            "--out" => out = Some(value()),
            "--only" => {
                let parts: Vec<u32> = value()
                    .split(',')
                    .map(|p| p.parse().unwrap_or_else(|_| fail("--only s,t[,n]")))
                    .collect();
                if !(2..=3).contains(&parts.len()) {
                    fail("--only s,t[,n]");
                }
                only = Some((parts[0], parts[1], parts.get(2).map(|&n| n as usize)));
            }
            "--strategy" => strategy = Some(value()),
            "--root" => root_filter = Some(value()),
            "--knuth" => knuth = value().parse().unwrap_or_else(|_| fail("--knuth inválido")),
            "--help" | "-h" => {
                println!("{HELP}");
                return;
            }
            other => fail(&format!("opción desconocida {other}\n{HELP}")),
        }
    }
    let policy = match &model {
        Some(path) => {
            let bytes = std::fs::read(path).unwrap_or_else(|e| fail(&format!("{path}: {e}")));
            Policy::from_onnx(&bytes).unwrap_or_else(|e| fail(&format!("{path}: {e}")))
        }
        None => Policy::from_onnx(EMBEDDED_POLICY).unwrap_or_else(|e| fail(&format!("modelo incluido: {e}"))),
    };

    // Una fila por (caso, raíz, estrategia); cada semilla es una tarea independiente.
    struct Row {
        s: u32,
        t: u32,
        problem: Problem,
        expected: &'static str,
        root: String,
        stars: bool,
        name: &'static str,
        heuristic: Heuristic,
        lookahead: bool,
    }
    let mut rows = Vec::new();
    let mut tasks = Vec::new();
    // Un K_n fuera de la lista (p. ej. 5,5,43) sólo se estima: su veredicto no se conoce.
    let instances: Vec<(u32, u32, usize, &str)> = match only {
        Some((s, t, Some(n))) if !CASES.contains(&(s, t)) => {
            if knuth == 0 {
                fail("un caso sin R(s,t) conocido sólo admite --knuth");
            }
            vec![(s, t, n, "abierto")]
        }
        _ => CASES
            .into_iter()
            .filter(|&c| only.map_or(true, |o| (o.0, o.1) == c))
            .flat_map(|(s, t)| {
                let r = known_ramsey(s, t).expect("caso conocido") as usize;
                [(s, t, r - 1, "SAT"), (s, t, r, "UNSAT")]
            })
            .filter(|&(_, _, n, _)| only.map_or(true, |o| o.2.map_or(true, |m| m == n)))
            .collect(),
    };
    for (s, t, n, expected) in instances {
        let problem = Problem::new(s, t, n).unwrap_or_else(|e| fail(&e));
        let degrees = problem.star_degrees();
        let star_label = if degrees.is_empty() {
            "estrellas: ninguna (cotas)".to_string()
        } else {
            format!("estrellas {}..{}", degrees.start(), degrees.end())
        };
        for (root, stars) in [("vacía".to_string(), false), (star_label, true)] {
            if root_filter.as_deref().is_some_and(|f| !root.starts_with(f)) {
                continue;
            }
            for (name, heuristic, lookahead) in strategies() {
                if strategy.as_deref().is_some_and(|f| !f.split(',').any(|x| x == name)) {
                    continue;
                }
                // Agotar un árbol con selección que sólo depende del estado da el mismo
                // árbol con cualquier primer color; las políticas dependen de la historia.
                let runs = if expected == "UNSAT" && !history_dependent(heuristic) { 1 } else { seeds };
                tasks.extend((0..runs).map(|seed| (rows.len(), seed)));
                rows.push(Row { s, t, problem, expected, root: root.clone(), stars, name, heuristic, lookahead });
            }
        }
    }
    if knuth != 0 {
        // K_R (o un caso abierto): su árbol completo es lo que una prueba UNSAT debe agotar.
        let selected: Vec<&Row> = rows
            .iter()
            .filter(|r| r.expected != "SAT" && !history_dependent(r.heuristic))
            .collect();
        let estimates: Vec<Mutex<Option<(f64, f64)>>> = selected.iter().map(|_| Mutex::new(None)).collect();
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..jobs {
                scope.spawn(|| loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(row) = selected.get(index) else { break };
                    let bases: Vec<Colouring> = if row.stars {
                        row.problem.star_degrees().map(|d| row.problem.canonical_star(d)).collect()
                    } else {
                        vec![Colouring::empty(row.problem.n)]
                    };
                    let (mut nodes, mut variance) = (0.0, 0.0);
                    for (class, base) in bases.iter().enumerate() {
                        let estimate = knuth_estimate(
                            row.problem, base, row.heuristic, row.lookahead, knuth, 0x9e37_79b9 + class as u64,
                        )
                        .unwrap_or_else(|e| fail(&e));
                        nodes += estimate.nodes;
                        variance += estimate.std_error * estimate.std_error;
                    }
                    *estimates[index].lock().unwrap() = Some((nodes, variance.sqrt()));
                });
            }
        });
        let mut table = format!(
            "| caso | raíz | heurística | transiciones estimadas (2N̂, {knuth} sondas) | error estándar |
|---|---|---|---:|---:|
"
        );
        for (row, estimate) in selected.iter().zip(&estimates) {
            let (nodes, error) = estimate.lock().unwrap().expect("estimación terminada");
            let _ = writeln!(
                table,
                "| R({},{}) K{} | {} | {} | {:.3e} | {:.1e} |",
                row.s, row.t, row.problem.n, row.root, row.name, 2.0 * nodes, 2.0 * error
            );
        }
        print!("{table}");
        if let Some(path) = out {
            std::fs::write(&path, &table).unwrap_or_else(|e| fail(&format!("{path}: {e}")));
        }
        return;
    }
    let results: Vec<Mutex<Option<Report>>> = tasks.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(row, seed)) = tasks.get(index) else { break };
                let row = &rows[row];
                let mut config = Config::new(row.heuristic, row.lookahead, budget);
                config.seed = seed;
                let report = if row.stars {
                    solve_stars(row.problem, config, Some(&policy))
                } else {
                    solve(row.problem, &Colouring::empty(row.problem.n), config, Some(&policy))
                }
                .unwrap_or_else(|e| fail(&e));
                if let Outcome::Sat(model) = &report.outcome {
                    if model.has_clique(0, row.s) || model.has_clique(1, row.t) || model.free_edges() != 0 {
                        fail("testigo SAT inválido");
                    }
                }
                eprintln!(
                    "c R({},{}) K{} {} {} semilla {seed}: {} en {:.1} s",
                    row.s, row.t, row.problem.n, row.root, row.name,
                    verdict(&report.outcome), report.elapsed.as_secs_f64()
                );
                *results[index].lock().unwrap() = Some(report);
            });
        }
    });

    let mut table = String::new();
    let _ = writeln!(
        table,
        "| caso | esperado | raíz | heurística | veredicto | transiciones (mediana) | mín–máx | decisiones | conflictos | checks | consultas | tiempo |"
    );
    let _ = writeln!(table, "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|");
    let mut wrong = 0;
    for (index, row) in rows.iter().enumerate() {
        let reports: Vec<Report> = tasks
            .iter()
            .zip(&results)
            .filter(|((r, _), _)| *r == index)
            .map(|(_, report)| report.lock().unwrap().take().expect("tarea terminada"))
            .collect();
        let verdicts: Vec<&str> = reports.iter().map(|r| verdict(&r.outcome)).collect();
        wrong += verdicts.iter().filter(|&&v| v != "UNKNOWN" && v != row.expected).count();
        let got = if verdicts.iter().all(|&v| v == verdicts[0]) {
            verdicts[0].to_string()
        } else {
            verdicts.join("/")
        };
        let stat = |f: fn(&Report) -> u64| median(reports.iter().map(f).collect());
        let transitions: Vec<u64> = reports.iter().map(|r| r.counters.transitions).collect();
        let elapsed: f64 = reports.iter().map(|r| r.elapsed.as_secs_f64()).sum();
        let _ = writeln!(
            table,
            "| R({},{}) K{} | {} | {} | {} | {got} | {} | {}–{} | {} | {} | {} | {} | {elapsed:.3} s |",
            row.s,
            row.t,
            row.problem.n,
            row.expected,
            row.root,
            row.name,
            median(transitions.clone()),
            transitions.iter().min().unwrap(),
            transitions.iter().max().unwrap(),
            stat(|r| r.counters.decisions),
            stat(|r| r.counters.conflicts),
            stat(|r| r.counters.checks),
            stat(|r| r.counters.policy_queries),
        );
    }
    print!("{table}");
    if let Some(path) = out {
        std::fs::write(&path, &table).unwrap_or_else(|e| fail(&format!("{path}: {e}")));
    }
    if wrong != 0 {
        fail(&format!("{wrong} veredictos contradicen el valor conocido"));
    }
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(2);
}
