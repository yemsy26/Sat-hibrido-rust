//! Extremo a extremo del comando `motor`: veredicto, verificacion y codigos de salida.
//! Los casos que necesitan VeriPB llevan #[ignore]: `cargo test --release --test motor_cli -- --include-ignored`.
use motor_core::dimacs::parse_dimacs;
use motor_core::hybrid::cdcl::SwarmConfig;
use motor_core::hybrid::{solve_hybrid, solve_hybrid_parallel, verify_model, HybridConfig};
use motor_core::verifier::{check_unsat_proof, ProofCheck};
use std::path::PathBuf;
use std::process::{Command, Output};

const EXE: &str = env!("CARGO_BIN_EXE_motor");

// Formulas que no dependen de la busqueda para saber el veredicto.
const SAT: &str = "p cnf 3 3\n1 2 0\n-1 3 0\n-2 -3 0\n";
const UNSAT: &str = "p cnf 2 4\n1 2 0\n1 -2 0\n-1 2 0\n-1 -2 0\n";

struct Scratch(PathBuf);
impl Scratch {
    fn new(name: &str) -> Self {
        let directory = std::env::temp_dir().join(format!("motor_cli_{}_{name}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        Self(directory)
    }
    fn file(&self, name: &str, text: &str) -> String {
        let path = self.0.join(name);
        std::fs::write(&path, text).unwrap();
        path.to_str().unwrap().to_owned()
    }
    fn path(&self, name: &str) -> String {
        self.0.join(name).to_str().unwrap().to_owned()
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn motor(arguments: &[&str]) -> Output {
    Command::new(EXE).args(arguments).output().unwrap()
}

fn text(output: &Output) -> (String, String) {
    (String::from_utf8_lossy(&output.stdout).into_owned(), String::from_utf8_lossy(&output.stderr).into_owned())
}

fn lines_with(output: &str, prefix: &str) -> Vec<String> {
    output.lines().filter(|line| line.starts_with(prefix)).map(str::to_owned).collect()
}

#[test]
fn a_model_is_printed_and_it_satisfies_the_formula() {
    let scratch = Scratch::new("sat");
    let input = scratch.file("f.cnf", SAT);
    let output = motor(&["solve", &input, "--no-verify"]);
    let (stdout, stderr) = text(&output);
    assert_eq!(output.status.code(), Some(10), "{stderr}");
    assert_eq!(lines_with(&stdout, "s "), ["s SATISFIABLE"]);
    let model: Vec<i32> = lines_with(&stdout, "v ")[0][2..].split_whitespace().map(|t| t.parse().unwrap()).collect();
    assert_eq!(model.last(), Some(&0));
    assert!(verify_model(&parse_dimacs(SAT).unwrap(), &model[..model.len() - 1]));
    assert!(stderr.contains("c verificado: el modelo"));
}

#[test]
fn an_unsat_without_a_certificate_says_so() {
    let scratch = Scratch::new("unchecked");
    let output = motor(&["solve", &scratch.file("f.cnf", UNSAT), "--no-verify"]);
    let (stdout, stderr) = text(&output);
    assert_eq!(output.status.code(), Some(20), "{stderr}");
    assert_eq!(lines_with(&stdout, "s "), ["s UNSATISFIABLE"]);
    assert!(stderr.contains("AVISO: UNSAT sin certificado"));
}

#[test]
fn without_the_verifier_nothing_is_searched_and_the_error_explains_why() {
    let scratch = Scratch::new("missing");
    let output = Command::new(EXE).args(["solve", &scratch.file("f.cnf", UNSAT)]).env("PATH", "").output().unwrap();
    let (stdout, stderr) = text(&output);
    assert_eq!(output.status.code(), Some(1), "{stdout}");
    assert!(stdout.is_empty(), "{stdout}");
    assert!(stderr.contains("--no-verify"), "{stderr}");
}

#[test]
fn recubing_only_runs_when_the_missing_certificate_is_accepted() {
    let scratch = Scratch::new("recube");
    let input = scratch.file("f.cnf", UNSAT);
    let refused = motor(&["solve", &input, "--workers", "2", "--recube-conflicts", "100"]);
    assert_eq!(refused.status.code(), Some(1));
    assert!(text(&refused).0.is_empty());
    let accepted = motor(&["solve", &input, "--workers", "2", "--recube-conflicts", "100", "--no-verify"]);
    assert_eq!(accepted.status.code(), Some(20));
}

#[test]
fn bad_input_and_bad_commands_are_errors() {
    let scratch = Scratch::new("bad");
    assert_eq!(motor(&["solve", &scratch.file("f.cnf", "1 x 0\n"), "--no-verify"]).status.code(), Some(1));
    assert_eq!(motor(&["solve", &scratch.path("missing.cnf"), "--no-verify"]).status.code(), Some(1));
    assert_eq!(motor(&["solve"]).status.code(), Some(1));
    assert_eq!(motor(&["gen", "nada"]).status.code(), Some(1));
    assert_eq!(motor(&[]).status.code(), Some(1));
}

// Sin simetria ni paridad el palomar crece exponencialmente: no termina en un segundo.
#[test]
fn the_time_limit_gives_no_verdict() {
    let scratch = Scratch::new("timeout");
    let input = scratch.path("php.cnf");
    let (pigeons, holes) = (16_i32, 15_i32);
    let variable = |pigeon: i32, hole: i32| pigeon * holes + hole + 1;
    let mut cnf = format!("p cnf {} 0\n", pigeons * holes);
    for pigeon in 0..pigeons {
        cnf += &(0..holes).map(|hole| format!("{} ", variable(pigeon, hole))).collect::<String>();
        cnf += "0\n";
    }
    for hole in 0..holes {
        for first in 0..pigeons {
            for second in first + 1..pigeons {
                cnf += &format!("-{} -{} 0\n", variable(first, hole), variable(second, hole));
            }
        }
    }
    std::fs::write(&input, cnf).unwrap();
    let output = motor(&["solve", &input, "--no-verify", "--no-symmetry", "--no-parity", "--timeout", "1"]);
    assert_eq!(output.status.code(), Some(124));
    assert_eq!(lines_with(&text(&output).0, "s "), ["s UNKNOWN"]);
}

#[test]
fn generators_write_the_formula_the_solver_reads() {
    let scratch = Scratch::new("gen");
    let (ramsey, erdos) = (scratch.path("r.cnf"), scratch.path("e.cnf"));
    assert_eq!(motor(&["gen", "ramsey", "5", &ramsey, "--cliques", "3", "3"]).status.code(), Some(0));
    assert_eq!(motor(&["solve", &ramsey, "--no-verify"]).status.code(), Some(10));
    assert_eq!(motor(&["gen", "erdos", "11", &erdos, "--c", "1"]).status.code(), Some(0));
    assert_eq!(motor(&["solve", &erdos, "--no-verify"]).status.code(), Some(10));
}

#[test]
#[ignore = "requiere veripb"]
fn a_verified_unsat_keeps_only_the_proof_that_was_asked_for() {
    let scratch = Scratch::new("verified");
    let input = scratch.file("f.cnf", UNSAT);
    let output = motor(&["solve", &input]);
    let (stdout, stderr) = text(&output);
    assert_eq!(output.status.code(), Some(20), "{stderr}");
    assert_eq!(lines_with(&stdout, "s "), ["s UNSATISFIABLE"]);
    assert!(stderr.contains("c verificado: VeriPB"), "{stderr}");
    let proof = scratch.path("kept.pbp");
    assert_eq!(motor(&["solve", &input, "--proof", &proof]).status.code(), Some(20));
    assert!(std::fs::metadata(&proof).unwrap().len() > 0);
}

// La prueba de una formula no vale para otra: VeriPB tiene que rechazarla.
#[test]
#[ignore = "requiere veripb"]
fn a_proof_does_not_certify_a_different_formula() {
    let scratch = Scratch::new("rejected");
    let (unsat, sat) = (scratch.file("u.cnf", UNSAT), scratch.file("s.cnf", SAT));
    let proof = scratch.path("u.pbp");
    assert_eq!(motor(&["solve", &unsat, "--proof", &proof]).status.code(), Some(20));
    let accepted = check_unsat_proof(unsat.as_ref(), proof.as_ref(), None).unwrap();
    assert_eq!(accepted, ProofCheck::Verified);
    let verdict = check_unsat_proof(sat.as_ref(), proof.as_ref(), None).unwrap();
    assert!(matches!(verdict, ProofCheck::Rejected(_)), "{verdict:?}");
    let garbage = scratch.file("g.pbp", "pseudo-Boolean proof version 3.0\nconclusion UNSAT : 1\nend pseudo-Boolean proof\n");
    let verdict = check_unsat_proof(sat.as_ref(), garbage.as_ref(), None).unwrap();
    assert!(matches!(verdict, ProofCheck::Rejected(_)), "{verdict:?}");
}

// Simetria, paridad y busqueda juntas: los numeros de Ramsey conocidos, a uno y otro lado.
#[test]
#[ignore = "requiere veripb"]
fn known_ramsey_numbers_come_out_verified() {
    let scratch = Scratch::new("ramsey");
    let cases = [(3, 3, 5, 10), (3, 3, 6, 20), (3, 4, 8, 10), (3, 4, 9, 20), (3, 5, 13, 10), (3, 5, 14, 20), (4, 4, 17, 10), (4, 4, 18, 20)];
    for (s, t, n, code) in cases {
        {
            let input = scratch.path(&format!("r{s}{t}_{n}.cnf"));
            let (ss, ts, ns) = (s.to_string(), t.to_string(), n.to_string());
            assert_eq!(motor(&["gen", "ramsey", &ns, &input, "--cliques", &ss, &ts]).status.code(), Some(0));
            let output = motor(&["solve", &input, "--timeout", "300"]);
            assert_eq!(output.status.code(), Some(code), "R({s},{t}) en K{n}: {}", text(&output).1);
        }
    }
}

// La traza comprimida tiene que quedar completa antes de pasarsela a VeriPB.
#[test]
#[ignore = "requiere veripb"]
fn a_compressed_proof_comes_out_verified() {
    let scratch = Scratch::new("zst");
    let input = scratch.file("f.cnf", "p cnf 2 4\n1 2 0\n-1 2 0\n1 -2 0\n-1 -2 0\n");
    let proof = scratch.path("f.pbp.zst");
    let output = motor(&["solve", &input, "--proof", &proof]);
    assert_eq!(output.status.code(), Some(20), "{}", text(&output).1);
    assert_eq!(std::fs::read(&proof).unwrap()[..4], [0x28, 0xb5, 0x2f, 0xfd], "no es un marco zstd");
}

// El banco no ejecuta VeriPB: su UNSAT no se puede leer como certificado.
#[test]
fn the_trace_bench_warns_that_it_does_not_check() {
    let scratch = Scratch::new("trace");
    let input = scratch.file("f.cnf", "p cnf 1 2\n1 0\n-1 0\n");
    let output = motor(&["trace", &input, "--route", "check"]);
    assert_eq!(output.status.code(), Some(20));
    assert!(text(&output).1.contains("sin comprobar"), "{}", text(&output).1);
}

#[test]
fn the_reference_bench_runs_from_the_single_command() {
    let output = motor(&["bench", "--budget", "1000", "--seeds", "1", "--jobs", "1", "--only", "3,3,5"]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output).1);
}

#[test]
fn the_reference_bench_runs_from_any_folder() {
    let scratch = Scratch::new("bench_folder");
    let output = Command::new(EXE)
        .args(["bench", "--budget", "1000", "--seeds", "1", "--jobs", "1", "--only", "3,3,5"])
        .current_dir(&scratch.0)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0), "{}", text(&output).1);
}

#[cfg(not(feature = "gpu"))]
#[test]
fn the_gpu_search_says_how_to_enable_it() {
    let output = motor(&["k43", "--self-test"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(text(&output).1.contains("--features gpu"), "{}", text(&output).1);
}

// TensorRT real, con el SDK fuera del PATH: el binario tiene que encontrar sus DLL solo.
#[cfg(feature = "neuro-trt")]
#[test]
#[ignore = "requiere GPU y el SDK de TensorRT"]
fn the_policy_loads_without_the_sdk_on_the_path() {
    let search = std::env::var_os("PATH").unwrap_or_default();
    let kept = std::env::split_paths(&search).filter(|entry| !entry.to_string_lossy().to_lowercase().contains("tensorrt"));
    let model = concat!(env!("CARGO_MANIFEST_DIR"), "/neuro_ramsey_policy.onnx");
    let output = Command::new(EXE)
        .args(["k43", "--neuro-model", model, "--lookahead", "--explicit-trail", "--epochs", "1"])
        .env("PATH", std::env::join_paths(kept).unwrap())
        .output()
        .unwrap();
    let (stdout, stderr) = text(&output);
    assert_eq!(output.status.code(), Some(0), "{stderr}");
    assert!(stdout.contains("c neuro-trt-loaded") && stdout.contains("fallbacks=0"), "{stdout}");
}

// Dos cadenas XOR sobre las mismas entradas, en distinto orden, con paridades opuestas: no hay
// solucion y es la familia que obliga a certificar la capa de paridad.
#[test]
#[ignore = "requiere veripb"]
fn an_inconsistent_parity_system_comes_out_verified() {
    let scratch = Scratch::new("parity");
    let inputs = 24_i32;
    let mut fresh = inputs;
    let mut clauses: Vec<[i32; 3]> = Vec::new();
    let mut xor = |a: i32, b: i32, c: i32| {
        // a + b + c = 0: se prohiben las cuatro asignaciones de paridad impar.
        clauses.extend([[a, b, -c], [a, -b, c], [-a, b, c], [-a, -b, -c]]);
    };
    let mut ends = Vec::new();
    for stride in [1, 5] {
        let mut sum = 1;
        for step in 1..inputs {
            fresh += 1;
            xor(sum, (step * stride) % inputs + 1, fresh);
            sum = fresh;
        }
        ends.push(sum);
    }
    // Las dos sumas son la misma variable logica; se les exige ser distintas.
    let cnf: String = clauses.iter().map(|c| format!("{} {} {} 0\n", c[0], c[1], c[2])).collect();
    let cnf = format!("{cnf}{} {} 0\n-{} -{} 0\n", ends[0], ends[1], ends[0], ends[1]);
    let input = scratch.file("f.cnf", &format!("p cnf {fresh} {}\n{cnf}", clauses.len() + 2));
    let output = motor(&["solve", &input, "--timeout", "120"]);
    assert_eq!(output.status.code(), Some(20), "{}", text(&output).1);
}

fn random_formula(next: &mut impl FnMut(u64) -> u64) -> (i32, Vec<Vec<i32>>) {
    let variables = 1 + next(10) as i32;
    let clauses = 1 + next(45);
    let mut formula: Vec<Vec<i32>> = (0..clauses).map(|_| {
        (0..next(4)).map(|_| (1 + next(variables as u64) as i32) * if next(2) == 0 { 1 } else { -1 }).collect()
    }).collect();
    // Casi todas tendrian una clausula vacia: se dejan en una de cada veinte formulas.
    if next(20) != 0 {
        formula.retain(|clause: &Vec<i32>| !clause.is_empty());
    }
    (variables, formula)
}

// Circuito pequeno de puertas AND, XOR e ITE que a menudo repiten entradas, con unidades,
// equivalencias y salidas que se exigen distintas: lo que el cierre por congruencia aprovecha.
fn gate_formula(next: &mut impl FnMut(u64) -> u64) -> (i32, Vec<Vec<i32>>) {
    let mut formula: Vec<Vec<i32>> = Vec::new();
    let mut variables = 2 + next(4) as i32;
    let mut previous: Vec<i32> = Vec::new();
    let mut outputs: Vec<i32> = Vec::new();
    while variables < 13 {
        let out = variables + 1;
        let kind = next(4);
        let arity = if kind == 0 { 3 } else if kind == 3 { 3 } else { 2 };
        let inputs: Vec<i32> = if previous.len() >= arity && next(2) == 0 {
            previous[..arity].to_vec()
        } else {
            (0..arity).map(|_| (1 + next(variables as u64) as i32) * if next(3) == 0 { -1 } else { 1 }).collect()
        };
        match (kind, arity) {
            (0, 3) => {
                let (c, t, e) = (inputs[0], inputs[1], inputs[2]);
                formula.extend([vec![-out, -c, t], vec![out, -c, -t], vec![-out, c, e], vec![out, c, -e]]);
            }
            (1, 2) | (2, 2) => {
                let (a, b) = (inputs[0], inputs[1]);
                formula.extend([vec![-out, a], vec![-out, b], vec![out, -a, -b]]);
            }
            _ => {
                // Todas las asignaciones de paridad distinta a la de la salida quedan prohibidas.
                let mut support = inputs.clone();
                support.push(-out);
                for bits in 0..1_u32 << support.len() {
                    if bits.count_ones() % 2 == 0 {
                        formula.push(support.iter().enumerate().map(|(bit, &literal)| {
                            if bits >> bit & 1 == 1 { literal } else { -literal }
                        }).collect());
                    }
                }
            }
        }
        previous = inputs;
        outputs.push(out);
        variables = out;
    }
    for _ in 0..next(4) {
        let pick = |next: &mut dyn FnMut(u64) -> u64| (1 + next(variables as u64) as i32) * if next(2) == 0 { -1 } else { 1 };
        match next(3) {
            0 => formula.push(vec![pick(next)]),
            1 => {
                let (a, b) = (pick(next), pick(next));
                formula.extend([vec![-a, b], vec![a, -b]]);
            }
            _ => {
                let (a, b) = (outputs[next(outputs.len() as u64) as usize], outputs[next(outputs.len() as u64) as usize]);
                formula.extend([vec![a, b], vec![-a, -b]]);
            }
        }
    }
    (variables, formula)
}

#[test]
fn gate_circuits_agree_with_brute_force_in_every_configuration() {
    let mut next = generator(0x2545_f491_4f6c_dd1d);
    let (mut sat, mut unsat) = (0, 0);
    for round in 0..400 {
        let (variables, formula) = gate_formula(&mut next);
        let expected = brute_force(variables, &formula);
        if expected { sat += 1 } else { unsat += 1 }
        for (symmetry, parity) in [(true, true), (false, false)] {
            let config = HybridConfig { symmetry, parity, ..HybridConfig::default() };
            let serial = solve_hybrid(&formula, config).model;
            assert_eq!(serial.is_some(), expected, "ronda {round}: {formula:?}");
            assert!(serial.is_none_or(|model| verify_model(&formula, &model)), "ronda {round}: modelo falso");
            let swarm = SwarmConfig { workers: 2, ..SwarmConfig::default() };
            let parallel = solve_hybrid_parallel(&formula, config, swarm).model;
            assert_eq!(parallel.is_some(), expected, "ronda {round} enjambre: {formula:?}");
        }
    }
    assert!(sat > 30 && unsat > 30, "el banco no cubre ambos veredictos: {sat} SAT, {unsat} UNSAT");
}

// Sin cierre, una refutacion por congruencia tendria que buscarse; con el, la traza la da hecha.
#[test]
fn two_copies_of_a_circuit_with_different_outputs_need_no_search() {
    assert_miter_needs_no_search("miter", 0);
}

// Cada paso de la cadena gasta una pasada; con menos rondas que pasos nunca se llegaba a las puertas.
#[test]
fn a_chain_of_constants_does_not_hide_the_gates() {
    assert_miter_needs_no_search("miter_chain", 64);
}

fn assert_miter_needs_no_search(name: &str, chain: i32) {
    let scratch = Scratch::new(name);
    let mut clauses: Vec<Vec<i32>> = Vec::new();
    let inputs = 24;
    let mut ends = [0; 2];
    let mut fresh = inputs;
    for _ in 0..chain {
        fresh += 1;
        clauses.push(if fresh == inputs + 1 { vec![fresh] } else { vec![-(fresh - 1), fresh] });
    }
    for end in &mut ends {
        let mut sum = 1;
        for step in 2..=inputs {
            fresh += 1;
            let (out, a, b) = (fresh, sum, step);
            if step % 2 == 0 {
                clauses.extend([vec![-out, a], vec![-out, b], vec![out, -a, -b]]);
            } else {
                clauses.extend([vec![-out, a, b], vec![-out, -a, -b], vec![out, -a, b], vec![out, a, -b]]);
            }
            sum = out;
        }
        *end = sum;
    }
    clauses.extend([vec![ends[0], ends[1]], vec![-ends[0], -ends[1]]]);
    let cnf: String = clauses.iter().map(|clause| {
        clause.iter().map(|literal| format!("{literal} ")).collect::<String>() + "0\n"
    }).collect();
    let input = scratch.file("f.cnf", &format!("p cnf {fresh} {}\n{cnf}", clauses.len()));
    let output = motor(&["solve", &input, "--json", "--no-verify"]);
    assert_eq!(output.status.code(), Some(20), "{}", text(&output).1);
    assert!(text(&output).0.contains("\"conflicts\":1,") || text(&output).0.contains("\"conflicts\":0,"), "{}", text(&output).0);
}

// Cada lema de la capa pasa por el verificador, en serie y en el enjambre.
#[test]
#[ignore = "requiere veripb"]
fn gate_circuits_are_certified_serial_and_in_the_swarm() {
    let scratch = Scratch::new("gatefuzz");
    let mut next = generator(0x9fb2_1c65_1e98_df25);
    let (mut sat, mut unsat) = (0, 0);
    for round in 0..300 {
        let (variables, formula) = gate_formula(&mut next);
        let cnf: String = formula.iter().map(|clause| {
            clause.iter().map(|literal| format!("{literal} ")).collect::<String>() + "0\n"
        }).collect();
        let expected = brute_force(variables, &formula);
        let input = scratch.file("f.cnf", &format!("p cnf {variables} {}\n{cnf}", formula.len()));
        let mut arguments = vec!["solve", input.as_str()];
        if round % 3 == 0 {
            arguments.extend(["--workers", "2"]);
        }
        let output = motor(&arguments);
        assert_eq!(output.status.code(), Some(if expected { 10 } else { 20 }), "ronda {round}:\n{cnf}\n{}", text(&output).1);
        if expected { sat += 1 } else { unsat += 1 }
    }
    assert!(sat > 30 && unsat > 30, "el banco no cubre ambos veredictos: {sat} SAT, {unsat} UNSAT");
}

fn brute_force(variables: i32, formula: &[Vec<i32>]) -> bool {
    (0..1_u32 << variables).any(|bits| formula.iter().all(|clause| {
        clause.iter().any(|&literal| (bits >> (literal.unsigned_abs() - 1) & 1 == 1) == (literal > 0))
    }))
}

fn generator(seed: u64) -> impl FnMut(u64) -> u64 {
    let mut state = seed;
    move |bound| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    }
}

// Cada combinacion de capas y el enjambre, sin traza: el veredicto tiene que ser el real.
#[test]
fn every_configuration_agrees_with_brute_force() {
    let mut next = generator(0x9e37_79b9_7f4a_7c15);
    let (mut sat, mut unsat) = (0, 0);
    for round in 0..400 {
        let (variables, formula) = random_formula(&mut next);
        let expected = brute_force(variables, &formula);
        if expected { sat += 1 } else { unsat += 1 }
        for (symmetry, parity) in [(true, true), (true, false), (false, true), (false, false)] {
            let config = HybridConfig { symmetry, parity, ..HybridConfig::default() };
            let serial = solve_hybrid(&formula, config).model;
            assert_eq!(serial.is_some(), expected, "ronda {round} simetria={symmetry} paridad={parity}: {formula:?}");
            assert!(serial.is_none_or(|model| verify_model(&formula, &model)), "ronda {round}: modelo falso");
            let swarm = SwarmConfig { workers: 2, ..SwarmConfig::default() };
            let parallel = solve_hybrid_parallel(&formula, config, swarm).model;
            assert_eq!(parallel.is_some(), expected, "ronda {round} enjambre: {formula:?}");
            assert!(parallel.is_none_or(|model| verify_model(&formula, &model)), "ronda {round}: modelo falso");
        }
    }
    assert!(sat > 30 && unsat > 30, "el banco no cubre ambos veredictos: {sat} SAT, {unsat} UNSAT");
}

// Formulas cerradas bajo una permutacion con ciclos de 2 y de 3: los eslabones de empate de
// la ruptura de simetria, con y sin pareja inversa. Cada UNSAT pasa por VeriPB y todo SAT se
// compara con la fuerza bruta.
#[test]
#[ignore = "requiere veripb"]
fn symmetric_formulas_agree_with_brute_force_and_every_unsat_is_certified() {
    const IMAGE: [i32; 10] = [2, 3, 1, 5, 6, 4, 8, 7, 10, 9];
    let scratch = Scratch::new("orbits");
    let mut next = generator(0x71c7_1c71_c71c_71c7);
    let (mut sat, mut unsat) = (0, 0);
    for round in 0..300 {
        let mut formula: Vec<Vec<i32>> = Vec::new();
        for _ in 0..1 + next(14) {
            let mut clause: Vec<i32> = (0..1 + next(3)).map(|_| {
                (1 + next(10) as i32) * if next(2) == 0 { 1 } else { -1 }
            }).collect();
            // Seis imagenes recorren la orbita: el orden de la permutacion es 6.
            for _ in 0..6 {
                formula.push(clause.clone());
                clause = clause.iter().map(|&literal| IMAGE[literal.unsigned_abs() as usize - 1] * literal.signum()).collect();
            }
        }
        let cnf: String = formula.iter().map(|clause| {
            clause.iter().map(|literal| format!("{literal} ")).collect::<String>() + "0
"
        }).collect();
        let expected = brute_force(10, &formula);
        let input = scratch.file("f.cnf", &format!("p cnf 10 {}
{cnf}", formula.len()));
        let output = motor(&["solve", &input]);
        let code = if expected { 10 } else { 20 };
        assert_eq!(output.status.code(), Some(code), "ronda {round}:
{cnf}
{}", text(&output).1);
        if expected { sat += 1 } else { unsat += 1 }
    }
    assert!(sat > 30 && unsat > 30, "el banco no cubre ambos veredictos: {sat} SAT, {unsat} UNSAT");
}

// El enjambre por el comando real: la traza de todos los hilos tiene que pasar por VeriPB.
#[test]
#[ignore = "requiere veripb"]
fn the_swarm_agrees_with_brute_force_and_every_unsat_is_certified() {
    let scratch = Scratch::new("swarmfuzz");
    let mut next = generator(0x5851_f42d_4c95_7f2d);
    let (mut sat, mut unsat) = (0, 0);
    for round in 0..200 {
        let (variables, formula) = random_formula(&mut next);
        let cnf: String = formula.iter().map(|clause| {
            clause.iter().map(|literal| format!("{literal} ")).collect::<String>() + "0
"
        }).collect();
        let expected = brute_force(variables, &formula);
        let input = scratch.file("f.cnf", &format!("p cnf {variables} {}
{cnf}", formula.len()));
        let workers = (2 + round % 3).to_string();
        let mut arguments = vec!["solve", input.as_str(), "--workers", workers.as_str()];
        if round % 4 == 0 {
            arguments.push("--no-share");
        }
        let output = motor(&arguments);
        let code = if expected { 10 } else { 20 };
        assert_eq!(output.status.code(), Some(code), "ronda {round}:
{cnf}
{}", text(&output).1);
        if expected { sat += 1 } else { unsat += 1 }
    }
    assert!(sat > 30 && unsat > 30, "el banco no cubre ambos veredictos: {sat} SAT, {unsat} UNSAT");
}

// Formulas con busqueda real, con y sin intercambio de clausulas, con traza plana y comprimida.
#[test]
#[ignore = "requiere veripb"]
fn hard_formulas_through_the_swarm_come_out_verified() {
    let scratch = Scratch::new("swarmhard");
    let mut next = generator(0x2127_599b_f432_5c37);
    for round in 0..6 {
        let variables = 120 + 10 * round as u64;
        let clauses = variables * 43 / 10 + 40;
        let cnf: String = (0..clauses).map(|_| {
            let mut used: Vec<u64> = Vec::new();
            while used.len() < 3 {
                let variable = 1 + next(variables);
                if !used.contains(&variable) {
                    used.push(variable);
                }
            }
            used.iter().map(|&variable| format!("{}{variable} ", if next(2) == 0 { "-" } else { "" })).collect::<String>() + "0
"
        }).collect();
        let input = scratch.file("f.cnf", &format!("p cnf {variables} {clauses}
{cnf}"));
        let proof = scratch.path(if round % 2 == 0 { "p.pbp" } else { "p.pbp.zst" });
        let mut arguments = vec!["solve", input.as_str(), "--workers", "4", "--proof", proof.as_str()];
        if round == 5 {
            arguments.push("--no-share");
        }
        let output = motor(&arguments);
        assert!(matches!(output.status.code(), Some(10 | 20)), "ronda {round}: {}", text(&output).1);
        if output.status.code() == Some(20) {
            assert!(text(&output).1.contains("verificado"), "{}", text(&output).1);
        }
    }
    let ramsey = scratch.path("r.cnf");
    assert_eq!(motor(&["gen", "ramsey", "14", &ramsey, "--cliques", "3", "5"]).status.code(), Some(0));
    assert_eq!(motor(&["solve", &ramsey, "--workers", "4"]).status.code(), Some(20));
}

// El mismo banco por el comando real: cada UNSAT tiene que pasar por VeriPB.
#[test]
#[ignore = "requiere veripb"]
fn random_formulas_agree_with_brute_force_and_every_unsat_is_certified() {
    let scratch = Scratch::new("fuzz");
    let mut next = generator(0x2545_f491_4f6c_dd1d);
    let (mut sat, mut unsat) = (0, 0);
    for round in 0..300 {
        let (variables, formula) = random_formula(&mut next);
        let cnf: String = formula.iter().map(|clause| {
            clause.iter().map(|literal| format!("{literal} ")).collect::<String>() + "0\n"
        }).collect();
        let expected = brute_force(variables, &formula);
        let input = scratch.file("f.cnf", &format!("p cnf {variables} {}\n{cnf}", formula.len()));
        let output = motor(&["solve", &input]);
        let code = if expected { 10 } else { 20 };
        assert_eq!(output.status.code(), Some(code), "ronda {round}:\n{cnf}\n{}", text(&output).1);
        if expected { sat += 1 } else { unsat += 1 }
    }
    assert!(sat > 30 && unsat > 30, "el banco no cubre ambos veredictos: {sat} SAT, {unsat} UNSAT");
}

#[test]
fn json_output_is_one_object_with_the_same_exit_code() {
    let scratch = Scratch::new("json");
    let sat = motor(&["solve", &scratch.file("s.cnf", SAT), "--no-verify", "--json"]);
    let (stdout, _) = text(&sat);
    assert_eq!(sat.status.code(), Some(10));
    assert_eq!(stdout.trim().lines().count(), 1, "{stdout}");
    assert!(stdout.starts_with("{\"status\":\"SAT\",\"model\":["), "{stdout}");
    let unsat = motor(&["solve", &scratch.file("u.cnf", UNSAT), "--no-verify", "--json"]);
    assert_eq!(unsat.status.code(), Some(20));
    assert!(text(&unsat).0.contains("\"proof_checked\":false"));
    let opb = motor(&["solve", &scratch.file("o.opb", "min: +1 x1 +1 x2 ;\n+1 x1 +1 x2 >= 1 ;\n"), "--no-verify", "--json"]);
    assert_eq!(opb.status.code(), Some(30));
    assert_eq!(text(&opb).0.trim().lines().count(), 1, "{}", text(&opb).0);
    assert!(text(&opb).0.contains("\"cost\":1"));
    let missing = motor(&["solve", &scratch.path("nada.cnf"), "--json"]);
    assert_eq!(missing.status.code(), Some(1));
    assert!(text(&missing).0.starts_with("{\"status\":\"ERROR\",\"error\":\""), "{}", text(&missing).0);
}

#[test]
fn the_flags_the_opb_route_cannot_honour_are_refused() {
    let scratch = Scratch::new("opb_flags");
    let input = scratch.file("o.opb", "+1 x1 >= 1 ;\n");
    for flag in [&["--csv", "x"][..], &["--preprocess-only"], &["--workers", "2", "--use-gpu"]] {
        let mut arguments = vec!["solve", &input, "--no-verify"];
        arguments.extend_from_slice(flag);
        let output = motor(&arguments);
        assert_eq!(output.status.code(), Some(1), "{flag:?}");
        assert!(text(&output).1.contains("no admitida"), "{}", text(&output).1);
    }
    let output = motor(&["solve", &input, "--no-verify", "--workers", "2"]);
    assert_eq!(output.status.code(), Some(10), "{}", text(&output).1);
}
