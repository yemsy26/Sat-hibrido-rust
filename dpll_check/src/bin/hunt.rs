// Fase 17: lanzador. Resuelve un DIMACS con el motor hibrido, en serie o con el enjambre.
//
//   hunt <formula.cnf> [opciones]
//     --workers K         enjambre de K workers; sin esta opcion, un solo hilo
//     --cubes C           cubos por worker (4)
//     --no-share          enjambre sin intercambio de clausulas
//     --no-symmetry       sin ruptura de simetrias
//     --no-parity         sin capa de paridad
//     --flips F           flips por variable del oraculo (16)
//     --interval I        conflictos entre re-igniciones (500); 0 apaga el oraculo
//     --seed S            semilla del oraculo
//     --proof F.pbp       traza VeriPB 3.0; solo en serie: el enjambre no emite traza
//     --proof-cap-gib G   tope de la traza; superado, la ejecucion sigue SIN certificado
//     --model F           escribe el modelo si la formula es satisfacible
//     --preprocess-only   solo las capas de simetria y paridad, sin busqueda
//     --timeout S         aborta a los S segundos (estado TIMEOUT, codigo 124)
//     --heartbeat S       linea de progreso en stderr cada S segundos
//     --csv ETIQUETA      una fila CSV en la salida estandar (`hunt --header` da la cabecera)
//
// El motor no tiene puntos de control ni contadores visibles mientras busca: el unico
// progreso observable desde fuera es el crecimiento de la traza.
#![forbid(unsafe_code)]
#![deny(warnings)]

#[allow(dead_code)]
#[path = "../hybrid.rs"]
mod hybrid;

use hybrid::cdcl::SwarmConfig;
use hybrid::proof::PbProof;
use hybrid::{solve_hybrid, solve_hybrid_parallel, solve_hybrid_with_proof, verify_model, HybridConfig, SolveStats};
use std::fs::File;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

type Formula = Vec<Vec<i32>>;

const HEADER: &str = "label,mode,workers,cubes_per_worker,share,symmetry,parity,flips,interval,variables,clauses,status,total_ms,parse_ms,symmetry_generators,symmetry_clauses,symmetry_ms,parity_equations,parity_ms,search_ms,decisions,conflicts,restarts,deleted_clauses,reignitions,reignition_flips,reignition_ms,cubes_generated,cubes_started,cubes_unsat,cubes_cancelled,exports,imports,trace_bytes,trace_lines,trace_dominances,trace_deletions,trace_truncated";

struct Options {
    input: String,
    workers: Option<usize>,
    cubes: usize,
    share: bool,
    symmetry: bool,
    parity: bool,
    flips: usize,
    interval: usize,
    seed: Option<u64>,
    proof: Option<String>,
    cap: u64,
    model: Option<String>,
    preprocess_only: bool,
    timeout: Option<u64>,
    heartbeat: Option<u64>,
    csv: Option<String>,
}

fn parse_options(arguments: &[String]) -> Result<Options, String> {
    let defaults = HybridConfig::default();
    let mut options = Options {
        input: String::new(), workers: None, cubes: SwarmConfig::default().cubes_per_worker, share: true,
        symmetry: true, parity: true, flips: defaults.oracle.flips_per_variable,
        interval: defaults.reignition_interval, seed: None, proof: None, cap: u64::MAX, model: None,
        preprocess_only: false, timeout: None, heartbeat: None, csv: None,
    };
    let mut index = 0;
    while index < arguments.len() {
        let flag = arguments[index].as_str();
        index += 1;
        let mut value = |name: &str| -> Result<String, String> {
            let found = arguments.get(index).cloned().ok_or(format!("{name} necesita un valor"))?;
            index += 1;
            Ok(found)
        };
        let number = |text: String| text.parse::<u64>().map_err(|_| format!("Numero no valido: {text}"));
        match flag {
            "--workers" => options.workers = Some(number(value(flag)?)?.max(1) as usize),
            "--cubes" => options.cubes = number(value(flag)?)?.max(1) as usize,
            "--no-share" => options.share = false,
            "--no-symmetry" => options.symmetry = false,
            "--no-parity" => options.parity = false,
            "--flips" => options.flips = number(value(flag)?)? as usize,
            "--interval" => options.interval = number(value(flag)?)? as usize,
            "--seed" => options.seed = Some(number(value(flag)?)?),
            "--proof" => options.proof = Some(value(flag)?),
            "--proof-cap-gib" => options.cap = number(value(flag)?)?.saturating_mul(1 << 30),
            "--model" => options.model = Some(value(flag)?),
            "--preprocess-only" => options.preprocess_only = true,
            "--timeout" => options.timeout = Some(number(value(flag)?)?),
            "--heartbeat" => options.heartbeat = Some(number(value(flag)?)?.max(1)),
            "--csv" => options.csv = Some(value(flag)?),
            _ if flag.starts_with("--") => return Err(format!("Opcion desconocida: {flag}")),
            _ if options.input.is_empty() => options.input = flag.to_owned(),
            _ => return Err(format!("Sobra el argumento {flag}")),
        }
    }
    if options.input.is_empty() {
        return Err("Uso: hunt <formula.cnf> [opciones]".into());
    }
    if options.proof.is_some() && options.workers.is_some() {
        return Err("El enjambre no emite traza: --proof exige ejecutar en serie, sin --workers".into());
    }
    Ok(options)
}

// El numero de clausulas de la traza es el de clausulas leidas, no el de la cabecera.
fn parse_dimacs(text: &str) -> Result<Formula, String> {
    let mut formula = Formula::new();
    let mut clause = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('%') {
            break;
        }
        if line.starts_with('c') || line.starts_with('p') {
            continue;
        }
        for token in line.split_whitespace() {
            let literal: i32 = token.parse().map_err(|_| format!("Literal no valido: {token}"))?;
            if literal == i32::MIN {
                return Err(format!("Literal fuera de rango: {token}"));
            }
            if literal == 0 {
                formula.push(std::mem::take(&mut clause));
            } else {
                clause.push(literal);
            }
        }
    }
    if clause.is_empty() { Ok(formula) } else { Err("La ultima clausula no termina en 0".into()) }
}

// Lo que el vigilante puede ver de la traza mientras el motor trabaja.
#[derive(Default)]
struct Gauge {
    bytes: AtomicU64,
    lines: AtomicU64,
    truncated: AtomicBool,
}

// Cuenta lo que pasa hacia el fichero y deja de escribir al superar el tope: una traza
// truncada no certifica nada, pero la busqueda no se pierde por llenar el disco.
struct Meter<W: Write> {
    sink: W,
    gauge: Arc<Gauge>,
    cap: u64,
}
impl<W: Write> Write for Meter<W> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.gauge.truncated.load(Ordering::Relaxed) {
            return Ok(buffer.len());
        }
        if self.gauge.bytes.load(Ordering::Relaxed) + buffer.len() as u64 > self.cap {
            self.gauge.truncated.store(true, Ordering::Relaxed);
            eprintln!("c TRAZA TRUNCADA al llegar al tope: el resultado ya no llevara certificado");
            return Ok(buffer.len());
        }
        self.sink.write_all(buffer)?;
        self.gauge.bytes.fetch_add(buffer.len() as u64, Ordering::Relaxed);
        let lines = buffer.iter().filter(|&&byte| byte == b'\n').count();
        self.gauge.lines.fetch_add(lines as u64, Ordering::Relaxed);
        Ok(buffer.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.sink.flush()
    }
}

struct Outcome {
    model: Option<Vec<i32>>,
    symmetry: hybrid::SymmetryStats,
    parity: hybrid::ParityStats,
    search: SolveStats,
    search_ms: f64,
    // generados, empezados, UNSAT, cancelados
    cubes: [usize; 4],
    // dominancias y borrados de la traza
    trace: [u64; 2],
}

fn row(prefix: &str, status: &str, total_ms: f64, parse_ms: f64, outcome: Option<&Outcome>, gauge: &Gauge) -> String {
    let trace = format!(
        "{},{}", gauge.bytes.load(Ordering::Relaxed), gauge.lines.load(Ordering::Relaxed),
    );
    let truncated = gauge.truncated.load(Ordering::Relaxed);
    let Some(outcome) = outcome else {
        // Sin resultado: el motor no expone contadores a mitad de busqueda.
        return format!("{prefix},{status},{total_ms:.3},{parse_ms:.3},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,{trace},NA,NA,{truncated}");
    };
    let stats = &outcome.search;
    format!(
        "{prefix},{status},{total_ms:.3},{parse_ms:.3},{},{},{:.3},{},{:.3},{:.3},{},{},{},{},{},{},{:.3},{},{},{},{},{},{},{trace},{},{},{truncated}",
        outcome.symmetry.generators, outcome.symmetry.clauses, outcome.symmetry.ms,
        outcome.parity.equations, outcome.parity.ms, outcome.search_ms,
        stats.decisions, stats.conflicts, stats.restarts, stats.deleted_clauses,
        stats.reignitions, stats.reignition_flips, stats.reignition_ms,
        outcome.cubes[0], outcome.cubes[1], outcome.cubes[2], outcome.cubes[3],
        stats.exported_clauses, stats.imported_clauses, outcome.trace[0], outcome.trace[1],
    )
}

// Las dos capas previas a la busqueda, cada una sobre la formula original.
fn preprocess(
    options: &Options, formula: &Formula, prefix: &str, parse_ms: f64, started: Instant, gauge: &Arc<Gauge>,
) -> Result<i32, String> {
    let mut outcome = Outcome {
        model: None, symmetry: Default::default(), parity: Default::default(), search: SolveStats::default(),
        search_ms: 0.0, cubes: [0; 4], trace: [0, 0],
    };
    if options.symmetry {
        let start = Instant::now();
        let breaking = match &options.proof {
            Some(path) => {
                let failed = |error: io::Error| format!("{path}: {error}");
                let sink = Meter { sink: File::create(path).map_err(failed)?, gauge: Arc::clone(gauge), cap: options.cap };
                let mut proof = PbProof::new(sink, formula.len());
                let breaking = hybrid::symmetry::break_symmetries_with_proof(formula, Some(&mut proof));
                outcome.trace = [proof.dominances(), proof.deletions()];
                proof.finish().map_err(failed)?;
                breaking
            }
            None => hybrid::symmetry::break_symmetries(formula),
        };
        outcome.symmetry = hybrid::SymmetryStats {
            generators: breaking.generators.len(), clauses: breaking.clauses.len(),
            ms: start.elapsed().as_secs_f64() * 1000.0,
        };
    }
    if options.parity {
        let start = Instant::now();
        outcome.parity.equations = hybrid::parity::reduce_parity(formula).equations;
        outcome.parity.ms = start.elapsed().as_secs_f64() * 1000.0;
    }
    let total_ms = started.elapsed().as_secs_f64() * 1000.0;
    if options.csv.is_some() {
        println!("{}", row(prefix, "PREPROCESSED", total_ms, parse_ms, Some(&outcome), gauge));
    }
    eprintln!(
        "c simetria: {} generadores, {} predicados, {:.1} ms; paridad: {} ecuaciones, {:.1} ms; traza {} bytes",
        outcome.symmetry.generators, outcome.symmetry.clauses, outcome.symmetry.ms,
        outcome.parity.equations, outcome.parity.ms, gauge.bytes.load(Ordering::Relaxed),
    );
    println!("s UNKNOWN");
    Ok(0)
}

fn run(options: &Options) -> Result<i32, String> {
    let started = Instant::now();
    let text = std::fs::read_to_string(&options.input).map_err(|error| format!("{}: {error}", options.input))?;
    let formula = parse_dimacs(&text)?;
    drop(text);
    let parse_ms = started.elapsed().as_secs_f64() * 1000.0;
    let variables = formula.iter().flatten().map(|literal| literal.unsigned_abs()).max().unwrap_or(0);
    let mut config = HybridConfig { symmetry: options.symmetry, parity: options.parity, ..HybridConfig::default() };
    config.oracle.flips_per_variable = options.flips;
    config.reignition_interval = options.interval;
    if let Some(seed) = options.seed {
        config.oracle.seed = seed;
    }
    let mode = match (options.preprocess_only, options.workers, &options.proof) {
        (true, ..) => "preprocess",
        (false, Some(_), _) => "swarm",
        (false, None, Some(_)) => "serial_certified",
        (false, None, None) => "serial",
    };
    let prefix = format!(
        "{},{mode},{},{},{},{},{},{},{},{variables},{}",
        options.csv.as_deref().unwrap_or(""), options.workers.unwrap_or(1),
        if options.workers.is_some() { options.cubes } else { 0 },
        options.workers.is_some() && options.share, options.symmetry, options.parity,
        options.flips, options.interval, formula.len(),
    );
    eprintln!("c {}: {variables} variables, {} clausulas, modo {mode}", options.input, formula.len());

    // Vigilante: progreso y limite de tiempo, sin tocar el motor.
    let gauge = Arc::new(Gauge::default());
    {
        let (gauge, prefix, csv) = (Arc::clone(&gauge), prefix.clone(), options.csv.is_some());
        let (timeout, heartbeat) = (options.timeout, options.heartbeat);
        std::thread::spawn(move || {
            let mut next = heartbeat;
            loop {
                std::thread::sleep(Duration::from_millis(100));
                let elapsed = started.elapsed();
                if next.is_some_and(|due| elapsed.as_secs() >= due) {
                    eprintln!(
                        "c t={}s traza={} bytes, {} lineas{}", elapsed.as_secs(),
                        gauge.bytes.load(Ordering::Relaxed), gauge.lines.load(Ordering::Relaxed),
                        if gauge.truncated.load(Ordering::Relaxed) { " (TRUNCADA)" } else { "" },
                    );
                    next = next.zip(heartbeat).map(|(due, step)| due + step);
                }
                if timeout.is_some_and(|limit| elapsed.as_secs() >= limit) {
                    if csv {
                        println!("{}", row(&prefix, "TIMEOUT", elapsed.as_secs_f64() * 1000.0, parse_ms, None, &gauge));
                    }
                    println!("s UNKNOWN");
                    std::process::exit(124);
                }
            }
        });
    }

    if options.preprocess_only {
        return preprocess(options, &formula, &prefix, parse_ms, started, &gauge);
    }
    let outcome = if let Some(workers) = options.workers {
        let swarm = SwarmConfig {
            workers, cubes_per_worker: options.cubes, share_enabled: options.share, ..SwarmConfig::default()
        };
        let report = solve_hybrid_parallel(&formula, config, swarm);
        let stats = &report.swarm;
        Outcome {
            model: report.model, symmetry: report.symmetry, parity: report.parity, search: stats.total,
            search_ms: report.conquer_ms,
            cubes: [stats.cubes_generated, stats.cubes_started, stats.cubes_unsat, stats.cubes_cancelled],
            trace: [0, 0],
        }
    } else if let Some(path) = &options.proof {
        let failed = |error: io::Error| format!("{path}: {error}");
        let sink = Meter { sink: File::create(path).map_err(failed)?, gauge: Arc::clone(&gauge), cap: options.cap };
        let mut proof = PbProof::new(sink, formula.len());
        let report = solve_hybrid_with_proof(&formula, config, &mut proof);
        let trace = [proof.dominances(), proof.deletions()];
        proof.finish().map_err(failed)?;
        Outcome {
            model: report.model, symmetry: report.symmetry, parity: report.parity, search: report.cdcl,
            search_ms: report.cdcl_ms, cubes: [0; 4], trace,
        }
    } else {
        let report = solve_hybrid(&formula, config);
        Outcome {
            model: report.model, symmetry: report.symmetry, parity: report.parity, search: report.cdcl,
            search_ms: report.cdcl_ms, cubes: [0; 4], trace: [0, 0],
        }
    };
    let total_ms = started.elapsed().as_secs_f64() * 1000.0;
    let status = if outcome.model.is_some() { "SAT" } else { "UNSAT" };
    if options.csv.is_some() {
        println!("{}", row(&prefix, status, total_ms, parse_ms, Some(&outcome), &gauge));
    }
    eprintln!(
        "c {total_ms:.1} ms: simetria {} generadores, {} predicados ({:.1} ms); {} decisiones, {} conflictos, {} re-igniciones",
        outcome.symmetry.generators, outcome.symmetry.clauses, outcome.symmetry.ms,
        outcome.search.decisions, outcome.search.conflicts, outcome.search.reignitions,
    );
    if options.proof.is_some() {
        eprintln!(
            "c traza: {} bytes, {} lineas, {} dominancias, {} borrados{}",
            gauge.bytes.load(Ordering::Relaxed), gauge.lines.load(Ordering::Relaxed), outcome.trace[0], outcome.trace[1],
            if gauge.truncated.load(Ordering::Relaxed) { " -- TRUNCADA, sin valor de certificado" } else { "" },
        );
    }
    match &outcome.model {
        Some(model) => {
            // El motor ya lo comprobo; aqui se repite contra la formula leida del fichero.
            assert!(verify_model(&formula, model), "El modelo no satisface la formula");
            println!("s SATISFIABLE");
            if let Some(path) = &options.model {
                let line: Vec<String> = model.iter().map(i32::to_string).collect();
                std::fs::write(path, format!("v {} 0\n", line.join(" "))).map_err(|error| format!("{path}: {error}"))?;
            }
            Ok(10)
        }
        None => {
            println!("s UNSATISFIABLE");
            Ok(20)
        }
    }
}

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.first().is_some_and(|flag| flag == "--header") {
        println!("{HEADER}");
        return;
    }
    // Codigos de salida de la competicion SAT: 10 satisfacible, 20 insatisfacible.
    match parse_options(&arguments).and_then(|options| run(&options)) {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_row_has_the_columns_of_the_header() {
        let columns = HEADER.split(',').count();
        let gauge = Gauge::default();
        let prefix = "label,serial,1,0,false,true,true,16,500,10,20";
        let outcome = Outcome {
            model: None, symmetry: Default::default(), parity: Default::default(), search: SolveStats::default(),
            search_ms: 0.0, cubes: [0; 4], trace: [0, 0],
        };
        assert_eq!(row(prefix, "UNSAT", 1.0, 0.5, Some(&outcome), &gauge).split(',').count(), columns);
        assert_eq!(row(prefix, "TIMEOUT", 1.0, 0.5, None, &gauge).split(',').count(), columns);
    }

    #[test]
    fn dimacs_is_read_clause_by_clause() {
        let formula = parse_dimacs("c x
p cnf 3 3
1 -2 0 3
0

-1 0
%
9 0
").unwrap();
        assert_eq!(formula, vec![vec![1, -2], vec![3], vec![-1]]);
        assert!(parse_dimacs("1 2").is_err());
        assert!(parse_dimacs("1 x 0").is_err());
    }

    #[test]
    fn a_trace_past_its_cap_stops_being_written() {
        let gauge = Arc::new(Gauge::default());
        let mut meter = Meter { sink: Vec::new(), gauge: Arc::clone(&gauge), cap: 10 };
        meter.write_all(b"12345
").unwrap();
        meter.write_all(b"678901
").unwrap();
        meter.write_all(b"1
").unwrap();
        assert_eq!(meter.sink, b"12345
");
        assert!(gauge.truncated.load(Ordering::Relaxed));
        assert_eq!((gauge.bytes.load(Ordering::Relaxed), gauge.lines.load(Ordering::Relaxed)), (6, 1));
    }

    #[test]
    fn the_swarm_cannot_be_asked_for_a_trace() {
        let arguments: Vec<String> = ["f.cnf", "--workers", "4", "--proof", "p.pbp"].iter().map(|text| text.to_string()).collect();
        assert!(parse_options(&arguments).is_err());
        let options = parse_options(&arguments[..3]).unwrap();
        assert_eq!((options.workers, options.flips, options.interval), (Some(4), 16, 500));
    }
}
