// Fase 17: lanzador. Resuelve un DIMACS con el motor hibrido, en serie o con el enjambre.
//
//   hunt <formula.cnf> [opciones]
//     --workers K         enjambre de K workers; sin esta opcion, un solo hilo
//     --cubes C           cubos por worker (4)
//     --recube-conflicts C cede cubos tras C conflictos y los reenvía a la GPU (opcional)
//     --recube-split K     fija 2 o 3 variables VSIDS por split (4 u 8 hijos; defecto 3)
//     --hard-cube F        exporta un trail parcial tras >3 recubings en un linaje
//     --no-share          enjambre sin intercambio de clausulas
//     --no-symmetry       sin ruptura de simetrias
//     --no-parity         sin capa de paridad
//     --ramsey-symm N     la formula es de Ramsey sobre K_N (gen_ramsey): se comprueba que es
//                         simetrica en los vertices y se le anaden las clausulas que los
//                         ordenan (ramsey_symmetry.rs). No admite --proof
//     --flips F           flips por variable del oraculo (16)
//     --interval I        conflictos entre re-igniciones (500); 0 apaga el oraculo
//     --seed S            semilla del oraculo
//     --proof F.pbp       traza VeriPB 3.0; solo en serie: el enjambre no emite traza
//     --proof-cap-gib G   tope de la traza; superado, la ejecucion sigue SIN certificado
//     --model F           escribe el modelo si la formula es satisfacible
//     --deep-trail F      exporta records de asignacion parcial; exige --workers
//     --preprocess-only   solo las capas de simetria y paridad, sin busqueda
//     --timeout S         aborta a los S segundos (estado TIMEOUT, codigo 124)
//     --heartbeat S       linea de progreso en stderr cada S segundos
//     --csv ETIQUETA      una fila CSV en la salida estandar (`hunt --header` da la cabecera)
//     --use-gpu           filtra cubos con propagacion unitaria CUDA antes del enjambre;
//                         exige --workers y compilar con --features gpu.
//                         Los cubos Unknown conservan la busqueda CDCL existente.
//
// El motor no tiene puntos de control. En serie, el unico progreso observable desde fuera
// es el crecimiento de la traza. El enjambre publica sus conflictos y cubos cerrados: con
// --heartbeat o --timeout sale una linea `c status:` en cada latido y otra al agotar el limite.
// `deny` y no `forbid`: lanzar un nucleo CUDA es una llamada `unsafe`, y gpu_dispatcher.rs
// la permite en una sola funcion. En el resto del binario sigue siendo un error.
#![deny(unsafe_code)]
#![deny(warnings)]

#[allow(dead_code)]
#[path = "../hybrid.rs"]
mod hybrid;
// Mejora 2: ruptura de simetrias conocidas, fuera del motor. Solo cambia la formula que
// se le entrega, y solo si se pide.
#[allow(dead_code)]
#[path = "../symmetry_breaker.rs"]
mod symmetry_breaker;
#[path = "../ramsey_symmetry.rs"]
mod ramsey_symmetry;
// Mejora 3: puente con la GPU, fuera del motor. Solo se usa si se pide.
#[allow(dead_code)]
#[path = "../gpu_dispatcher.rs"]
mod gpu_dispatcher;

use hybrid::cdcl::{SwarmConfig, SwarmProgress};
use hybrid::proof::PbProof;
use hybrid::{
    solve_hybrid, solve_hybrid_parallel_with_progress, solve_hybrid_with_proof, verify_model, HybridConfig, SolveStats,
};
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
    ramsey_symm: Option<u32>,
    flips: usize,
    interval: usize,
    seed: Option<u64>,
    proof: Option<String>,
    cap: u64,
    model: Option<String>,
    deep_trail: Option<String>,
    hard_cube: Option<String>,
    preprocess_only: bool,
    timeout: Option<u64>,
    heartbeat: Option<u64>,
    csv: Option<String>,
    use_gpu: bool,
    recube: Option<hybrid::cdcl::RecubeConfig>,
}

fn parse_options(arguments: &[String]) -> Result<Options, String> {
    let defaults = HybridConfig::default();
    let mut options = Options {
        input: String::new(), workers: None, cubes: SwarmConfig::default().cubes_per_worker, share: true,
        symmetry: true, parity: true, ramsey_symm: None, flips: defaults.oracle.flips_per_variable,
        interval: defaults.reignition_interval, seed: None, proof: None, cap: u64::MAX, model: None,
        deep_trail: None, hard_cube: None,
        preprocess_only: false, timeout: None, heartbeat: None, csv: None, use_gpu: false, recube: None,
    };
    let mut index = 0;
    let mut recube_split = None;
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
            "--recube-conflicts" => {
                let budget = usize::try_from(number(value(flag)?)?.max(1))
                    .map_err(|_| "Presupuesto de conflictos demasiado grande")?;
                options.recube = Some(hybrid::cdcl::RecubeConfig {
                    max_conflicts_per_cube: budget,
                    ..hybrid::cdcl::RecubeConfig::default()
                });
            }
            "--recube-split" => {
                let bits = number(value(flag)?)?;
                if !(2..=3).contains(&bits) {
                    return Err("--recube-split exige 2 o 3 (4 u 8 sub-cubos)".into());
                }
                recube_split = Some(bits as usize);
            }
            "--no-share" => options.share = false,
            "--no-symmetry" => options.symmetry = false,
            "--no-parity" => options.parity = false,
            "--ramsey-symm" => {
                options.ramsey_symm = Some(u32::try_from(number(value(flag)?)?).map_err(|_| "Demasiados vertices")?);
            }
            "--flips" => options.flips = number(value(flag)?)? as usize,
            "--interval" => options.interval = number(value(flag)?)? as usize,
            "--seed" => options.seed = Some(number(value(flag)?)?),
            "--proof" => options.proof = Some(value(flag)?),
            "--proof-cap-gib" => options.cap = number(value(flag)?)?.saturating_mul(1 << 30),
            "--model" => options.model = Some(value(flag)?),
            "--deep-trail" => options.deep_trail = Some(value(flag)?),
            "--hard-cube" => options.hard_cube = Some(value(flag)?),
            "--preprocess-only" => options.preprocess_only = true,
            "--timeout" => options.timeout = Some(number(value(flag)?)?),
            "--heartbeat" => options.heartbeat = Some(number(value(flag)?)?.max(1)),
            "--csv" => options.csv = Some(value(flag)?),
            "--use-gpu" => options.use_gpu = true,
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
    if options.proof.is_some() && options.ramsey_symm.is_some() {
        return Err("--ramsey-symm no admite --proof: la traza hablaria de una formula que no es la del fichero".into());
    }
    if options.use_gpu && (options.workers.is_none() || options.preprocess_only) {
        return Err("--use-gpu exige --workers y no admite --preprocess-only".into());
    }
    if options.recube.is_some() && (options.workers.is_none() || options.preprocess_only) {
        return Err("--recube-conflicts exige --workers y no admite --preprocess-only".into());
    }
    if options.deep_trail.is_some() && (options.workers.is_none() || options.preprocess_only) {
        return Err("--deep-trail exige --workers y no admite --preprocess-only".into());
    }
    if let Some(bits) = recube_split {
        let recube = options.recube.as_mut().ok_or("--recube-split exige --recube-conflicts")?;
        recube.split_variables = bits;
    }
    if options.hard_cube.is_some() && options.recube.is_none() {
        return Err("--hard-cube exige --recube-conflicts y --workers".into());
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

// Linea de estado del enjambre. `last` es el instante y los conflictos de la linea anterior:
// el ritmo es el de ese tramo, no la media desde el arranque.
fn status_line(
    elapsed: Duration,
    conflicts: u64,
    last: (Duration, u64, u64),
    closed: u64,
    total: u64,
    lbd_sum: u64,
    restarts: u64,
) -> String {
    let span = elapsed.saturating_sub(last.0).as_secs_f64();
    let rate = if span > 0.0 {
        conflicts.saturating_sub(last.1) as f64 / span
    } else {
        0.0
    };
    let restart_rate = if span > 0.0 {
        restarts.saturating_sub(last.2) as f64 / span
    } else {
        0.0
    };

    // Medida por conflicto: los terminales sin cláusula aprendida aportan LBD cero.
    // Las lecturas atómicas independientes producen una instantánea aproximada.
    let mean_lbd = if conflicts != 0 {
        lbd_sum as f64 / conflicts as f64
    } else {
        0.0
    };

    let cubes = if total == 0 {
        "sin cubos todavia".to_owned()
    } else {
        format!("cubos cerrados {closed}/{total}")
    };

    format!(
        "c status: t={}s, {conflicts} conflictos, {rate:.0} conflictos/s, \
         {cubes}; LBD_medio={mean_lbd:.1}, restarts={restarts}, \
         restarts/s={restart_rate:.2}",
        elapsed.as_secs(),
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

// Lotes que la CPU enviara a la GPU: cubos por lote y literales por cubo como mucho.
const GPU_BATCH_CUBES: usize = 1 << 16;
const GPU_CUBE_DEPTH: usize = 32;
// Tamano de la sonda de arranque, en literales de la formula que puede llegar a leer: con
// formulas grandes son pocos cubos, para que no retrase la busqueda mas de unos segundos.
const GPU_PROBE_WORK: usize = 1 << 31;

// Abre la tarjeta, reserva sus buffers a la medida de la formula y la deja subida. Lo que
// encuentra lo cuenta por stderr; la busqueda no se entera de que existe.
fn open_gpu(formula: &Formula) -> Result<gpu_dispatcher::GpuConqueror, String> {
    let failed = |error: gpu_dispatcher::GpuError| format!("--use-gpu: {error}");
    let mib = |bytes: u64| bytes as f64 / (1 << 20) as f64;
    let start = Instant::now();
    let plan = gpu_dispatcher::BufferPlan::for_formula(formula, GPU_BATCH_CUBES, GPU_CUBE_DEPTH);
    let mut conqueror = gpu_dispatcher::GpuConqueror::new(plan).map_err(failed)?;
    let info = conqueror.info();
    eprintln!(
        "c gpu: {} (dispositivo {}, capacidad de computo {}.{}); VRAM {:.0} MiB libres de {:.0} MiB",
        info.name, info.ordinal, info.compute_capability.0, info.compute_capability.1,
        mib(info.free_bytes), mib(info.total_bytes),
    );
    conqueror.load_formula(formula).map_err(failed)?;
    if !conqueror.verify_formula().map_err(failed)? {
        return Err("--use-gpu: la formula releida de VRAM no coincide con la enviada".into());
    }
    eprintln!(
        "c gpu: {:.1} MiB reservados; formula subida y releida sin diferencias ({} clausulas, {} literales); lotes de hasta {} cubos de {} literales; quedan {:.0} MiB libres ({:.1} ms)",
        mib(plan.bytes()), plan.clauses, plan.literals, plan.cubes, GPU_CUBE_DEPTH,
        mib(conqueror.free_bytes().map_err(failed)?), start.elapsed().as_secs_f64() * 1000.0,
    );
    // Sonda: cubos de un solo literal (x1, no x1, x2, ...) por el nucleo, para ver que decide
    // y a que ritmo con esta formula. Un cubo refutado aqui es un literal que falla.
    let probes = (GPU_PROBE_WORK / plan.literals.max(1)).clamp(16, GPU_BATCH_CUBES).min(2 * plan.variables);
    let cubes: Vec<Vec<i32>> = (1..=plan.variables as i32).flat_map(|id| [vec![id], vec![-id]]).take(probes).collect();
    let start = Instant::now();
    let verdicts = conqueror.dispatch_cubes(formula, &cubes);
    let seconds = start.elapsed().as_secs_f64();
    if let Some(error) = conqueror.last_error() {
        return Err(format!("--use-gpu: {error}"));
    }
    let count = |wanted: fn(&gpu_dispatcher::SolveResult) -> bool| verdicts.iter().filter(|verdict| wanted(verdict)).count();
    eprintln!(
        "c gpu: sonda de {} cubos de un literal en {:.1} ms ({:.0} cubos/s): {} UNSAT, {} SAT, {} sin decidir",
        cubes.len(), seconds * 1000.0, cubes.len() as f64 / seconds.max(1e-9),
        count(|verdict| *verdict == gpu_dispatcher::SolveResult::Unsat),
        count(|verdict| matches!(verdict, gpu_dispatcher::SolveResult::Sat(_))),
        count(|verdict| *verdict == gpu_dispatcher::SolveResult::Unknown),
    );
    eprintln!("c gpu: prefiltro Cube and Conquer listo; los cubos no decididos pasan al enjambre CPU");
    Ok(conqueror)
}

fn solve_hybrid_parallel_with_gpu(
    formula: &Formula,
    config: HybridConfig,
    swarm: SwarmConfig,
    progress: Option<&SwarmProgress>,
    gpu: Option<&mut gpu_dispatcher::GpuConqueror>,
    recube: Option<hybrid::cdcl::RecubeConfig>,
) -> hybrid::ParallelHybridReport {
    let Some(gpu) = gpu else {
        return match recube {
            Some(recube) => hybrid::solve_hybrid_parallel_with_recubing(
                formula, config, swarm, recube, progress, None,
            ),
            None => solve_hybrid_parallel_with_progress(formula, config, swarm, progress),
        };
    };
    let max_cubes = gpu.plan().cubes.min(hybrid::cdcl::CUBE_FILTER_BATCH_CUBES);
    let max_cube_literals = gpu.plan().cube_literals;
    let mut filter = |cubes: &[Vec<i32>]| -> Vec<hybrid::cdcl::CubeVerdict> {
        use gpu_dispatcher::SolveResult;
        use hybrid::cdcl::CubeVerdict;

        let mut verdicts = vec![CubeVerdict::Unknown; cubes.len()];
        let max_width = cubes.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let batch_capacity = max_cubes.min(max_cube_literals / max_width);
        if batch_capacity == 0 { return verdicts; }

        for (batch_index, batch) in cubes.chunks(batch_capacity).enumerate() {
            // El plan contiene la formula original, no la extendida por preprocess.
            let results = gpu.dispatch_cubes(formula, batch);
            if let Some(error) = gpu.last_error() {
                eprintln!("c gpu: lote devuelto a CPU: {error}");
                continue;
            }
            if results.len() != batch.len() { continue; }

            let offset = batch_index * batch_capacity;
            for (index, result) in results.into_iter().enumerate() {
                verdicts[offset + index] = match result {
                    SolveResult::Sat(model) => {
                        // dispatch_cubes ya verifica formula y cubo en la CPU.
                        verdicts[offset + index] = CubeVerdict::Sat(model);
                        return verdicts;
                    }
                    SolveResult::Unsat => {
                        if let Some(progress) = progress {
                            progress.gpu_filtered.fetch_add(1, Ordering::Relaxed);
                        }
                        CubeVerdict::Unsat
                    }
                    SolveResult::Unknown => CubeVerdict::Unknown,
                };
            }
        }
        verdicts
    };
    match recube {
        Some(recube) => hybrid::solve_hybrid_parallel_with_recubing(
            formula, config, swarm, recube, progress, Some(&mut filter),
        ),
        None => hybrid::solve_hybrid_parallel_with_filter(
            formula, config, swarm, progress, Some(&mut filter),
        ),
    }
}

fn run(options: &Options) -> Result<i32, String> {
    let started = Instant::now();
    let text = std::fs::read_to_string(&options.input).map_err(|error| format!("{}: {error}", options.input))?;
    let mut formula = parse_dimacs(&text)?;
    drop(text);
    let parse_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut variables = formula.iter().flatten().map(|literal| literal.unsigned_abs()).max().unwrap_or(0);
    // Sin --ramsey-symm la formula llega al motor tal como esta en el fichero.
    if let Some(vertices) = options.ramsey_symm {
        let start = Instant::now();
        if ramsey_symmetry::edge_count(vertices) != Some(variables) {
            return Err(format!("--ramsey-symm {vertices}: la formula tiene {variables} variables, no las de K_{vertices}"));
        }
        if !ramsey_symmetry::is_vertex_symmetric(vertices, &formula) {
            return Err(format!("--ramsey-symm {vertices}: la formula no es simetrica en los vertices; no se anade nada"));
        }
        let mut breaker = symmetry_breaker::SymmetryBreaker::new(variables + 1);
        let before = formula.len();
        ramsey_symmetry::inject_ramsey_symmetries(vertices, &mut breaker, &mut formula);
        // El motor tiene que ver tambien las auxiliares: el limite de variables crece.
        variables = breaker.next_free() - 1;
        eprintln!(
            "c symm: K_{vertices} simetrica en los vertices; {} clausulas y {} variables auxiliares anadidas ({:.1} ms)",
            formula.len() - before, breaker.auxiliaries(), start.elapsed().as_secs_f64() * 1000.0,
        );
    }
    // Sin --use-gpu no se abre CUDA ni se construye el prefiltro.
    let mut conqueror = if options.use_gpu { Some(open_gpu(&formula)?) } else { None };
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
    // Sin GPU, latido ni limite, el enjambre conserva su ruta sin observador.
    let gpu_active = conqueror.is_some();
    let recube_active = options.recube.is_some();
    let deep_active = options.deep_trail.is_some();
    let hard_active = options.hard_cube.is_some();
    let mut progress_state = (mode == "swarm"
        && (options.heartbeat.is_some() || options.timeout.is_some() || gpu_active || deep_active || hard_active))
        .then(SwarmProgress::default);
    let deep_trail_writer = if let Some(path) = &options.deep_trail {
        let observer = progress_state.as_mut().expect("deep-trail requires swarm progress");
        Some(observer.enable_deep_trail(path).map_err(|error| format!("{path}: {error}"))?)
    } else { None };
    let hard_cube_writer = if let Some(path) = &options.hard_cube {
        let observer = progress_state.as_mut().expect("hard-cube requires swarm progress");
        Some(observer.enable_hard_cube(path).map_err(|error| format!("{path}: {error}"))?)
    } else { None };
    if let Some(recube) = &options.recube {
        eprintln!("c micro-cubing: budget={}, split_variables={}, max_children={}, gc_interval=10000, protected_lbd=4",
            recube.max_conflicts_per_cube, recube.split_variables, 1usize << recube.split_variables);
    }
    let progress = progress_state.map(Arc::new);
    {
        let (gauge, prefix, csv) = (Arc::clone(&gauge), prefix.clone(), options.csv.is_some());
        let (timeout, heartbeat) = (options.timeout, options.heartbeat);
        let progress = progress.clone();
        std::thread::spawn(move || {
            let mut next = heartbeat;
            // Instante y conflictos de la ultima linea `c status:`.
            // Instante, conflictos y reinicios de la última publicación.
            let mut last = (Duration::ZERO, 0u64, 0u64);
            loop {
                std::thread::sleep(Duration::from_millis(100));
                let elapsed = started.elapsed();
                let beat = next.is_some_and(|due| elapsed.as_secs() >= due);
                let expired = timeout.is_some_and(|limit| elapsed.as_secs() >= limit);
                if beat {
                    eprintln!(
                        "c t={}s traza={} bytes, {} lineas{}", elapsed.as_secs(),
                        gauge.bytes.load(Ordering::Relaxed), gauge.lines.load(Ordering::Relaxed),
                        if gauge.truncated.load(Ordering::Relaxed) { " (TRUNCADA)" } else { "" },
                    );
                    next = next.zip(heartbeat).map(|(due, step)| due + step);
                }
                // En cada latido y al agotar el limite: sin resultado no habra contadores
                // finales, y esta linea es lo unico que queda de la busqueda.
                if let Some(progress) = progress.as_deref().filter(|_| beat || expired) {
                    let conflicts = progress.conflicts.load(Ordering::Relaxed);
                    let lbd_sum = progress.lbd_sum.load(Ordering::Relaxed);
                    let restarts = progress.restarts.load(Ordering::Relaxed);
                    let cpu_closed = progress.cubes_closed.load(Ordering::Relaxed);
                    let total = progress.cubes_total.load(Ordering::Relaxed);
                    let recube_suffix = if recube_active {
                        format!("; recubings={}, yielded={}",
                            progress.cubes_split.load(Ordering::Relaxed),
                            progress.cubes_yielded.load(Ordering::Relaxed))
                    } else {
                        String::new()
                    };
                    let depth_suffix = if deep_active {
                        format!("; deepest_trail={}",
                            progress.max_trail_depth.load(Ordering::Relaxed))
                    } else { String::new() };
                    let hard_suffix = if hard_active {
                        format!("; hard_cube_queued={}", progress.hard_cube_queued.load(Ordering::Relaxed))
                    } else { String::new() };

                    if gpu_active {
                        let gpu_filtered = progress.gpu_filtered.load(Ordering::Relaxed);
                        eprintln!(
                            "{}{recube_suffix}{depth_suffix}{hard_suffix}; gpu_filtered={gpu_filtered}",
                            status_line(
                                elapsed,
                                conflicts,
                                last,
                                cpu_closed.saturating_add(gpu_filtered),
                                total,
                                lbd_sum,
                                restarts,
                            ),
                        );
                    } else {
                        eprintln!(
                            "{}{recube_suffix}{depth_suffix}{hard_suffix}",
                            status_line(
                                elapsed,
                                conflicts,
                                last,
                                cpu_closed,
                                total,
                                lbd_sum,
                                restarts,
                            ),
                        );
                    }

                    last = (elapsed, conflicts, restarts);
                }
                if expired {
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
        let report = solve_hybrid_parallel_with_gpu(
            &formula, config, swarm, progress.as_deref(), conqueror.as_mut(), options.recube,
        );
        let stats = &report.swarm;
        if recube_active {
            eprintln!(
                "c recubing: yielded={}, splits={}, leaves={}, gpu_filtered={}",
                stats.cubes_yielded, stats.cubes_split,
                stats.cubes_generated - stats.cubes_split,
                progress.as_ref().map_or(0, |p| p.gpu_filtered.load(Ordering::Relaxed)),
            );
        }
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
    // El enjambre ya unio sus workers; drenar el ultimo record antes de salir.
    drop(deep_trail_writer);
    drop(hard_cube_writer);
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
    if gpu_active {
        let gpu_filtered = progress.as_deref().map_or(0, |progress| {
            progress.gpu_filtered.load(Ordering::Relaxed)
        });
        eprintln!("c gpu: gpu_filtered={gpu_filtered} cubos refutados por GPU");
    }
    if hard_active {
        eprintln!("c hard-cube: queued={}; archivo parcial, no certificado SAT",
            progress.as_deref().map_or(0, |p| p.hard_cube_queued.load(Ordering::Relaxed)));
    }
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
    fn the_gpu_is_only_opened_when_asked_for() {
        let arguments = |text: &str| -> Vec<String> { text.split_whitespace().map(str::to_owned).collect() };
        assert!(!parse_options(&arguments("f.cnf --workers 4")).unwrap().use_gpu);
        let options = parse_options(&arguments("f.cnf --use-gpu --workers 4")).unwrap();
        assert!(options.use_gpu);
        assert_eq!(options.workers, Some(4));
    }

    #[test]
    fn the_swarm_cannot_be_asked_for_a_trace() {
        let arguments: Vec<String> = ["f.cnf", "--workers", "4", "--proof", "p.pbp"].iter().map(|text| text.to_string()).collect();
        assert!(parse_options(&arguments).is_err());
        let options = parse_options(&arguments[..3]).unwrap();
        assert_eq!((options.workers, options.flips, options.interval), (Some(4), 16, 500));
    }

    #[test]
    fn the_status_line_gives_the_rate_since_the_previous_one() {
        let seconds = Duration::from_secs;

        assert_eq!(
            status_line(
                seconds(60),
                120_000,
                (seconds(30), 30_000, 390),
                3,
                64,
                600_000,
                450,
            ),
            "c status: t=60s, 120000 conflictos, 3000 conflictos/s, \
             cubos cerrados 3/64; LBD_medio=5.0, restarts=450, restarts/s=2.00",
        );

        assert_eq!(
            status_line(
                seconds(30),
                0,
                (Duration::ZERO, 0, 0),
                0,
                0,
                0,
                0,
            ),
            "c status: t=30s, 0 conflictos, 0 conflictos/s, \
             sin cubos todavia; LBD_medio=0.0, restarts=0, restarts/s=0.00",
        );

        let same_instant = status_line(
            seconds(30),
            500,
            (seconds(30), 400, 2),
            0,
            16,
            2_000,
            3,
        );
        assert!(same_instant.contains(", 0 conflictos/s, "));
        assert!(same_instant.contains("LBD_medio=4.0"));
        assert!(same_instant.ends_with("restarts=3, restarts/s=0.00"));
    }

    #[test]
    fn ramsey_symmetry_breaking_is_off_unless_asked_for() {
        let arguments = |text: &str| -> Vec<String> { text.split_whitespace().map(str::to_owned).collect() };
        assert_eq!(parse_options(&arguments("f.cnf")).unwrap().ramsey_symm, None);
        let options = parse_options(&arguments("f.cnf --ramsey-symm 43 --workers 2")).unwrap();
        assert_eq!(options.ramsey_symm, Some(43));
        assert!(options.symmetry);
        assert!(parse_options(&arguments("f.cnf --ramsey-symm")).is_err());
        assert!(parse_options(&arguments("f.cnf --ramsey-symm 18 --proof p.pbp")).is_err());
    }

    #[test]
    fn deep_trail_is_explicit_and_requires_swarm_search() {
        let args = |text: &str| text.split_whitespace().map(str::to_owned).collect::<Vec<_>>();
        assert!(parse_options(&args("f.cnf")).unwrap().deep_trail.is_none());
        let options = parse_options(&args("f.cnf --workers 2 --deep-trail fragmento.sat")).unwrap();
        assert_eq!(options.deep_trail.as_deref(), Some("fragmento.sat"));
        assert!(parse_options(&args("f.cnf --deep-trail fragmento.sat")).is_err());
        assert!(parse_options(&args("f.cnf --workers 2 --deep-trail")).is_err());
        assert!(parse_options(&args("f.cnf --workers 2 --deep-trail f --preprocess-only")).is_err());
    }

    #[test]
    fn micro_cubing_is_explicit_and_independent_of_argument_order() {
        let args = |text: &str| text.split_whitespace().map(str::to_owned).collect::<Vec<_>>();
        let plain = parse_options(&args("f.cnf --workers 2")).unwrap();
        assert!(plain.recube.is_none() && plain.hard_cube.is_none());
        let defaults = parse_options(&args("f.cnf --workers 2 --recube-conflicts 50000")).unwrap();
        let config = defaults.recube.unwrap();
        assert_eq!(config.max_conflicts_per_cube, 50_000);
        assert_eq!(config.split_variables, 3);
        assert!(config.heavy_tail_gc);
        for text in [
            "f.cnf --workers 2 --recube-split 2 --recube-conflicts 50000",
            "f.cnf --workers 2 --recube-conflicts 50000 --recube-split 2",
        ] {
            let options = parse_options(&args(text)).unwrap();
            assert_eq!(options.recube.unwrap().split_variables, 2);
        }
        for text in [
            "f.cnf --workers 2 --recube-split 3",
            "f.cnf --workers 2 --recube-conflicts 50000 --recube-split 0",
            "f.cnf --workers 2 --recube-conflicts 50000 --recube-split 4",
            "f.cnf --recube-conflicts 50000",
            "f.cnf --workers 2 --recube-conflicts 50000 --preprocess-only",
        ] {
            assert!(parse_options(&args(text)).is_err(), "{text}");
        }
    }

    #[test]
    fn hard_cube_is_opt_in_and_requires_recubing() {
        let args = |text: &str| text.split_whitespace().map(str::to_owned).collect::<Vec<_>>();
        let options = parse_options(&args("f.cnf --workers 2 --recube-conflicts 50000 --hard-cube hard.sat")).unwrap();
        assert_eq!(options.hard_cube.as_deref(), Some("hard.sat"));
        assert!(options.deep_trail.is_none());
        for text in [
            "f.cnf --workers 2 --hard-cube hard.sat",
            "f.cnf --hard-cube hard.sat --recube-conflicts 50000",
            "f.cnf --workers 2 --recube-conflicts 50000 --hard-cube",
        ] {
            assert!(parse_options(&args(text)).is_err(), "{text}");
        }
    }
}
