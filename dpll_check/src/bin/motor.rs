// Comando unico del motor: traduce la linea de comandos a la API de solver.rs e imprime el
// resultado. No da un resultado por bueno sin comprobarlo: un SAT contra la formula del
// fichero y un UNSAT con VeriPB sobre la traza.
//
//   motor solve <formula.cnf> [opciones]
//   motor solve <problema.opb> [opciones]   restricciones lineales y objetivo min:
//   motor gen ramsey <N> [salida.cnf] [--cliques <S> <T>]
//   motor gen erdos <N> [salida.cnf] [--c <C>]
//   motor k43 [opciones]    busqueda GPU en K43; exige --features gpu (--help para las opciones)
//   motor bench [opciones]  banco del DFS de referencia sobre R(s,t) conocidos
//   motor trace <formula.cnf> [opciones]  banco del puente de la traza (Chronos-VFS)
//   motor --header      cabecera de las filas de --csv
//
// Opciones de solve:
//     --no-verify         acepta un UNSAT sin certificado; se avisa en la salida
//     --json              una sola linea JSON en la salida estandar, en lugar de `s` y `v`;
//                         un error tambien sale asi, con "status":"ERROR"
//     --cnf F             en un .opb, conserva la ultima CNF traducida
//     --workers K         enjambre de K workers, con una sola traza; sin esta opcion, un hilo
//     --cubes C           cubos por worker (4)
//     --recube-conflicts C cede cubos tras C conflictos y los reenvía a la GPU (opcional)
//     --recube-split K     fija 2 o 3 variables VSIDS por split (4 u 8 hijos; defecto 3)
//     --hard-cube F        exporta un trail parcial tras >3 recubings en un linaje
//     --no-share          enjambre sin intercambio de clausulas
//     --no-symmetry       sin ruptura de simetrias
//     --no-parity         sin capa de paridad
//     --no-congruence     sin equivalencias ni cierre por congruencia
//     --ramsey-symm N     la formula es de Ramsey sobre K_N (motor gen ramsey): se comprueba que es
//                         simetrica en los vertices y se le anaden las clausulas que los
//                         ordenan (ramsey_symmetry.rs). No admite --proof
//     --flips F           flips por variable del oraculo (16)
//     --interval I        conflictos entre re-igniciones (500); 0 apaga el oraculo
//     --seed S            semilla del oraculo
//     --proof F.pbp       conserva la traza VeriPB 3.0; sin esta opcion va a un temporal
//                         que se borra al terminar. Con F.pbp.zst pasa por el anillo en RAM
//                         y se comprime al vuelo
//     --proof-cap-gib G   tope de la traza (64 al verificar); superado, un UNSAT queda
//                         sin certificado y se descarta
//     --model F           escribe el modelo si la formula es satisfacible
//     --deep-trail F      exporta records de asignacion parcial; exige --workers
//     --preprocess-only   solo las capas de simetria y paridad, sin busqueda
//     --timeout S         aborta a los S segundos (estado TIMEOUT, codigo 124)
//     --heartbeat S       linea de progreso en stderr cada S segundos
//     --csv ETIQUETA      una fila CSV en la salida estandar (`motor --header` da la cabecera)
//     --use-gpu           filtra cubos con propagacion unitaria CUDA antes del enjambre;
//                         exige --workers y compilar con --features gpu.
//                         Los cubos Unknown conservan la busqueda CDCL existente.
//
// Un .opb no admite --ramsey-symm, --use-gpu, --recube-*, --deep-trail, --hard-cube,
// --preprocess-only ni --csv.
//
// El motor no tiene puntos de control. El enjambre publica sus conflictos y cubos cerrados: con
// --heartbeat o --timeout sale una linea `c status:` en cada latido y otra al agotar el limite.
// `deny` y no `forbid`: lanzar un nucleo CUDA es una llamada `unsafe`, y gpu_dispatcher.rs
// la permite en una sola funcion. En el resto del binario sigue siendo un error.
#![deny(unsafe_code)]
#![deny(warnings)]

use motor_core::dimacs::Formula;
use motor_core::solver::{self, json_string, Event, Extras, Halt, Problem, Stats, Status};
use motor_core::verifier;
use motor_core::{gen_erdos, gen_ramsey, gpu_dispatcher, hybrid, ramsey_bench, ramsey_symmetry, symmetry_breaker, trace_bridge};
use hybrid::cdcl::{CubeFilter, SwarmConfig, SwarmProgress};
use hybrid::HybridConfig;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const USAGE: &str = "Uso:\n  motor solve <formula.cnf> [opciones]\n  motor solve <problema.opb> [--timeout S] [--workers K] [--no-verify] [--json] [--cnf F] [--proof F]\n  motor gen ramsey <N> [salida.cnf] [--cliques <S> <T>]\n  motor gen erdos <N> [salida.cnf] [--c <C>]\n  motor k43 [opciones]\n  motor bench [opciones]\n  motor trace <formula.cnf> [opciones]\n  motor --header";
const HEADER: &str = "label,mode,workers,cubes_per_worker,share,symmetry,parity,flips,interval,variables,clauses,status,total_ms,parse_ms,symmetry_generators,symmetry_clauses,symmetry_ms,parity_equations,parity_ms,search_ms,decisions,conflicts,restarts,deleted_clauses,reignitions,reignition_flips,reignition_ms,cubes_generated,cubes_started,cubes_unsat,cubes_cancelled,exports,imports,trace_bytes,trace_lines,trace_dominances,trace_deletions,trace_truncated";

struct Options {
    input: String,
    workers: Option<usize>,
    cubes: usize,
    share: bool,
    symmetry: bool,
    parity: bool,
    congruence: bool,
    ramsey_symm: Option<u32>,
    flips: usize,
    interval: usize,
    seed: Option<u64>,
    verify: bool,
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
    json: bool,
    cnf: Option<String>,
}

fn parse_options(arguments: &[String]) -> Result<Options, String> {
    let defaults = HybridConfig::default();
    let mut options = Options {
        input: String::new(), workers: None, cubes: SwarmConfig::default().cubes_per_worker, share: true,
        symmetry: true, parity: true, congruence: true, ramsey_symm: None, flips: defaults.oracle.flips_per_variable,
        interval: defaults.reignition_interval, seed: None, verify: true, proof: None, cap: u64::MAX, model: None,
        deep_trail: None, hard_cube: None,
        preprocess_only: false, timeout: None, heartbeat: None, csv: None, use_gpu: false, recube: None,
        json: false, cnf: None,
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
            "--no-congruence" => options.congruence = false,
            "--ramsey-symm" => {
                options.ramsey_symm = Some(u32::try_from(number(value(flag)?)?).map_err(|_| "Demasiados vertices")?);
            }
            "--flips" => options.flips = number(value(flag)?)? as usize,
            "--interval" => options.interval = number(value(flag)?)? as usize,
            "--seed" => options.seed = Some(number(value(flag)?)?),
            "--no-verify" => options.verify = false,
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
            "--json" => options.json = true,
            "--cnf" => options.cnf = Some(value(flag)?),
            _ if flag.starts_with("--") => return Err(format!("Opcion desconocida: {flag}")),
            _ if options.input.is_empty() => options.input = flag.to_owned(),
            _ => return Err(format!("Sobra el argumento {flag}")),
        }
    }
    if options.input.is_empty() {
        return Err(USAGE.into());
    }
    if options.input.ends_with(".opb") {
        let refused = [
            ("--ramsey-symm", options.ramsey_symm.is_some()), ("--use-gpu", options.use_gpu),
            ("--recube-conflicts", options.recube.is_some()), ("--deep-trail", options.deep_trail.is_some()),
            ("--hard-cube", options.hard_cube.is_some()), ("--preprocess-only", options.preprocess_only),
            ("--csv", options.csv.is_some()),
        ];
        if let Some((flag, _)) = refused.iter().find(|(_, used)| *used) {
            return Err(format!("Opcion no admitida con .opb: {flag}"));
        }
    } else if options.cnf.is_some() {
        return Err("--cnf solo tiene sentido con un .opb".into());
    }
    if options.json && options.csv.is_some() {
        return Err("--json y --csv son dos salidas distintas: elige una".into());
    }
    options.verify &= !options.preprocess_only;
    if options.verify && options.recube.is_some() {
        return Err("--recube-conflicts no emite traza y su UNSAT no se puede verificar: quitalo o anade --no-verify".into());
    }
    if options.verify && options.ramsey_symm.is_some() {
        return Err("--ramsey-symm cambia la formula y su UNSAT no se puede verificar: quitalo o anade --no-verify".into());
    }
    if options.proof.is_some() && options.recube.is_some() {
        return Err("--recube-conflicts no emite traza: no admite --proof".into());
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

fn row(prefix: &str, status: &str, total_ms: f64, parse_ms: f64, stats: &Stats) -> String {
    let search = &stats.search;
    format!(
        "{prefix},{status},{total_ms:.3},{parse_ms:.3},{},{},{:.3},{},{:.3},{:.3},{},{},{},{},{},{},{:.3},{},{},{},{},{},{},{},{},{},{},{}",
        stats.symmetry.generators, stats.symmetry.clauses, stats.symmetry.ms,
        stats.parity.equations, stats.parity.ms, stats.search_ms,
        search.decisions, search.conflicts, search.restarts, search.deleted_clauses,
        search.reignitions, search.reignition_flips, search.reignition_ms,
        stats.cubes_generated, stats.cubes_started, stats.cubes_unsat, stats.cubes_cancelled,
        search.exported_clauses, search.imported_clauses, stats.trace_bytes, stats.trace_lines,
        stats.trace_dominances, stats.trace_deletions, stats.trace_truncated,
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

// Prefiltro del enjambre sobre `formula`, la original: el plan de la GPU no conoce la extendida.
fn gpu_filter<'a>(
    gpu: &'a mut gpu_dispatcher::GpuConqueror, formula: &'a Formula, progress: Arc<SwarmProgress>,
) -> impl FnMut(&[Vec<i32>]) -> Vec<hybrid::cdcl::CubeVerdict> + 'a {
    let max_cubes = gpu.plan().cubes.min(hybrid::cdcl::CUBE_FILTER_BATCH_CUBES);
    let max_cube_literals = gpu.plan().cube_literals;
    move |cubes: &[Vec<i32>]| {
        use gpu_dispatcher::SolveResult;
        use hybrid::cdcl::CubeVerdict;

        let mut verdicts = vec![CubeVerdict::Unknown; cubes.len()];
        let max_width = cubes.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let batch_capacity = max_cubes.min(max_cube_literals / max_width);
        if batch_capacity == 0 { return verdicts; }

        for (batch_index, batch) in cubes.chunks(batch_capacity).enumerate() {
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
                        progress.gpu_filtered.fetch_add(1, Ordering::Relaxed);
                        CubeVerdict::Unsat
                    }
                    SolveResult::Unknown => CubeVerdict::Unknown,
                };
            }
        }
        verdicts
    }
}

// Lineas de avance por stderr: las del latido y, al agotar el limite, la ultima del enjambre.
fn progress_printer(options: &Options, progress: Arc<SwarmProgress>) -> solver::EventHandler {
    let (heartbeat, swarm, json) = (options.heartbeat.is_some(), options.workers.is_some(), options.json);
    let (gpu_active, recube_active) = (options.use_gpu, options.recube.is_some());
    let (deep_active, hard_active) = (options.deep_trail.is_some(), options.hard_cube.is_some());
    // Instante, conflictos y reinicios de la última publicación.
    let last = Mutex::new((Duration::ZERO, 0u64, 0u64));
    Arc::new(move |event: &Event| {
        let report = match event {
            Event::Improved { cost } => {
                if !json {
                    println!("o {cost}");
                }
                return;
            }
            Event::Progress(report) => report,
        };
        if heartbeat && !report.stopping {
            eprintln!(
                "c t={}s traza={} bytes, {} lineas{}", report.elapsed.as_secs(), report.trace_bytes, report.trace_lines,
                if report.trace_truncated { " (TRUNCADA)" } else { "" },
            );
        }
        // Sin resultado no habra contadores finales: esta linea es lo unico que queda de la busqueda.
        if !swarm {
            return;
        }
        let recube_suffix = if recube_active {
            format!("; recubings={}, yielded={}",
                progress.cubes_split.load(Ordering::Relaxed), progress.cubes_yielded.load(Ordering::Relaxed))
        } else {
            String::new()
        };
        let depth_suffix = if deep_active {
            format!("; deepest_trail={}", progress.max_trail_depth.load(Ordering::Relaxed))
        } else { String::new() };
        let hard_suffix = if hard_active {
            format!("; hard_cube_queued={}", progress.hard_cube_queued.load(Ordering::Relaxed))
        } else { String::new() };
        let mut last = last.lock().expect("sin panicos con el candado");
        let gpu_filtered = progress.gpu_filtered.load(Ordering::Relaxed);
        let closed = if gpu_active { report.cubes_closed.saturating_add(gpu_filtered) } else { report.cubes_closed };
        let line = status_line(report.elapsed, report.conflicts, *last, closed, report.cubes_total, report.lbd_sum, report.restarts);
        if gpu_active {
            eprintln!("{line}{recube_suffix}{depth_suffix}{hard_suffix}; gpu_filtered={gpu_filtered}");
        } else {
            eprintln!("{line}{recube_suffix}{depth_suffix}{hard_suffix}");
        }
        *last = (report.elapsed, report.conflicts, report.restarts);
    })
}

fn run(options: Options) -> Result<i32, String> {
    let started = Instant::now();
    let failed = |error: solver::SolveError| error.to_string();
    // Se pregunta antes de leer: no tiene sentido cargar una formula enorme para nada.
    if options.verify {
        verifier::ensure_available()?;
    }
    let mut problem = Problem::from_file(&options.input).map_err(failed)?;
    let parse_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut variables = problem.variables();
    // Sin --ramsey-symm la formula llega al motor tal como esta en el fichero.
    if let Some(vertices) = options.ramsey_symm {
        let start = Instant::now();
        let mut formula = problem.formula().ok_or("--ramsey-symm exige CNF")?.clone();
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
        problem = Problem::cnf(formula).map_err(failed)?;
    }
    let formula = problem.formula();
    // Sin --use-gpu no se abre CUDA ni se construye el prefiltro.
    let mut conqueror = match formula.filter(|_| options.use_gpu) {
        Some(formula) => Some(open_gpu(formula)?),
        None => None,
    };
    let tracing = options.verify || options.proof.is_some();
    let mode = match (options.preprocess_only, options.workers, tracing) {
        (true, ..) => "preprocess",
        (false, Some(_), _) => "swarm",
        (false, None, true) => "serial_certified",
        (false, None, false) => "serial",
    };
    let prefix = format!(
        "{},{mode},{},{},{},{},{},{},{},{variables},{}",
        options.csv.as_deref().unwrap_or(""), options.workers.unwrap_or(1),
        if options.workers.is_some() { options.cubes } else { 0 },
        options.workers.is_some() && options.share, options.symmetry, options.parity,
        options.flips, options.interval, problem.constraints(),
    );
    match formula {
        Some(_) => eprintln!("c {}: {variables} variables, {} clausulas, modo {mode}", options.input, problem.constraints()),
        None => eprintln!("c {}: {variables} variables, {} restricciones", options.input, problem.constraints()),
    }

    let mut progress_state = SwarmProgress::default();
    let deep_trail_writer = if let Some(path) = &options.deep_trail {
        Some(progress_state.enable_deep_trail(path).map_err(|error| format!("{path}: {error}"))?)
    } else { None };
    let hard_cube_writer = if let Some(path) = &options.hard_cube {
        Some(progress_state.enable_hard_cube(path).map_err(|error| format!("{path}: {error}"))?)
    } else { None };
    if let Some(recube) = &options.recube {
        eprintln!("c micro-cubing: budget={}, split_variables={}, max_children={}, gc_interval=10000, protected_lbd=4",
            recube.max_conflicts_per_cube, recube.split_variables, 1usize << recube.split_variables);
    }
    let progress = Arc::new(progress_state);
    let request = solver::Options {
        workers: options.workers, cubes: options.cubes, share: options.share, symmetry: options.symmetry,
        parity: options.parity, congruence: options.congruence, flips: options.flips, interval: options.interval, seed: options.seed,
        certify: options.verify, proof: options.proof.as_ref().map(PathBuf::from), cnf: options.cnf.as_ref().map(PathBuf::from),
        proof_cap: (options.cap != u64::MAX).then_some(options.cap),
        // El limite cuenta desde el arranque del comando, lectura incluida.
        timeout: options.timeout.map(|limit| Duration::from_secs(limit).saturating_sub(started.elapsed())),
        stop: None, on_event: Some(progress_printer(&options, Arc::clone(&progress))),
        progress_interval: options.heartbeat.map(Duration::from_secs), preprocess_only: options.preprocess_only,
    };
    let solution = {
        let mut filter = match (conqueror.as_mut(), formula) {
            (Some(gpu), Some(formula)) => Some(gpu_filter(gpu, formula, Arc::clone(&progress))),
            _ => None,
        };
        let extras = Extras {
            filter: filter.as_mut().map(|filter| filter as CubeFilter<'_>),
            recube: options.recube, progress: Some(Arc::clone(&progress)),
        };
        solver::solve_with(&problem, &request, extras).map_err(failed)?
    };
    // El enjambre ya unio sus workers; drenar el ultimo record antes de salir.
    drop(deep_trail_writer);
    drop(hard_cube_writer);
    let (stats, total_ms) = (&solution.stats, started.elapsed().as_secs_f64() * 1000.0);
    if options.recube.is_some() {
        eprintln!(
            "c recubing: yielded={}, splits={}, leaves={}, gpu_filtered={}",
            stats.cubes_yielded, stats.cubes_split, stats.cubes_generated.saturating_sub(stats.cubes_split),
            progress.gpu_filtered.load(Ordering::Relaxed),
        );
    }
    let status = match (&solution.halt, solution.status) {
        (Some(Halt::Rejected(_)), _) => "REJECTED",
        (Some(_), _) => "TIMEOUT",
        (None, Status::Sat) => "SAT",
        (None, Status::Unsat) => "UNSAT",
        (None, Status::Optimum) => "OPTIMUM",
        (None, Status::Unknown) => "PREPROCESSED",
    };
    if options.csv.is_some() {
        println!("{}", row(&prefix, status, total_ms, parse_ms, stats));
    }
    if options.preprocess_only {
        eprintln!(
            "c simetria: {} generadores, {} predicados, {:.1} ms; paridad: {} ecuaciones, {:.1} ms; traza {} bytes",
            stats.symmetry.generators, stats.symmetry.clauses, stats.symmetry.ms,
            stats.parity.equations, stats.parity.ms, stats.trace_bytes,
        );
    } else {
        eprintln!(
            "c {total_ms:.1} ms: simetria {} generadores, {} predicados ({:.1} ms); {} decisiones, {} conflictos, {} re-igniciones",
            stats.symmetry.generators, stats.symmetry.clauses, stats.symmetry.ms,
            stats.search.decisions, stats.search.conflicts, stats.search.reignitions,
        );
    }
    let congruence = &stats.congruence;
    if congruence.equivalences + congruence.units != 0 || congruence.refuted {
        eprintln!(
            "c congruencia: {} AND, {} XOR, {} ITE; {} equivalencias, {} fijadas, {} clausulas menos ({:.1} ms)",
            congruence.ands, congruence.xors, congruence.ites, congruence.equivalences, congruence.units,
            congruence.clauses_removed, congruence.ms,
        );
    }
    if options.use_gpu {
        eprintln!("c gpu: gpu_filtered={} cubos refutados por GPU", progress.gpu_filtered.load(Ordering::Relaxed));
    }
    if options.hard_cube.is_some() {
        eprintln!("c hard-cube: queued={}; archivo parcial, no certificado SAT", progress.hard_cube_queued.load(Ordering::Relaxed));
    }
    if tracing && !options.preprocess_only {
        eprintln!(
            "c traza: {} bytes, {} lineas, {} dominancias, {} borrados{}",
            stats.trace_bytes, stats.trace_lines, stats.trace_dominances, stats.trace_deletions,
            if stats.trace_truncated { " -- TRUNCADA, sin valor de certificado" } else { "" },
        );
    }
    if solution.model_checked {
        eprintln!("c verificado: el modelo satisface {}", if formula.is_some() { "la formula del fichero" } else { "las restricciones del fichero" });
    }
    let refuted = matches!(solution.status, Status::Unsat | Status::Optimum) && solution.halt.is_none();
    match (refuted, solution.proof_checked, formula.is_some()) {
        (true, true, true) => eprintln!("c verificado: VeriPB acepto la prueba de UNSAT"),
        (true, true, false) => eprintln!("c verificado: VeriPB acepto la prueba sobre la CNF traducida"),
        (true, false, true) => eprintln!("c AVISO: UNSAT sin certificado (--no-verify)"),
        (true, false, false) => eprintln!("c AVISO: refutacion sin certificado (--no-verify)"),
        _ => {}
    }
    if matches!(solution.halt, Some(Halt::Timeout | Halt::Cancelled)) {
        eprintln!("c limite de tiempo agotado");
    }
    let line = solution.model.as_ref().map(|model| match formula {
        Some(_) => format!("v {} 0", model.iter().map(i32::to_string).collect::<Vec<_>>().join(" ")),
        None => format!("v {}", model.iter().map(|&literal| {
            format!("{}x{}", if literal < 0 { "-" } else { "" }, literal.unsigned_abs())
        }).collect::<Vec<_>>().join(" ")),
    });
    if let (Some(path), Some(line)) = (&options.model, &line) {
        std::fs::write(path, format!("{line}\n")).map_err(|error| format!("{path}: {error}"))?;
    }
    if options.json {
        println!("{}", solution.to_json());
    } else {
        println!("s {}", match solution.status {
            Status::Sat => "SATISFIABLE",
            Status::Unsat => "UNSATISFIABLE",
            Status::Optimum => "OPTIMUM FOUND",
            Status::Unknown => "UNKNOWN",
        });
        if let Some(line) = line.as_ref().filter(|_| options.csv.is_none()) {
            println!("{line}");
        }
    }
    match &solution.halt {
        Some(Halt::Rejected(reason)) if !options.json => Err(format!("Resultado descartado: {reason}")),
        _ => Ok(solution.exit_code()),
    }
}

fn dispatch(arguments: &[String]) -> Result<i32, String> {
    match arguments {
        [flag] if flag == "--header" => {
            println!("{HEADER}");
            Ok(0)
        }
        [command, rest @ ..] if command == "solve" => match parse_options(rest).and_then(run) {
            // Quien ha pedido JSON no tiene por que leer stderr para enterarse de un error.
            Err(error) if rest.iter().any(|argument| argument == "--json") => {
                println!("{{\"status\":\"ERROR\",\"error\":{}}}", json_string(&error));
                Ok(1)
            }
            result => result,
        },
        [command, family, rest @ ..] if command == "gen" && family == "ramsey" => gen_ramsey::run(rest).map(|()| 0),
        [command, family, rest @ ..] if command == "gen" && family == "erdos" => gen_erdos::run(rest).map(|()| 0),
        [command, rest @ ..] if command == "k43" => k43(rest),
        [command, rest @ ..] if command == "bench" => {
            ramsey_bench::run(rest);
            Ok(0)
        }
        [command, rest @ ..] if command == "trace" => trace_bridge::run(rest),
        _ => Err(USAGE.into()),
    }
}

#[cfg(feature = "gpu")]
fn k43(arguments: &[String]) -> Result<i32, String> {
    Ok(motor_core::k43::run(arguments))
}

#[cfg(not(feature = "gpu"))]
fn k43(_: &[String]) -> Result<i32, String> {
    Err("motor k43 necesita la GPU: compila con --features gpu o --features neuro-trt".into())
}

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    // Codigos de salida de la competicion SAT: 10 satisfacible, 20 insatisfacible.
    match dispatch(&arguments) {
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

    // Estas pruebas miran otras opciones; la obligacion de verificar tiene la suya.
    fn parse_options(arguments: &[String]) -> Result<Options, String> {
        let mut unverified = arguments.to_vec();
        unverified.push("--no-verify".into());
        super::parse_options(&unverified)
    }

    #[test]
    fn verification_is_the_default_and_opting_out_is_explicit() {
        let arguments = |text: &str| -> Vec<String> { text.split_whitespace().map(str::to_owned).collect() };
        assert!(super::parse_options(&arguments("f.cnf")).unwrap().verify);
        assert!(!super::parse_options(&arguments("f.cnf --no-verify")).unwrap().verify);
        assert!(!super::parse_options(&arguments("f.cnf --preprocess-only")).unwrap().verify);
        assert!(super::parse_options(&arguments("f.cnf --workers 4")).is_ok());
        assert!(super::parse_options(&arguments("f.cnf --workers 4 --recube-conflicts 100")).is_err());
        assert!(super::parse_options(&arguments("f.cnf --ramsey-symm 6")).is_err());
    }

    #[test]
    fn every_row_has_the_columns_of_the_header() {
        let columns = HEADER.split(',').count();
        let prefix = "label,serial,1,0,false,true,true,16,500,10,20";
        assert_eq!(row(prefix, "UNSAT", 1.0, 0.5, &Stats::default()).split(',').count(), columns);
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
    fn the_swarm_can_be_asked_for_a_trace() {
        let arguments: Vec<String> = ["f.cnf", "--workers", "4", "--proof", "p.pbp"].iter().map(|text| text.to_string()).collect();
        assert!(parse_options(&arguments).is_ok());
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
        // Sin el --no-verify que añade el envoltorio, que se tomaría por el valor que falta.
        assert!(super::parse_options(&args("f.cnf --workers 2 --deep-trail")).is_err());
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
