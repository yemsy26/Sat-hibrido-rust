//! API de integracion: una sola llamada que resuelve y comprueba, sin imprimir nada ni terminar
//! el proceso. Es lo que usa el comando `motor` y lo que debe usar otro programa; el resto de
//! modulos de la biblioteca son internos y pueden cambiar.
//!
//! ```
//! use motor_core::solver::{solve, Options, Problem, Status};
//!
//! let problem = Problem::dimacs("p cnf 2 2\n1 2 0\n-1 0\n").unwrap();
//! // `certify` esta activo por defecto y un UNSAT necesita entonces VeriPB en el PATH.
//! let options = Options { certify: false, ..Options::default() };
//! let solution = solve(&problem, &options).unwrap();
//! assert_eq!(solution.status, Status::Sat);
//! assert_eq!(solution.model, Some(vec![-1, 2]));
//! assert!(solution.model_checked);
//! ```
//!
//! Lo que garantiza. `Status::Sat` y `Status::Optimum` traen un modelo evaluado contra el problema.
//! `Status::Unsat` y `Status::Optimum` con `proof_checked` los ha aceptado VeriPB; sin
//! `certify`, `proof_checked` es falso y hay que creer al solver. Una parada o una prueba
//! rechazada nunca salen como `Unsat`. En un `.opb` VeriPB comprueba la CNF traducida: el
//! traductor queda dentro de lo que hay que creer.
//!
//! Se pueden lanzar varias resoluciones a la vez, cada una con sus hilos y sus ficheros.
use crate::dimacs::{parse_dimacs, write_dimacs, Formula};
use crate::hybrid::cdcl::{CubeFilter, RecubeConfig, SwarmConfig, SwarmProgress};
use crate::hybrid::proof::PbProof;
use crate::hybrid::{self, verify_model, HybridConfig, CongruenceStats, ParityStats, SolveStats, SymmetryStats};
use crate::opb;
use crate::verifier::{self, ProofCheck};
use chronos_vfs::{RingWriter, ZstdConfig, ZstdSink};
use std::fmt::Write as _;
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

// La traza que hay que verificar no puede llenar el disco sin que nadie lo haya pedido.
const DEFAULT_PROOF_CAP: u64 = 64 << 30;
const TRACE_RING_BYTES: usize = 64 << 20;
const WATCH_POLL: Duration = Duration::from_millis(20);

/// Por que no se pudo ni intentar: el resultado de un intento fallido es una [`Solution`].
#[derive(Clone, Debug, PartialEq)]
pub enum SolveError {
    /// Hace falta VeriPB y no se puede ejecutar.
    Verifier(String),
    /// El problema o las opciones no son validos.
    Input(String),
    /// Fallo al leer o escribir un fichero.
    Io(String),
    /// Fallo interno del motor; el mensaje es el de su panico.
    Internal(String),
}
impl std::fmt::Display for SolveError {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (Self::Verifier(message) | Self::Input(message) | Self::Io(message) | Self::Internal(message)) = self;
        out.write_str(message)
    }
}
impl std::error::Error for SolveError {}

fn at_path(path: &Path) -> impl Fn(io::Error) -> SolveError + '_ {
    move |error| SolveError::Io(format!("{}: {error}", path.display()))
}

/// Pide la parada desde cualquier hilo. Se puede clonar y compartir.
#[derive(Clone, Debug, Default)]
pub struct StopToken(Arc<AtomicBool>);
impl StopToken {
    pub fn stop(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_stopped(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    /// Deja el testigo listo para reutilizarlo.
    pub fn clear(&self) {
        self.0.store(false, Ordering::Release);
    }
}

#[derive(Clone, Debug)]
enum Instance {
    Cnf(Formula),
    Pb(opb::Problem),
}

/// Lo que hay que resolver. Se construye una vez y se puede resolver varias veces.
#[derive(Clone, Debug)]
pub struct Problem {
    instance: Instance,
    // VeriPB lee el fichero original, no una copia escrita por este programa.
    file: Option<PathBuf>,
}
impl Problem {
    /// Clausulas con literales DIMACS: `-3` es la variable 3 negada.
    pub fn cnf(formula: Formula) -> Result<Self, SolveError> {
        if let Some(&bad) = formula.iter().flatten().find(|&&literal| literal == 0 || literal == i32::MIN) {
            return Err(SolveError::Input(format!("Literal no valido: {bad}")));
        }
        Ok(Self { instance: Instance::Cnf(formula), file: None })
    }
    /// Restricciones lineales y objetivo opcional.
    pub fn pb(problem: opb::Problem) -> Result<Self, SolveError> {
        let objective = problem.objective.iter().flatten();
        let terms = problem.constraints.iter().flat_map(|constraint| &constraint.terms).chain(objective);
        for &(_, literal) in terms {
            if literal == 0 || literal.unsigned_abs() > problem.variables {
                return Err(SolveError::Input(format!("Literal {literal} fuera de las {} variables", problem.variables)));
            }
        }
        Ok(Self { instance: Instance::Pb(problem), file: None })
    }
    /// Texto en formato DIMACS CNF.
    pub fn dimacs(text: &str) -> Result<Self, SolveError> {
        Self::cnf(parse_dimacs(text).map_err(SolveError::Input)?)
    }
    /// Texto en formato OPB.
    pub fn opb(text: &str) -> Result<Self, SolveError> {
        Self::pb(opb::parse(text).map_err(SolveError::Input)?)
    }
    /// Lee un fichero: OPB si termina en `.opb`, DIMACS en cualquier otro caso.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, SolveError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(at_path(path))?;
        let opb = path.extension().is_some_and(|extension| extension == "opb");
        let parsed = if opb { Self::opb(&text) } else { Self::dimacs(&text) };
        let mut problem = parsed.map_err(|error| SolveError::Input(format!("{}: {error}", path.display())))?;
        problem.file = Some(path.to_owned());
        Ok(problem)
    }
    /// Las clausulas, si el problema es CNF.
    pub fn formula(&self) -> Option<&Formula> {
        match &self.instance {
            Instance::Cnf(formula) => Some(formula),
            Instance::Pb(_) => None,
        }
    }
    pub fn variables(&self) -> u32 {
        match &self.instance {
            Instance::Cnf(formula) => last_variable(formula),
            Instance::Pb(problem) => problem.variables,
        }
    }
    /// Clausulas o restricciones.
    pub fn constraints(&self) -> usize {
        match &self.instance {
            Instance::Cnf(formula) => formula.len(),
            Instance::Pb(problem) => problem.constraints.len(),
        }
    }
}

fn last_variable(formula: &Formula) -> u32 {
    formula.iter().flatten().map(|literal| literal.unsigned_abs()).max().unwrap_or(0)
}

/// Avance de la busqueda, una instantanea aproximada.
#[derive(Clone, Copy, Debug)]
pub struct Progress {
    pub elapsed: Duration,
    pub conflicts: u64,
    pub restarts: u64,
    pub lbd_sum: u64,
    pub cubes_closed: u64,
    pub cubes_total: u64,
    pub trace_bytes: u64,
    pub trace_lines: u64,
    pub trace_truncated: bool,
    /// Ya se ha pedido la parada, por tiempo o desde fuera.
    pub stopping: bool,
}

#[derive(Clone, Copy, Debug)]
pub enum Event {
    Progress(Progress),
    /// Una solucion mejor de un `.opb` con objetivo, ya comprobada.
    Improved { cost: i128 },
}
impl Event {
    pub fn to_json(&self) -> String {
        match self {
            Self::Improved { cost } => format!("{{\"event\":\"improved\",\"cost\":{cost}}}"),
            Self::Progress(progress) => format!(
                "{{\"event\":\"progress\",\"elapsed_ms\":{:.3},\"conflicts\":{},\"restarts\":{},\"cubes_closed\":{},\"cubes_total\":{},\"trace_bytes\":{},\"trace_truncated\":{},\"stopping\":{}}}",
                progress.elapsed.as_secs_f64() * 1000.0, progress.conflicts, progress.restarts, progress.cubes_closed,
                progress.cubes_total, progress.trace_bytes, progress.trace_truncated, progress.stopping,
            ),
        }
    }
}

/// Recibe los avisos desde otro hilo. Mientras no vuelve, no se atiende el limite de tiempo.
pub type EventHandler = Arc<dyn Fn(&Event) + Send + Sync>;

#[derive(Clone)]
pub struct Options {
    /// `None`: un hilo. `Some(k)`: enjambre de k hilos con una sola traza.
    pub workers: Option<usize>,
    /// Cubos por hilo del enjambre.
    pub cubes: usize,
    /// Los hilos del enjambre comparten clausulas.
    pub share: bool,
    /// Ruptura de simetrias.
    pub symmetry: bool,
    /// Capa de paridad.
    pub parity: bool,
    pub congruence: bool,
    /// Flips por variable del oraculo de busqueda local.
    pub flips: usize,
    /// Conflictos entre re-igniciones del oraculo; 0 lo apaga.
    pub interval: usize,
    /// Semilla del oraculo.
    pub seed: Option<u64>,
    /// Comprobar cada UNSAT y cada optimo con VeriPB. Sin VeriPB, `solve` falla antes de buscar.
    pub certify: bool,
    /// Conserva la traza; con la extension `.zst` se comprime al vuelo.
    pub proof: Option<PathBuf>,
    /// En un `.opb`, conserva la ultima CNF traducida.
    pub cnf: Option<PathBuf>,
    /// Tope de la traza en bytes. Superado, un UNSAT no queda certificado y se descarta.
    /// Por defecto 64 GiB al certificar y sin tope en otro caso.
    pub proof_cap: Option<u64>,
    /// Tiempo maximo, comprobacion incluida.
    pub timeout: Option<Duration>,
    /// Permite parar desde otro hilo.
    pub stop: Option<StopToken>,
    /// Recibe el avance y las mejoras.
    pub on_event: Option<EventHandler>,
    /// Cada cuanto se avisa del avance; sin valor, solo al pedir la parada.
    pub progress_interval: Option<Duration>,
    /// Solo las capas de simetria y paridad, sin busqueda: el resultado es `Status::Unknown`.
    pub preprocess_only: bool,
}
impl Default for Options {
    fn default() -> Self {
        let hybrid = HybridConfig::default();
        Self {
            workers: None, cubes: SwarmConfig::default().cubes_per_worker, share: true, symmetry: hybrid.symmetry,
            parity: hybrid.parity, congruence: hybrid.congruence, flips: hybrid.oracle.flips_per_variable, interval: hybrid.reignition_interval,
            seed: None, certify: true, proof: None, cnf: None, proof_cap: None, timeout: None, stop: None,
            on_event: None, progress_interval: None, preprocess_only: false,
        }
    }
}

/// Piezas experimentales del comando `motor`: prefiltro GPU y recubing. No son parte de la API
/// estable y ninguna de ellas admite un `.opb`.
#[derive(Default)]
pub struct Extras<'a> {
    pub filter: Option<CubeFilter<'a>>,
    pub recube: Option<RecubeConfig>,
    pub progress: Option<Arc<SwarmProgress>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Sat,
    Unsat,
    Optimum,
    Unknown,
}
impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sat => "SAT",
            Self::Unsat => "UNSAT",
            Self::Optimum => "OPTIMUM",
            Self::Unknown => "UNKNOWN",
        }
    }
}

/// Por que termino antes de tener la respuesta completa.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Halt {
    Timeout,
    Cancelled,
    /// El resultado se descarto: el modelo no era valido o la prueba no se acepto.
    Rejected(String),
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub variables: u32,
    pub clauses: usize,
    /// Veces que se llamo al solver; mas de una al optimizar.
    pub passes: u32,
    pub symmetry: SymmetryStats,
    pub parity: ParityStats,
    pub congruence: CongruenceStats,
    pub search: SolveStats,
    pub search_ms: f64,
    pub cubes_generated: usize,
    pub cubes_started: usize,
    pub cubes_unsat: usize,
    pub cubes_cancelled: usize,
    pub cubes_yielded: usize,
    pub cubes_split: usize,
    pub trace_bytes: u64,
    pub trace_lines: u64,
    pub trace_dominances: u64,
    pub trace_deletions: u64,
    pub trace_truncated: bool,
}

#[derive(Clone, Debug)]
pub struct Solution {
    pub status: Status,
    /// Una entrada por variable, con su signo. Con `Status::Sat` y `Halt::Timeout`, en un `.opb`,
    /// es la mejor solucion encontrada, sin demostrar que sea optima.
    pub model: Option<Vec<i32>>,
    pub cost: Option<i128>,
    /// El modelo se evaluo contra el problema.
    pub model_checked: bool,
    /// VeriPB acepto la prueba de UNSAT o de optimalidad.
    pub proof_checked: bool,
    pub halt: Option<Halt>,
    /// Traza conservada, si se pidio y el resultado es definitivo.
    pub proof: Option<PathBuf>,
    pub elapsed: Duration,
    pub stats: Stats,
}
impl Solution {
    /// Codigo de salida del comando: 10 SAT, 20 UNSAT, 30 optimo, 124 parada, 1 descartado.
    pub fn exit_code(&self) -> i32 {
        match (&self.halt, self.status) {
            (Some(Halt::Rejected(_)), _) => 1,
            (Some(_), _) => 124,
            (None, Status::Sat) => 10,
            (None, Status::Unsat) => 20,
            (None, Status::Optimum) => 30,
            (None, Status::Unknown) => 0,
        }
    }
    /// El resultado como un objeto JSON en una sola linea.
    pub fn to_json(&self) -> String {
        let mut out = String::with_capacity(512);
        let _ = write!(out, "{{\"status\":\"{}\",\"model\":", self.status.as_str());
        match &self.model {
            Some(model) => {
                out.push('[');
                for (index, literal) in model.iter().enumerate() {
                    let _ = write!(out, "{}{literal}", if index == 0 { "" } else { "," });
                }
                out.push(']');
            }
            None => out.push_str("null"),
        }
        match self.cost {
            Some(cost) => { let _ = write!(out, ",\"cost\":{cost}"); }
            None => out.push_str(",\"cost\":null"),
        }
        let (reason, detail) = match &self.halt {
            None => ("null".to_owned(), "null".to_owned()),
            Some(Halt::Timeout) => ("\"timeout\"".to_owned(), "null".to_owned()),
            Some(Halt::Cancelled) => ("\"cancelled\"".to_owned(), "null".to_owned()),
            Some(Halt::Rejected(why)) => ("\"rejected\"".to_owned(), json_string(why)),
        };
        let proof = self.proof.as_ref().map_or("null".to_owned(), |path| json_string(&path.to_string_lossy()));
        let stats = &self.stats;
        let _ = write!(
            out,
            ",\"model_checked\":{},\"proof_checked\":{},\"reason\":{reason},\"detail\":{detail},\"proof\":{proof},\"elapsed_ms\":{:.3},\"stats\":{{\"variables\":{},\"clauses\":{},\"passes\":{},\"conflicts\":{},\"decisions\":{},\"restarts\":{},\"symmetry_generators\":{},\"parity_equations\":{},\"congruence_equivalences\":{},\"congruence_units\":{},\"cubes_generated\":{},\"cubes_unsat\":{},\"trace_bytes\":{},\"trace_truncated\":{}}}}}",
            self.model_checked, self.proof_checked, self.elapsed.as_secs_f64() * 1000.0,
            stats.variables, stats.clauses, stats.passes, stats.search.conflicts, stats.search.decisions, stats.search.restarts,
            stats.symmetry.generators, stats.parity.equations, stats.congruence.equivalences, stats.congruence.units, stats.cubes_generated, stats.cubes_unsat, stats.trace_bytes, stats.trace_truncated,
        );
        out
    }
}

/// Una cadena como literal JSON, comillas incluidas.
pub fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if u32::from(control) < 0x20 => { let _ = write!(out, "\\u{:04x}", u32::from(control)); }
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

// Lo que el vigilante puede ver de la traza mientras el motor trabaja.
#[derive(Default)]
struct Gauge {
    bytes: AtomicU64,
    lines: AtomicU64,
    truncated: AtomicBool,
}
impl Gauge {
    fn reset(&self) {
        self.bytes.store(0, Ordering::Relaxed);
        self.lines.store(0, Ordering::Relaxed);
        self.truncated.store(false, Ordering::Relaxed);
    }
}

// Destino de la traza: el fichero tal cual o, si termina en .zst, el anillo y su compresor.
enum TraceFile {
    Plain(File),
    Compressed(RingWriter, ZstdSink),
}
impl TraceFile {
    fn create(path: &Path) -> io::Result<Self> {
        if path.extension().is_some_and(|extension| extension == "zst") {
            let (ring, sink) = chronos_vfs::zstd_bridge(path, TRACE_RING_BYTES, ZstdConfig::default())?;
            return Ok(Self::Compressed(ring, sink));
        }
        File::create(path).map(Self::Plain)
    }
    // Sin esperar al compresor, el fichero podria estar a medias al pasarselo a VeriPB.
    fn close(self) -> io::Result<()> {
        match self {
            Self::Plain(mut file) => file.flush(),
            Self::Compressed(ring, sink) => {
                ring.close();
                sink.join().map(|_| ())
            }
        }
    }
}
impl Write for TraceFile {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        match self {
            Self::Plain(file) => file.write(buffer),
            Self::Compressed(ring, _) => ring.write(buffer),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Plain(file) => file.flush(),
            Self::Compressed(ring, _) => ring.flush(),
        }
    }
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

// Carpeta propia de cada resolucion, para que varias a la vez no se pisen. Se borra al salir.
struct Workspace(PathBuf);
impl Workspace {
    fn create() -> Result<Self, SolveError> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let folder = std::env::temp_dir().join(format!("motor-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&folder).map_err(at_path(&folder))?;
        Ok(Self(folder))
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const CAUSE_NONE: u8 = 0;
const CAUSE_TIMEOUT: u8 = 1;
const CAUSE_CANCELLED: u8 = 2;

struct Context<'r> {
    options: &'r Options,
    certify: bool,
    cap: u64,
    progress: &'r SwarmProgress,
    gauge: &'r Arc<Gauge>,
    cause: &'r AtomicU8,
    started: Instant,
}
impl Context<'_> {
    fn stopping(&self) -> bool {
        self.progress.stop.load(Ordering::Acquire)
    }
    fn halt(&self) -> Halt {
        if self.cause.load(Ordering::Acquire) == CAUSE_TIMEOUT { Halt::Timeout } else { Halt::Cancelled }
    }
    fn config(&self) -> HybridConfig {
        let mut config = HybridConfig { symmetry: self.options.symmetry, parity: self.options.parity, congruence: self.options.congruence, ..HybridConfig::default() };
        config.oracle.flips_per_variable = self.options.flips;
        config.reignition_interval = self.options.interval;
        if let Some(seed) = self.options.seed {
            config.oracle.seed = seed;
        }
        config
    }
    fn open_trace(&self, path: &Path, clauses: usize) -> Result<PbProof<Meter<TraceFile>>, SolveError> {
        let sink = TraceFile::create(path).map_err(at_path(path))?;
        Ok(PbProof::new(Meter { sink, gauge: Arc::clone(self.gauge), cap: self.cap }, clauses))
    }
    fn finish(&self, mut stats: Stats, status: Status, halt: Option<Halt>) -> Solution {
        stats.trace_bytes = self.gauge.bytes.load(Ordering::Relaxed);
        stats.trace_lines = self.gauge.lines.load(Ordering::Relaxed);
        stats.trace_truncated = self.gauge.truncated.load(Ordering::Relaxed);
        Solution {
            status, model: None, cost: None, model_checked: false, proof_checked: false, halt,
            proof: None, elapsed: self.started.elapsed(), stats,
        }
    }
}

/// Resuelve `problem` y comprueba el resultado. Un `Err` significa que no se pudo intentar;
/// un intento que no llega a una respuesta devuelve `Status::Unknown` y su [`Halt`].
pub fn solve(problem: &Problem, options: &Options) -> Result<Solution, SolveError> {
    solve_with(problem, options, Extras::default())
}

/// [`solve`] con las piezas experimentales de [`Extras`].
pub fn solve_with(problem: &Problem, options: &Options, extras: Extras<'_>) -> Result<Solution, SolveError> {
    // Una afirmacion interna rota no debe tirar el programa que nos ha integrado.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(problem, options, extras))).unwrap_or_else(|panic| {
        let message = panic.downcast_ref::<String>().map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied()).unwrap_or("panico sin mensaje");
        Err(SolveError::Internal(format!("Fallo interno del motor: {message}")))
    })
}

fn run(problem: &Problem, options: &Options, mut extras: Extras<'_>) -> Result<Solution, SolveError> {
    let started = Instant::now();
    let certify = options.certify && !options.preprocess_only;
    let invalid = |message: &str| Err(SolveError::Input(message.to_owned()));
    if extras.recube.is_some() && certify {
        return invalid("recubing no emite traza y su UNSAT no se puede verificar");
    }
    if extras.recube.is_some() && options.proof.is_some() {
        return invalid("recubing no emite traza: no admite proof");
    }
    if (extras.recube.is_some() || extras.filter.is_some()) && options.workers.is_none() {
        return invalid("recubing y el prefiltro exigen workers");
    }
    if options.preprocess_only && matches!(problem.instance, Instance::Pb(_)) {
        return invalid("preprocess_only solo admite CNF");
    }
    if certify {
        verifier::ensure_available().map_err(SolveError::Verifier)?;
    }
    let progress = extras.progress.take().unwrap_or_default();
    let gauge = Arc::new(Gauge::default());
    let cause = AtomicU8::new(CAUSE_NONE);
    let cap = options.proof_cap.unwrap_or(if certify { DEFAULT_PROOF_CAP } else { u64::MAX });
    let context = Context { options, certify, cap, progress: &progress, gauge: &gauge, cause: &cause, started };
    std::thread::scope(|scope| {
        let (done, finished) = mpsc::channel::<()>();
        let watched = options.timeout.is_some() || options.stop.is_some() || options.on_event.is_some();
        if watched {
            let context = &context;
            scope.spawn(move || watch(context, &finished));
        }
        let result = match &problem.instance {
            Instance::Cnf(formula) => run_cnf(formula, problem.file.as_deref(), &context, extras),
            Instance::Pb(instance) if extras.filter.is_none() && extras.recube.is_none() => run_pb(instance, &context),
            Instance::Pb(_) => invalid("recubing y el prefiltro no admiten un .opb"),
        };
        drop(done);
        result
    })
}

// Vigilante: limite de tiempo, parada pedida desde fuera y avisos de avance. No toca el motor:
// solo levanta `progress.stop`, una y otra vez, porque el enjambre lo baja si acaba con modelo.
fn watch(context: &Context<'_>, finished: &mpsc::Receiver<()>) {
    let options = context.options;
    let mut next = options.progress_interval;
    let mut announced = false;
    loop {
        let elapsed = context.started.elapsed();
        let expired = options.timeout.is_some_and(|limit| elapsed >= limit);
        let cancelled = options.stop.as_ref().is_some_and(StopToken::is_stopped);
        if expired || cancelled {
            let cause = if expired { CAUSE_TIMEOUT } else { CAUSE_CANCELLED };
            let _ = context.cause.compare_exchange(CAUSE_NONE, cause, Ordering::AcqRel, Ordering::Acquire);
            context.progress.stop.store(true, Ordering::Release);
        }
        let stopping = expired || cancelled;
        let beat = next.is_some_and(|due| elapsed >= due);
        if beat {
            next = next.zip(options.progress_interval).map(|(due, step)| due + step);
        }
        if let Some(handler) = options.on_event.as_ref().filter(|_| beat || (stopping && !announced)) {
            let (observer, gauge) = (context.progress, context.gauge);
            handler(&Event::Progress(Progress {
                elapsed,
                conflicts: observer.conflicts.load(Ordering::Relaxed),
                restarts: observer.restarts.load(Ordering::Relaxed),
                lbd_sum: observer.lbd_sum.load(Ordering::Relaxed),
                cubes_closed: observer.cubes_closed.load(Ordering::Relaxed),
                cubes_total: observer.cubes_total.load(Ordering::Relaxed),
                trace_bytes: gauge.bytes.load(Ordering::Relaxed),
                trace_lines: gauge.lines.load(Ordering::Relaxed),
                trace_truncated: gauge.truncated.load(Ordering::Relaxed),
                stopping,
            }));
        }
        announced |= stopping;
        if !matches!(finished.recv_timeout(WATCH_POLL), Err(mpsc::RecvTimeoutError::Timeout)) {
            return;
        }
    }
}

struct Searched {
    model: Option<Vec<i32>>,
    stopped: bool,
}

// Una llamada al solver, en serie o con el enjambre, con traza si hay donde dejarla.
fn search(
    formula: &Formula, trace: Option<&Path>, context: &Context<'_>, filter: Option<CubeFilter<'_>>,
    recube: Option<RecubeConfig>, stats: &mut Stats,
) -> Result<Searched, SolveError> {
    let (options, config, observer) = (context.options, context.config(), context.progress);
    stats.passes += 1;
    if let Some(workers) = options.workers {
        let swarm = SwarmConfig {
            workers, cubes_per_worker: options.cubes, share_enabled: options.share, ..SwarmConfig::default()
        };
        // La traza se abre antes del enjambre y se cierra despues: el compresor tiene que acabar.
        let report = if let Some(path) = trace {
            let mut proof = context.open_trace(path, formula.len())?;
            let report = hybrid::solve_hybrid_parallel_with_proof(formula, config, swarm, Some(observer), filter, &mut proof);
            stats.trace_dominances += proof.dominances();
            stats.trace_deletions += proof.deletions();
            proof.finish().map_err(at_path(path))?.sink.close().map_err(at_path(path))?;
            report
        } else if let Some(recube) = recube {
            hybrid::solve_hybrid_parallel_with_recubing(formula, config, swarm, recube, Some(observer), filter)
        } else {
            hybrid::solve_hybrid_parallel_with_filter(formula, config, swarm, Some(observer), filter)
        };
        let swarm = &report.swarm;
        (stats.symmetry, stats.parity, stats.congruence) = (report.symmetry, report.parity, report.congruence);
        stats.search.add_search(swarm.total);
        stats.search_ms += report.conquer_ms;
        stats.cubes_generated += swarm.cubes_generated;
        stats.cubes_started += swarm.cubes_started;
        stats.cubes_unsat += swarm.cubes_unsat;
        stats.cubes_cancelled += swarm.cubes_cancelled;
        stats.cubes_yielded += swarm.cubes_yielded;
        stats.cubes_split += swarm.cubes_split;
        return Ok(Searched { stopped: swarm.stopped, model: report.model });
    }
    let report = if let Some(path) = trace {
        let mut proof = context.open_trace(path, formula.len())?;
        let report = hybrid::solve_hybrid_observed(formula, config, Some(&mut proof), Some(observer));
        stats.trace_dominances += proof.dominances();
        stats.trace_deletions += proof.deletions();
        proof.finish().map_err(at_path(path))?.sink.close().map_err(at_path(path))?;
        report
    } else {
        hybrid::solve_hybrid_observed(formula, config, None, Some(observer))
    };
    (stats.symmetry, stats.parity, stats.congruence) = (report.symmetry, report.parity, report.congruence);
    stats.search.add_search(report.cdcl);
    stats.search_ms += report.cdcl_ms;
    Ok(Searched { model: report.model, stopped: report.stopped })
}

// Las dos capas previas a la busqueda, cada una sobre la formula original.
fn preprocess_only(formula: &Formula, context: &Context<'_>, mut stats: Stats) -> Result<Solution, SolveError> {
    let options = context.options;
    if options.symmetry {
        let start = Instant::now();
        let breaking = match &options.proof {
            Some(path) => {
                let mut proof = context.open_trace(path, formula.len())?;
                let breaking = hybrid::symmetry::break_symmetries_with_proof(formula, Some(&mut proof));
                stats.trace_dominances = proof.dominances();
                stats.trace_deletions = proof.deletions();
                proof.finish().map_err(at_path(path))?.sink.close().map_err(at_path(path))?;
                breaking
            }
            None => hybrid::symmetry::break_symmetries(formula),
        };
        stats.symmetry = SymmetryStats {
            generators: breaking.generators.len(), clauses: breaking.clauses.len(),
            ms: start.elapsed().as_secs_f64() * 1000.0,
        };
    }
    if options.parity {
        let start = Instant::now();
        stats.parity.equations = hybrid::parity::reduce_parity(formula).equations;
        stats.parity.ms = start.elapsed().as_secs_f64() * 1000.0;
    }
    Ok(context.finish(stats, Status::Unknown, None))
}

fn run_cnf(
    formula: &Formula, file: Option<&Path>, context: &Context<'_>, extras: Extras<'_>,
) -> Result<Solution, SolveError> {
    let options = context.options;
    let variables = last_variable(formula);
    let mut stats = Stats { variables, clauses: formula.len(), ..Stats::default() };
    if options.preprocess_only {
        return preprocess_only(formula, context, stats);
    }
    let workspace = if context.certify { Some(Workspace::create()?) } else { None };
    let temporary = workspace.as_ref().filter(|_| options.proof.is_none()).map(|workspace| workspace.0.join("f.pbp"));
    let trace = options.proof.as_deref().or(temporary.as_deref());
    let searched = search(formula, trace, context, extras.filter, extras.recube, &mut stats)?;
    if let Some(model) = searched.model {
        if !verify_model(formula, &model) {
            let halt = Halt::Rejected("el modelo no satisface la formula del fichero".into());
            return Ok(context.finish(stats, Status::Unknown, Some(halt)));
        }
        let mut solution = context.finish(stats, Status::Sat, None);
        solution.model = Some(model);
        solution.model_checked = true;
        solution.proof = options.proof.clone();
        return Ok(solution);
    }
    if searched.stopped {
        return Ok(context.finish(stats, Status::Unknown, Some(context.halt())));
    }
    let (Some(workspace), Some(trace)) = (&workspace, trace) else {
        let mut solution = context.finish(stats, Status::Unsat, None);
        solution.proof = options.proof.clone();
        return Ok(solution);
    };
    let written = workspace.0.join("f.cnf");
    let cnf = match file {
        Some(file) => file,
        None => {
            write_dimacs(&written, variables, formula).map_err(at_path(&written))?;
            &written
        }
    };
    let verdict = verify(cnf, trace, context)?;
    let mut solution = match verdict {
        None => context.finish(stats, Status::Unsat, None),
        Some(halt) => context.finish(stats, Status::Unknown, Some(halt)),
    };
    solution.proof_checked = solution.status == Status::Unsat;
    if solution.proof_checked {
        solution.proof = options.proof.clone();
    }
    Ok(solution)
}

// Juzga la traza desde fuera de la busqueda. `None`: VeriPB la acepta.
fn verify(cnf: &Path, trace: &Path, context: &Context<'_>) -> Result<Option<Halt>, SolveError> {
    if context.gauge.truncated.load(Ordering::Relaxed) {
        return Ok(Some(Halt::Rejected("la traza llego a su tope y no certifica nada; sube proof_cap (--proof-cap-gib)".into())));
    }
    Ok(match verifier::check_unsat_proof(cnf, trace, Some(&context.progress.stop)).map_err(SolveError::Verifier)? {
        ProofCheck::Verified => None,
        ProofCheck::Rejected(reason) => Some(Halt::Rejected(format!("VeriPB rechazo la prueba ({reason})"))),
        ProofCheck::Stopped => Some(context.halt()),
    })
}

fn signed(problem: &opb::Problem, model: &[bool]) -> Vec<i32> {
    (1..=problem.variables as i32).map(|id| if model.get(id as usize).copied().unwrap_or(false) { id } else { -id }).collect()
}

fn run_pb(problem: &opb::Problem, context: &Context<'_>) -> Result<Solution, SolveError> {
    let options = context.options;
    let mut stats = Stats { variables: problem.variables, clauses: problem.constraints.len(), ..Stats::default() };
    let workspace = Workspace::create()?;
    let cnf = workspace.0.join("f.cnf");
    let compressed = options.proof.as_ref().is_some_and(|path| path.extension().is_some_and(|extension| extension == "zst"));
    let trace = workspace.0.join(if compressed { "f.pbp.zst" } else { "f.pbp" });
    let wants_trace = context.certify || options.proof.is_some();
    let mut bound: Option<i128> = None;
    let mut best: Option<(Vec<i32>, i128)> = None;
    // Una solucion comprobada sigue siendo valida; lo que puede faltar es probar que es optima.
    let finish = |stats: Stats, best: Option<(Vec<i32>, i128)>, halt: Halt, objective: bool| {
        let mut solution = context.finish(stats, if best.is_some() { Status::Sat } else { Status::Unknown }, Some(halt));
        if let Some((model, cost)) = best {
            solution.model = Some(model);
            solution.cost = objective.then_some(cost);
            solution.model_checked = true;
        }
        solution
    };
    let objective = problem.objective.is_some();
    let keep = |solution: &mut Solution| -> Result<(), SolveError> {
        if let Some(kept) = &options.cnf {
            std::fs::copy(&cnf, kept).map_err(at_path(kept))?;
        }
        if let Some(kept) = options.proof.as_ref().filter(|_| wants_trace) {
            std::fs::copy(&trace, kept).map_err(at_path(kept))?;
            solution.proof = Some(kept.clone());
        }
        Ok(())
    };
    let refuted = loop {
        if context.stopping() {
            return Ok(finish(stats, best, context.halt(), objective));
        }
        let (formula, variables) = opb::encode(problem, bound).map_err(SolveError::Input)?;
        write_dimacs(&cnf, variables, &formula).map_err(at_path(&cnf))?;
        context.gauge.reset();
        context.progress.cubes_closed.store(0, Ordering::Relaxed);
        let searched = search(&formula, wants_trace.then_some(trace.as_path()), context, None, None, &mut stats)?;
        let Some(literals) = searched.model else {
            if searched.stopped {
                return Ok(finish(stats, best, context.halt(), objective));
            }
            break true;
        };
        let size = literals.iter().map(|literal| literal.unsigned_abs() as usize).max().unwrap_or(0);
        let mut model = vec![false; size.max(problem.variables as usize) + 1];
        for literal in literals {
            model[literal.unsigned_abs() as usize] = literal > 0;
        }
        if !opb::satisfies(problem, &model) {
            let halt = Halt::Rejected("el modelo no cumple las restricciones del fichero".into());
            return Ok(context.finish(stats, Status::Unknown, Some(halt)));
        }
        let Some(cost_terms) = &problem.objective else {
            best = Some((signed(problem, &model), 0));
            break false;
        };
        let cost = opb::evaluate(cost_terms, &model);
        if bound.is_some_and(|bound| cost > bound) {
            let halt = Halt::Rejected("el modelo no mejora el coste exigido".into());
            return Ok(context.finish(stats, Status::Unknown, Some(halt)));
        }
        if let Some(handler) = &options.on_event {
            handler(&Event::Improved { cost });
        }
        bound = Some(cost - 1);
        best = Some((signed(problem, &model), cost));
    };
    if refuted && context.certify {
        if let Some(halt) = verify(&cnf, &trace, context)? {
            return Ok(finish(stats, best, halt, objective));
        }
    }
    let status = match (&best, refuted) {
        (None, _) => Status::Unsat,
        (Some(_), false) => Status::Sat,
        (Some(_), true) => Status::Optimum,
    };
    let mut solution = context.finish(stats, status, None);
    solution.proof_checked = refuted && context.certify;
    keep(&mut solution)?;
    if let Some((model, cost)) = best {
        solution.model = Some(model);
        solution.cost = objective.then_some(cost);
        solution.model_checked = true;
    }
    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trace_past_its_cap_stops_being_written() {
        let gauge = Arc::new(Gauge::default());
        let mut meter = Meter { sink: Vec::new(), gauge: Arc::clone(&gauge), cap: 10 };
        meter.write_all(b"12345\n").unwrap();
        meter.write_all(b"678901\n").unwrap();
        meter.write_all(b"1\n").unwrap();
        assert_eq!(meter.sink, b"12345\n");
        assert!(gauge.truncated.load(Ordering::Relaxed));
        assert_eq!((gauge.bytes.load(Ordering::Relaxed), gauge.lines.load(Ordering::Relaxed)), (6, 1));
    }

    #[test]
    fn json_strings_are_escaped() {
        assert_eq!(json_string("a\"b\\c\n\u{1}ñ"), "\"a\\\"b\\\\c\\n\\u0001ñ\"");
    }
}
