//! Backend experimental exclusivo para dos colores de K43. No modifica CDCL.
//! Un presupuesto agotado da `Unknown`; un fragmento con K5 sólo refuta ese fragmento.
use std::fmt;
use std::time::Duration;
#[cfg(feature = "gpu")]
use std::time::Instant;

#[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
#[path = "ramsey43_risk.rs"]
mod risk;

#[cfg(feature = "neuro-dataset")]
#[path = "ramsey43_dataset.rs"]
pub mod dataset;

pub const VERTICES: usize = 43;
pub const EDGES: usize = VERTICES * (VERTICES - 1) / 2;
pub const ROW_WORDS: usize = 2 * VERTICES;
const VERTEX_MASK: u64 = (1_u64 << VERTICES) - 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Graph43 {
    rows: [u64; ROW_WORDS],
}

impl Default for Graph43 {
    fn default() -> Self {
        Self {
            rows: [0; ROW_WORDS],
        }
    }
}

impl Graph43 {
    /// `[red[43], blue[43]]`; una arista desconocida no pertenece a ningún color.
    pub fn from_rows(rows: [u64; ROW_WORDS]) -> Result<Self, Error> {
        let graph = Self { rows };
        graph.validate()?;
        Ok(graph)
    }
    pub fn rows(&self) -> &[u64; ROW_WORDS] {
        &self.rows
    }
    pub fn validate(&self) -> Result<(), Error> {
        for vertex in 0..VERTICES {
            let red = self.rows[vertex];
            let blue = self.rows[VERTICES + vertex];
            if (red | blue) & !VERTEX_MASK != 0
                || (red | blue) >> vertex & 1 != 0
                || red & blue != 0
            {
                return Err(Error::Input(format!(
                    "fila {vertex}: bits fuera de K43, diagonal o colores solapados"
                )));
            }
            for other in vertex + 1..VERTICES {
                for colour in 0..2 {
                    if (self.rows[colour * VERTICES + vertex] >> other & 1)
                        != (self.rows[colour * VERTICES + other] >> vertex & 1)
                    {
                        return Err(Error::Input(format!(
                            "matriz no simétrica en ({vertex},{other})"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
    /// Variables DIMACS 1..903 en orden (0,1), (0,2), ..., (41,42).
    pub fn assign(&mut self, literal: i32) -> Result<(), Error> {
        let id = literal.unsigned_abs() as usize;
        let (a, b) = edge_vertices(id).ok_or_else(|| Error::Input(format!("literal {literal}: se requieren exclusivamente aristas 1..903; no se admiten auxiliares")))?;
        let colour = usize::from(literal < 0);
        if self.rows[(1 - colour) * VERTICES + a] >> b & 1 != 0 {
            return Err(Error::Input(format!(
                "asignación contradictoria de la arista {id}"
            )));
        }
        self.rows[colour * VERTICES + a] |= 1_u64 << b;
        self.rows[colour * VERTICES + b] |= 1_u64 << a;
        Ok(())
    }
    pub fn free_edges(&self) -> usize {
        (0..VERTICES)
            .map(|vertex| {
                let assigned = self.rows[vertex] | self.rows[VERTICES + vertex];
                ((!assigned & VERTEX_MASK) & (u64::MAX << (vertex + 1))).count_ones() as usize
            })
            .sum()
    }
    pub fn model(&self) -> Result<Vec<i32>, Error> {
        self.validate()?;
        if self.free_edges() != 0 {
            return Err(Error::Input(
                "el grafo no contiene las 903 aristas asignadas".into(),
            ));
        }
        let mut model = Vec::with_capacity(EDGES);
        for a in 0..VERTICES {
            for b in a + 1..VERTICES {
                let id = edge_id(a, b).expect("arista interna válida") as i32;
                model.push(if self.rows[a] >> b & 1 != 0 { id } else { -id });
            }
        }
        Ok(model)
    }
    #[cfg(feature = "gpu")]
    fn extends(&self, base: &Self) -> bool {
        self.rows
            .iter()
            .zip(base.rows)
            .all(|(&row, fixed)| row & fixed == fixed)
    }
}

pub fn edge_id(a: usize, b: usize) -> Option<usize> {
    (a < b && b < VERTICES).then(|| a * (2 * VERTICES - a - 1) / 2 + b - a)
}

pub fn edge_vertices(id: usize) -> Option<(usize, usize)> {
    if !(1..=EDGES).contains(&id) {
        return None;
    }
    let mut first = 1;
    for a in 0..VERTICES - 1 {
        let count = VERTICES - a - 1;
        if id < first + count {
            return Some((a, a + 1 + id - first));
        }
        first += count;
    }
    None
}

/// Representantes de todas las soluciones K43 desde base vacía.
/// R(4,5)=25 limita ambos grados a 18..24; complemento y renombramiento
/// permiten fijar la estrella roja de la raíz a grado 18, 19, 20 o 21.
pub fn canonical_star(degree: u32) -> Result<Graph43, Error> {
    if !(18..=21).contains(&degree) {
        return Err(Error::Input(
            "grado canónico debe pertenecer a 18..21".into(),
        ));
    }
    let mut graph = Graph43::default();
    for id in 1..VERTICES as i32 {
        graph.assign(if id <= degree as i32 { id } else { -id })?;
    }
    Ok(graph)
}

/// Lee una asignación parcial DIMACS, no una CNF. Exige terminador 0 final.
/// Rechaza auxiliares, contradicciones y contenido después del terminador.
pub fn parse_partial_dimacs(text: &str) -> Result<Graph43, Error> {
    let mut graph = Graph43::default();
    let (mut seen, mut ended) = (false, false);
    for (line_number, line) in text.lines().enumerate() {
        let mut tokens = line.split_whitespace();
        let Some(first) = tokens.next() else { continue };
        if first == "c" || first == "s" {
            continue;
        }
        if first != "v" {
            return Err(Error::Input(format!(
                "línea {}: se esperaba una línea v de asignación, no una CNF",
                line_number + 1
            )));
        }
        if ended {
            return Err(Error::Input("literales después del terminador 0".into()));
        }
        seen = true;
        while let Some(token) = tokens.next() {
            let literal = token
                .parse::<i32>()
                .map_err(|_| Error::Input(format!("literal inválido: {token}")))?;
            if literal == 0 {
                if tokens.next().is_some() {
                    return Err(Error::Input(
                        "0 sólo puede aparecer al final de la asignación".into(),
                    ));
                }
                ended = true;
                break;
            }
            graph.assign(literal)?;
        }
    }
    if !seen || !ended {
        return Err(Error::Input("falta una asignación v terminada en 0".into()));
    }
    Ok(graph)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckResult {
    Open,
    Bad { blue: bool, vertices: [usize; 5] },
    CompleteSafe,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Sat(Graph43),
    Unsat,
    Unknown,
}

#[derive(Clone, Copy, Debug)]
pub struct SearchConfig {
    pub prefix_bits: u32,
    pub order_seed: u32,
    pub steps: u32,
    /// 0 conserva la búsqueda residente hasta SAT/UNSAT o interrupción externa.
    pub epochs: u32,
    pub batch_warps: u32,
    /// Selecciona un kernel especializado; el DFS básico no ejecuta inferencias.
    pub lookahead: bool,
    /// Journal explícito de aristas del sufijo; no altera los prefijos existentes.
    pub explicit_trail: bool,
    /// Puente mock residente; activa journal y look-ahead especializados.
    pub neuro_mock: bool,
    /// TensorRT real; requiere una política cargada antes de iniciar la búsqueda.
    pub neuro_trt: bool,
    /// Decisiones libres entre solicitudes de política (1..=65536).
    pub policy_interval: u32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            prefix_bits: 4,
            order_seed: 0,
            steps: 4,
            epochs: 16,
            batch_warps: 64,
            lookahead: false,
            explicit_trail: false,
            neuro_mock: false,
            neuro_trt: false,
            policy_interval: 32,
        }
    }
}

impl SearchConfig {
    pub fn has_next_epoch(self, completed: u64) -> bool {
        self.epochs == 0 || completed < u64::from(self.epochs)
    }

    pub fn validate(self, graph: &Graph43) -> Result<u32, Error> {
        graph.validate()?;
        if !(1..=65_536).contains(&self.policy_interval) {
            return Err(Error::Input(
                "policy_interval debe pertenecer a 1..65536".into(),
            ));
        }
        if self.neuro_mock {
            #[cfg(not(feature = "neuro-mock"))]
            return Err(Error::Input(
                "--neuro-mock requiere recompilar con --features neuro-mock".into(),
            ));
            #[cfg(feature = "neuro-mock")]
            if !self.explicit_trail || !self.lookahead {
                return Err(Error::Input(
                    "neuro_mock requiere explicit_trail y lookahead".into(),
                ));
            }
        }
        if self.neuro_trt {
            #[cfg(not(feature = "neuro-trt"))]
            return Err(Error::Input(
                "--neuro-model requiere recompilar con --features neuro-trt".into(),
            ));
            #[cfg(feature = "neuro-trt")]
            if !self.explicit_trail || !self.lookahead || self.neuro_mock {
                return Err(Error::Input(
                    "TensorRT requiere journal/lookahead y excluye el mock".into(),
                ));
            }
        }
        if self.prefix_bits > 20 || self.prefix_bits as usize > graph.free_edges() {
            return Err(Error::Input(
                "prefix_bits debe ser <=20 y no superar las aristas libres".into(),
            ));
        }
        if !(1..=65_536).contains(&self.steps) || !(1..=512).contains(&self.batch_warps) {
            return Err(Error::Input(
                "steps=1..65536 por época, epochs=0 sin límite o >=1 y batch_warps=1..512; lanzamientos adaptativos para Windows"
                    .into(),
            ));
        }
        Ok(1_u32 << self.prefix_bits)
    }
}

#[derive(Clone, Debug)]
pub struct SearchReport {
    pub outcome: Outcome,
    pub universes: u32,
    pub epochs: u64,
    pub launches: u64,
    pub transitions: u64,
    pub prunes: u64,
    pub completed: u64,
    pub decisions: u64,
    pub forced_assignments: u64,
    pub lookahead_checks: u64,
    pub lookahead_conflicts: u64,
    pub policy_queries: u64,
    pub policy_responses: u64,
    pub policy_fallbacks: u64,
    pub elapsed: Duration,
    /// Tiempo del host entre launch y synchronize; no es tiempo exclusivo del kernel.
    pub launch_time: Duration,
    pub max_launch: Duration,
    pub min_batch_warps: u32,
    pub max_batch_warps: u32,
    pub max_launch_steps: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Input(String),
    NotCompiled,
    Unavailable(String),
    Device(String),
    Memory { needed: u64, free: u64 },
}

impl fmt::Display for Error {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) => write!(out, "entrada inválida: {message}"),
            Self::NotCompiled => write!(out, "recompila con --features gpu"),
            Self::Unavailable(message) => write!(out, "CUDA no disponible: {message}"),
            Self::Device(message) => write!(out, "error CUDA: {message}"),
            Self::Memory { needed, free } => write!(
                out,
                "VRAM insuficiente: se necesitan {needed} bytes y hay {free} libres"
            ),
        }
    }
}
impl std::error::Error for Error {}

pub struct NativeRamsey43 {
    device: backend::Device,
    name: String,
}

impl NativeRamsey43 {
    pub fn new(ordinal: usize) -> Result<Self, Error> {
        let (device, name) = backend::Device::open(ordinal)?;
        Ok(Self { device, name })
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn check(&self, graphs: &[Graph43]) -> Result<Vec<CheckResult>, Error> {
        for graph in graphs {
            graph.validate()?;
        }
        self.device.check(graphs)
    }
    /// Sonda explícita de inferencia para autotests, fuera del bucle de búsqueda.
    pub fn allowed_colours(
        &self,
        graphs: &[Graph43],
        edges: &[(usize, usize)],
    ) -> Result<Vec<u32>, Error> {
        if graphs.len() != edges.len() {
            return Err(Error::Input(
                "cada grafo necesita una arista para la sonda".into(),
            ));
        }
        for (graph, &(a, b)) in graphs.iter().zip(edges) {
            graph.validate()?;
            if edge_id(a, b).is_none() || (graph.rows[a] | graph.rows[VERTICES + a]) >> b & 1 != 0 {
                return Err(Error::Input(
                    "la sonda requiere a<b<43 y arista libre".into(),
                ));
            }
        }
        self.device.allowed_colours(graphs, edges)
    }
    /// La CPU sólo programa lotes: DFS, ramas y pruebas K5 ocurren en CUDA.
    /// UNSAT cubre exclusivamente las extensiones del grafo base proporcionado.
    pub fn solve(&self, graph: &Graph43, config: SearchConfig) -> Result<SearchReport, Error> {
        let universes = config.validate(graph)?;
        self.device.solve(graph, config, universes)
    }

    /// Carga una sola vez el engine; las cuatro clases reutilizan contexto y pesos.
    #[cfg(feature = "neuro-trt")]
    pub fn load_tensorrt(&self, path: &std::path::Path, batch_capacity: u32) -> Result<(), Error> {
        self.device.load_tensorrt(path, batch_capacity)
    }

    /// Captura padres reales sin alterar el presupuesto ni las decisiones del
    /// DFS; las etiquetas provienen de sondas independientes de sus candidatos.
    #[cfg(feature = "neuro-dataset")]
    pub fn solve_with_dataset(
        &self,
        graph: &Graph43,
        config: SearchConfig,
        exporter: &mut dataset::DatasetExporter,
        sample_every: u32,
        horizon: u32,
    ) -> Result<SearchReport, Error> {
        let universes = config.validate(graph)?;
        if sample_every == 0
            || !(32..=4096).contains(&horizon)
            || !config.explicit_trail
            || !config.lookahead
            || config.neuro_mock
            || config.neuro_trt
        {
            return Err(Error::Input("dataset requiere sample_every>=1, horizon=32..4096, explicit_trail y lookahead, sin neuro_mock".into()));
        }
        if exporter.config().sample_every != sample_every || exporter.config().horizon != horizon {
            return Err(Error::Input(
                "configuración de muestreo incompatible con el header del exporter".into(),
            ));
        }
        self.device
            .solve_with_dataset(graph, config, universes, exporter, sample_every, horizon)
    }
}

#[cfg(any(feature = "neuro-mock", feature = "neuro-dataset"))]
fn decode_feature_half(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 31);
    let fraction = u32::from(bits & 1023);
    match exponent {
        0 => {
            let value = fraction as f32 * 2.0_f32.powi(-24);
            if sign == 0 {
                value
            } else {
                -value
            }
        }
        31 => f32::from_bits(sign | 0x7f80_0000 | (fraction << 13)),
        _ => f32::from_bits(sign | ((exponent + 112) << 23) | (fraction << 13)),
    }
}

/// Referencia del laboratorio por enumeración de vértices, independiente de
/// popcounts/reducciones CUDA. No se llama durante la búsqueda normal.
#[cfg(any(feature = "neuro-mock", feature = "neuro-dataset"))]
fn feature_reference(graph: &Graph43, depth: u32, index: u16) -> [f32; 32] {
    let mut result = [0.0; 32];
    if usize::from(index) >= EDGES {
        return result;
    }
    let (a, b) = edge_vertices(usize::from(index) + 1).expect("ID de test válido");
    let mut neighbours = [[[false; VERTICES]; VERTICES]; 3];
    let mut degrees = [[0_u32; VERTICES]; 3];
    let mut edge_counts = [0_u32; 3];
    for vertex in 0..VERTICES {
        for other in 0..VERTICES {
            if vertex == other {
                continue;
            }
            let colour = if graph.rows[vertex] >> other & 1 != 0 {
                0
            } else if graph.rows[VERTICES + vertex] >> other & 1 != 0 {
                1
            } else {
                2
            };
            neighbours[colour][vertex][other] = true;
            degrees[colour][vertex] += 1;
            if vertex < other {
                edge_counts[colour] += 1;
            }
        }
    }
    for colour in 0..3 {
        result[colour] = edge_counts[colour] as f32 / EDGES as f32;
    }
    result[3] = depth as f32 / EDGES as f32;
    for colour in 0..2 {
        result[4 + colour * 2] = *degrees[colour].iter().min().unwrap() as f32 / 42.0;
        result[5 + colour * 2] = *degrees[colour].iter().max().unwrap() as f32 / 42.0;
    }
    for colour in 0..3 {
        result[8 + colour * 2] = degrees[colour][a].min(degrees[colour][b]) as f32 / 42.0;
        result[9 + colour * 2] = degrees[colour][a].max(degrees[colour][b]) as f32 / 42.0;
        result[14 + colour] = (0..VERTICES)
            .filter(|&vertex| neighbours[colour][a][vertex] && neighbours[colour][b][vertex])
            .count() as f32
            / 42.0;
    }
    for (offset, left, right) in [(17, 0, 1), (18, 0, 2), (19, 1, 2)] {
        result[offset] = (0..VERTICES)
            .map(|vertex| {
                u32::from(neighbours[left][a][vertex] && neighbours[right][b][vertex])
                    + u32::from(neighbours[right][a][vertex] && neighbours[left][b][vertex])
            })
            .sum::<u32>() as f32
            / 84.0;
    }
    result
}

#[cfg(feature = "gpu")]
mod backend {
    use super::*;
    use cudarc::driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
    };
    use std::sync::Arc;

    #[cfg(not(any(
        feature = "neuro-mock",
        feature = "neuro-trt",
        feature = "neuro-dataset"
    )))]
    const SOURCE: &str = concat!(
        "#define RAMSEY43_COMBINED_SOURCE 1\n",
        include_str!("ramsey43_native.cu")
    );
    #[cfg(all(
        any(feature = "neuro-mock", feature = "neuro-trt"),
        not(feature = "neuro-dataset")
    ))]
    const SOURCE: &str = concat!(
        "#define RAMSEY43_COMBINED_SOURCE 1\n",
        include_str!("ramsey43_native.cu"),
        "\n",
        include_str!("ramsey43_features.cuh")
    );
    #[cfg(feature = "neuro-dataset")]
    const SOURCE: &str = concat!(
        "#define RAMSEY43_COMBINED_SOURCE 1\n",
        include_str!("ramsey43_native.cu"),
        "\n",
        include_str!("ramsey43_features.cuh"),
        "\n",
        include_str!("ramsey43_dataset.cuh")
    );
    const ROWS_PER_UNIVERSE: usize = 86;
    const ALT_WORDS_PER_UNIVERSE: usize = 32;
    const CONTROL_WORDS_PER_UNIVERSE: usize = 5;
    const COUNTER_WORDS: usize = 7;

    fn reject_universe_errors(controls: &[u32]) -> Result<(), Error> {
        if let Some((universe, control)) = controls
            .chunks_exact(CONTROL_WORDS_PER_UNIVERSE)
            .enumerate()
            .find(|(_, control)| control[3] >= 4)
        {
            return Err(Error::Device(format!(
                "universo {universe} terminó en estado de error {}",
                control[3]
            )));
        }
        Ok(())
    }

    pub struct Device {
        context: Arc<CudaContext>,
        #[cfg(feature = "neuro-trt")]
        trt_policy: std::cell::RefCell<Option<(u32, risk::RiskEngine)>>,
        stream: Arc<CudaStream>,
        check: CudaFunction,
        search: CudaFunction,
        search_smart: CudaFunction,
        search_journal: CudaFunction,
        search_journal_smart: CudaFunction,
        lookahead_check: CudaFunction,
        #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
        search_neuro: CudaFunction,
        #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
        pack_features: CudaFunction,
        #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
        apply_scores: CudaFunction,
        #[cfg(feature = "neuro-dataset")]
        search_dataset: CudaFunction,
        #[cfg(feature = "neuro-dataset")]
        dataset_pack: CudaFunction,
        #[cfg(feature = "neuro-dataset")]
        dataset_probe: CudaFunction,
    }
    enum TrailBuffers {
        Implicit(CudaSlice<u32>),
        Explicit {
            journal: CudaSlice<u16>,
            capacity: u32,
        },
    }
    struct SearchBuffers {
        base: CudaSlice<u64>,
        rows: CudaSlice<u64>,
        trail: TrailBuffers,
        control: CudaSlice<u32>,
        found: CudaSlice<u32>,
        model: CudaSlice<u64>,
        counters: CudaSlice<u64>,
    }
    #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
    struct NeuroBuffers {
        stream: Arc<CudaStream>,
        engine: risk::RiskEngine,
        ids: CudaSlice<u16>,
        meta: CudaSlice<u32>,
        policy: CudaSlice<u32>,
        generation: CudaSlice<u64>,
        counters: CudaSlice<u64>,
        features_ready: cudarc::driver::CudaEvent,
        scores_ready: cudarc::driver::CudaEvent,
        batch_capacity: u32,
    }
    #[cfg(feature = "neuro-dataset")]
    const DATASET_CAPACITY: u32 = 32;
    #[cfg(feature = "neuro-dataset")]
    struct DatasetBuffers {
        sequence: CudaSlice<u64>,
        rows: CudaSlice<u64>,
        meta: CudaSlice<u32>,
        control: CudaSlice<u32>,
        x: CudaSlice<u16>,
        ids: CudaSlice<u16>,
        valid: CudaSlice<u32>,
        cost: CudaSlice<u32>,
        status: CudaSlice<u32>,
        journal: CudaSlice<u16>,
    }
    #[cfg(feature = "neuro-dataset")]
    struct DatasetSession<'a> {
        exporter: &'a mut dataset::DatasetExporter,
        sample_every: u32,
        horizon: u32,
        run_id: u64,
        canonical_degree: u8,
        captured: u64,
        gpu_dropped: u64,
        labeling_time: Duration,
    }
    #[cfg(feature = "neuro-dataset")]
    impl DatasetSession<'_> {
        fn print_stats(&self) {
            let writer = self.exporter.stats();
            eprintln!("c dataset: run={}; captured={}; gpu_dropped={}; queued={}; queue_dropped={}; labeling_ms={:.3}",
                self.run_id, self.captured, self.gpu_dropped, writer.submitted_records, writer.dropped_records,
                self.labeling_time.as_secs_f64() * 1000.0);
        }
    }
    #[derive(Debug, PartialEq, Eq)]
    struct JournalSnapshot {
        graphs: Vec<Graph43>,
        frames: Vec<Vec<u16>>,
        controls: Vec<[u32; CONTROL_WORDS_PER_UNIVERSE]>,
        counters: [u64; COUNTER_WORDS],
    }
    fn device(error: DriverError) -> Error {
        Error::Device(error.to_string())
    }
    impl Device {
        pub fn open(ordinal: usize) -> Result<(Self, String), Error> {
            let context = std::panic::catch_unwind(|| CudaContext::new(ordinal))
                .map_err(|_| Error::Unavailable("driver NVIDIA no encontrado".into()))?
                .map_err(|error| Error::Unavailable(error.to_string()))?;
            let name = context.name().map_err(device)?;
            let (major, minor) = context.compute_capability().map_err(device)?;
            let options = cudarc::nvrtc::CompileOptions {
                options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
                ..Default::default()
            };
            let ptx =
                std::panic::catch_unwind(|| cudarc::nvrtc::compile_ptx_with_opts(SOURCE, options))
                    .map_err(|_| Error::Unavailable("NVRTC no encontrado".into()))?
                    .map_err(|error| Error::Device(format!("NVRTC: {error:?}")))?;
            let module = context.load_module(ptx).map_err(device)?;
            let check = module.load_function("ramsey43_check").map_err(device)?;
            let search = module.load_function("ramsey43_search").map_err(device)?;
            let search_smart = module
                .load_function("ramsey43_search_smart")
                .map_err(device)?;
            let search_journal = module
                .load_function("ramsey43_search_journal")
                .map_err(device)?;
            let search_journal_smart = module
                .load_function("ramsey43_search_journal_smart")
                .map_err(device)?;
            let lookahead_check = module
                .load_function("ramsey43_lookahead_check")
                .map_err(device)?;
            #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
            let search_neuro = module
                .load_function("ramsey43_search_neuro_smart")
                .map_err(device)?;
            #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
            let pack_features = module
                .load_function("ramsey43_pack_features")
                .map_err(device)?;
            #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
            let apply_scores = module
                .load_function("ramsey43_apply_scores")
                .map_err(device)?;
            #[cfg(feature = "neuro-dataset")]
            let search_dataset = module
                .load_function("ramsey43_search_dataset_smart")
                .map_err(device)?;
            #[cfg(feature = "neuro-dataset")]
            let dataset_pack = module
                .load_function("ramsey43_dataset_pack")
                .map_err(device)?;
            #[cfg(feature = "neuro-dataset")]
            let dataset_probe = module
                .load_function("ramsey43_dataset_probe")
                .map_err(device)?;
            let stream = context.default_stream();
            Ok((
                Self {
                    context,
                    #[cfg(feature = "neuro-trt")]
                    trt_policy: std::cell::RefCell::new(None),
                    stream,
                    check,
                    search,
                    search_smart,
                    search_journal,
                    search_journal_smart,
                    lookahead_check,
                    #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
                    search_neuro,
                    #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
                    pack_features,
                    #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
                    apply_scores,
                    #[cfg(feature = "neuro-dataset")]
                    search_dataset,
                    #[cfg(feature = "neuro-dataset")]
                    dataset_pack,
                    #[cfg(feature = "neuro-dataset")]
                    dataset_probe,
                },
                name,
            ))
        }
        #[cfg(feature = "neuro-trt")]
        pub fn load_tensorrt(
            &self,
            path: &std::path::Path,
            batch_capacity: u32,
        ) -> Result<(), Error> {
            if self.trt_policy.borrow().is_some() {
                return Err(Error::Input("la política TensorRT ya está cargada".into()));
            }
            let engine = risk::RiskEngine::tensorrt(self.context.clone(), batch_capacity, path)?;
            debug_assert_eq!(engine.batch_capacity(), batch_capacity);
            self.trt_policy
                .borrow_mut()
                .replace((batch_capacity, engine));
            Ok(())
        }
        fn reserve(&self, needed: u64) -> Result<(), Error> {
            let (free, _) = self.context.mem_get_info().map_err(device)?;
            if needed > free as u64 {
                return Err(Error::Memory {
                    needed,
                    free: free as u64,
                });
            }
            Ok(())
        }
        pub fn validate_kernel_resources(&self) -> Result<(), Error> {
            // Sólo en --self-test: mide el código NVRTC/JIT realmente cargado.
            for (name, function) in [
                ("dfs", &self.search),
                ("smart", &self.search_smart),
                ("journal", &self.search_journal),
                ("journal-smart", &self.search_journal_smart),
                #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
                ("neuro-smart", &self.search_neuro),
                #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
                ("feature-pack", &self.pack_features),
                #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
                ("score-apply", &self.apply_scores),
                #[cfg(feature = "neuro-dataset")]
                ("dataset-smart", &self.search_dataset),
                #[cfg(feature = "neuro-dataset")]
                ("dataset-pack", &self.dataset_pack),
                #[cfg(feature = "neuro-dataset")]
                ("dataset-probe", &self.dataset_probe),
            ] {
                let registers = function.num_regs().map_err(device)?;
                let local = function.local_size_bytes().map_err(device)?;
                let blocks = function
                    .occupancy_max_active_blocks_per_multiprocessor(256, 0, None)
                    .map_err(device)?;
                println!("c kernel: mode={name}; registers={registers}; local_bytes={local}; active_blocks_per_sm={blocks}; threads_per_block=256");
                if local != 0 {
                    return Err(Error::Device(format!(
                        "kernel {name} usa memoria local: {local} bytes por hilo"
                    )));
                }
            }
            Ok(())
        }
        pub fn allowed_colours(
            &self,
            graphs: &[Graph43],
            edges: &[(usize, usize)],
        ) -> Result<Vec<u32>, Error> {
            if graphs.is_empty() {
                return Ok(Vec::new());
            }
            let count = u32::try_from(graphs.len())
                .map_err(|_| Error::Input("demasiados grafos de prueba".into()))?;
            self.reserve(u64::from(count) * (86 * 8 + 2 * 4 + 4))?;
            let flat: Vec<_> = graphs.iter().flat_map(|graph| graph.rows).collect();
            let endpoints: Vec<u32> = edges
                .iter()
                .flat_map(|&(a, b)| [a as u32, b as u32])
                .collect();
            let rows = self.stream.clone_htod(&flat).map_err(device)?;
            let endpoints = self.stream.clone_htod(&endpoints).map_err(device)?;
            let mut masks = self
                .stream
                .alloc_zeros::<u32>(graphs.len())
                .map_err(device)?;
            self.launch_lookahead_check(&rows, count, &endpoints, &mut masks)?;
            self.stream.clone_dtoh(&masks).map_err(device)
        }
        #[allow(unsafe_code)]
        fn launch_lookahead_check(
            &self,
            rows: &CudaSlice<u64>,
            count: u32,
            edges: &CudaSlice<u32>,
            masks: &mut CudaSlice<u32>,
        ) -> Result<(), Error> {
            let mut call = self.stream.launch_builder(&self.lookahead_check);
            call.arg(rows).arg(&count).arg(edges).arg(masks);
            // SAFETY: ABI de cuatro argumentos; 86 filas y dos extremos validados
            // por grafo, count salidas; siempre 32 lanes activos por universo.
            unsafe {
                call.launch(LaunchConfig {
                    grid_dim: (count.div_ceil(8), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(device)?;
            self.stream.synchronize().map_err(device)
        }
        pub fn check(&self, graphs: &[Graph43]) -> Result<Vec<CheckResult>, Error> {
            if graphs.is_empty() {
                return Ok(Vec::new());
            }
            let count = u32::try_from(graphs.len())
                .map_err(|_| Error::Input("demasiados grafos".into()))?;
            self.reserve(u64::from(count) * (86 * 8 + 4 + 8))?;
            let flat: Vec<_> = graphs.iter().flat_map(|graph| graph.rows).collect();
            let rows = self.stream.clone_htod(&flat).map_err(device)?;
            let mut verdicts = self
                .stream
                .alloc_zeros::<u32>(graphs.len())
                .map_err(device)?;
            let mut witnesses = self
                .stream
                .alloc_zeros::<u64>(graphs.len())
                .map_err(device)?;
            self.launch_check(&rows, count, &mut verdicts, &mut witnesses)?;
            let verdicts = self.stream.clone_dtoh(&verdicts).map_err(device)?;
            let witnesses = self.stream.clone_dtoh(&witnesses).map_err(device)?;
            verdicts
                .into_iter()
                .zip(witnesses)
                .zip(graphs)
                .map(|((verdict, witness), graph)| match verdict {
                    0 if witness == 0 => Ok(CheckResult::Open),
                    1 if witness >> 63 != 0 => {
                        let blue = witness >> 30 & 1 != 0;
                        let vertices =
                            std::array::from_fn(|index| ((witness >> (6 * index)) & 63) as usize);
                        if !vertices.windows(2).all(|pair| pair[0] < pair[1])
                            || vertices[4] >= VERTICES
                            || !(0..5).all(|a| {
                                (a + 1..5).all(|b| {
                                    graph.rows[usize::from(blue) * VERTICES + vertices[a]]
                                        >> vertices[b]
                                        & 1
                                        != 0
                                })
                            })
                        {
                            return Err(Error::Device("testigo K5 inválido".into()));
                        }
                        Ok(CheckResult::Bad { blue, vertices })
                    }
                    2 if witness == 0 && graph.free_edges() == 0 => Ok(CheckResult::CompleteSafe),
                    _ => Err(Error::Device(format!(
                        "veredicto/testigo inválido: {verdict}/{witness}"
                    ))),
                })
                .collect()
        }
        #[allow(unsafe_code)]
        fn launch_check(
            &self,
            rows: &CudaSlice<u64>,
            count: u32,
            verdicts: &mut CudaSlice<u32>,
            witnesses: &mut CudaSlice<u64>,
        ) -> Result<(), Error> {
            let mut call = self.stream.launch_builder(&self.check);
            call.arg(rows).arg(&count).arg(verdicts).arg(witnesses);
            // SAFETY: firma exacta de ramsey43_check; 86 filas por grafo validado,
            // count resultados/testigos, 8 warps por bloque y sincronización antes de leer.
            unsafe {
                call.launch(LaunchConfig {
                    grid_dim: (count.div_ceil(8), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(device)?;
            self.stream.synchronize().map_err(device)
        }
        fn search_kernel(&self, config: SearchConfig) -> &CudaFunction {
            match (config.explicit_trail, config.lookahead) {
                (false, false) => &self.search,
                (false, true) => &self.search_smart,
                (true, false) => &self.search_journal,
                (true, true) => &self.search_journal_smart,
            }
        }
        fn allocate_search_buffers(
            &self,
            graph: &Graph43,
            config: SearchConfig,
            universes: u32,
        ) -> Result<SearchBuffers, Error> {
            self.allocate_search_buffers_on_stream(graph, config, universes, &self.stream)
        }
        fn allocate_search_buffers_on_stream(
            &self,
            graph: &Graph43,
            config: SearchConfig,
            universes: u32,
            stream: &Arc<CudaStream>,
        ) -> Result<SearchBuffers, Error> {
            let count = universes as usize;
            let journal_capacity = graph.free_edges() - config.prefix_bits as usize;
            // Capacidad lógica cero conserva un puntero/stride físico válido.
            let journal_stride = journal_capacity.max(1);
            let trail_bytes = if config.explicit_trail {
                journal_stride * std::mem::size_of::<u16>()
            } else {
                ALT_WORDS_PER_UNIVERSE * std::mem::size_of::<u32>()
            };
            let needed = u64::from(universes) * (86 * 8 + trail_bytes as u64 + 5 * 4)
                + 2 * 86 * 8
                + 4
                + (COUNTER_WORDS * 8) as u64;
            self.reserve(needed)?;
            if config.explicit_trail {
                let suffix_order = if config.neuro_mock || config.neuro_trt {
                    "neural-candidates/score-guided"
                } else {
                    "lexical"
                };
                println!("c explicit-trail: journal_capacity={journal_capacity}; journal_stride={journal_stride}; journal_bytes={}; arena_bytes={needed}; suffix_order={suffix_order}", u64::from(universes) * trail_bytes as u64);
            }
            let trail = if config.explicit_trail {
                TrailBuffers::Explicit {
                    journal: stream.alloc_zeros(count * journal_stride).map_err(device)?,
                    capacity: journal_capacity as u32,
                }
            } else {
                TrailBuffers::Implicit(
                    stream
                        .alloc_zeros(count * ALT_WORDS_PER_UNIVERSE)
                        .map_err(device)?,
                )
            };
            Ok(SearchBuffers {
                base: stream.clone_htod(graph.rows()).map_err(device)?,
                rows: stream
                    .alloc_zeros(count * ROWS_PER_UNIVERSE)
                    .map_err(device)?,
                trail,
                control: stream
                    .alloc_zeros(count * CONTROL_WORDS_PER_UNIVERSE)
                    .map_err(device)?,
                found: stream.alloc_zeros(1).map_err(device)?,
                model: stream.alloc_zeros(ROW_WORDS).map_err(device)?,
                counters: stream.alloc_zeros(COUNTER_WORDS).map_err(device)?,
            })
        }
        // Sonda de laboratorio: descarga estado sólo durante --self-test.
        fn journal_snapshot(
            &self,
            graph: &Graph43,
            config: SearchConfig,
            budgets: &[u32],
        ) -> Result<JournalSnapshot, Error> {
            if !config.explicit_trail {
                return Err(Error::Input("la sonda requiere explicit_trail".into()));
            }
            let universes = config.validate(graph)?;
            let mut buffers = self.allocate_search_buffers(graph, config, universes)?;
            for &steps in budgets {
                let launch_config = SearchConfig { steps, ..config };
                launch_config.validate(graph)?;
                self.launch_search(
                    self.search_kernel(config),
                    &mut buffers,
                    universes,
                    0,
                    universes,
                    launch_config,
                )?;
            }
            let rows = self.stream.clone_dtoh(&buffers.rows).map_err(device)?;
            let controls = self.stream.clone_dtoh(&buffers.control).map_err(device)?;
            let totals = self.stream.clone_dtoh(&buffers.counters).map_err(device)?;
            let (journal, capacity) = match &buffers.trail {
                TrailBuffers::Explicit { journal, capacity } => (
                    self.stream.clone_dtoh(journal).map_err(device)?,
                    *capacity as usize,
                ),
                TrailBuffers::Implicit(_) => unreachable!("config explícita validada"),
            };
            let mut snapshot = JournalSnapshot {
                graphs: Vec::with_capacity(universes as usize),
                frames: Vec::with_capacity(universes as usize),
                controls: Vec::with_capacity(universes as usize),
                counters: totals.try_into().map_err(|_| {
                    Error::Device("contadores del journal de longitud incorrecta".into())
                })?,
            };
            for (universe, control) in controls
                .chunks_exact(CONTROL_WORDS_PER_UNIVERSE)
                .enumerate()
            {
                if control[0] != 1
                    || control[1] as usize > capacity
                    || control[2] != 0
                    || control[3] > 3
                    || control[4] > 1
                {
                    return Err(Error::Device(format!(
                        "control del journal inválido en universo {universe}: {control:?}"
                    )));
                }
                let offset = universe * ROWS_PER_UNIVERSE;
                snapshot.graphs.push(Graph43::from_rows(
                    rows[offset..offset + ROWS_PER_UNIVERSE]
                        .try_into()
                        .map_err(|_| Error::Device("filas de journal incompletas".into()))?,
                )?);
                let offset = universe * capacity.max(1);
                // Los frames fuera de profundidad están muertos y pueden conservar basura.
                snapshot
                    .frames
                    .push(journal[offset..offset + control[1] as usize].to_vec());
                snapshot.controls.push(
                    control
                        .try_into()
                        .map_err(|_| Error::Device("control de journal incompleto".into()))?,
                );
            }
            drop(buffers);
            self.stream.synchronize().map_err(device)?;
            Ok(snapshot)
        }
        fn check_journal_snapshot(
            snapshot: &JournalSnapshot,
            protected: &[Graph43],
        ) -> Result<(), Error> {
            if snapshot.graphs.len() != protected.len() {
                return Err(Error::Device(
                    "raíces protegidas del journal incompletas".into(),
                ));
            }
            for (universe, ((graph, frames), root)) in snapshot
                .graphs
                .iter()
                .zip(&snapshot.frames)
                .zip(protected)
                .enumerate()
            {
                let mut reconstructed = root.clone();
                let mut seen = [false; EDGES];
                for &frame in frames {
                    let index = usize::from(frame & 0x03ff);
                    let blue = frame & 0x0400 != 0;
                    let alternative_consumed = frame & 0x0800 != 0;
                    let forced = frame & 0x1000 != 0;
                    if index >= EDGES || frame & 0xe000 != 0 || (forced && !alternative_consumed) {
                        return Err(Error::Device(format!(
                            "frame del journal inválido en universo {universe}: {frame:#06x}"
                        )));
                    }
                    let (a, b) = edge_vertices(index + 1).expect("ID de frame validado");
                    if seen[index] || (root.rows[a] | root.rows[VERTICES + a]) >> b & 1 != 0 {
                        return Err(Error::Device(format!(
                            "frame duplicado o raíz sobrescrita en universo {universe}: {index}"
                        )));
                    }
                    seen[index] = true;
                    let literal = (index + 1) as i32;
                    reconstructed.assign(if blue { -literal } else { literal })?;
                }
                if graph != &reconstructed {
                    return Err(Error::Device(format!(
                        "grafo != raíz + frames vivos en universo {universe}"
                    )));
                }
            }
            Ok(())
        }
        pub fn validate_explicit_trail(&self) -> Result<usize, Error> {
            let common = SearchConfig {
                prefix_bits: 0,
                order_seed: 864,
                steps: 1,
                epochs: 1,
                batch_warps: 2,
                lookahead: false,
                explicit_trail: true,
                neuro_mock: false,
                neuro_trt: false,
                policy_interval: 32,
            };
            let empty = Graph43::default();
            let mut checks = 0;
            let prefixed = SearchConfig {
                prefix_bits: 1,
                ..common
            };
            let mut roots = vec![empty.clone(), empty.clone()];
            // mix(864)%903=857: el prefijo permutado fija (32,42), no (0,1).
            roots[0].assign(858)?;
            roots[1].assign(-858)?;
            let mut contradiction = Graph43::default();
            for (vertices, blue) in [([0, 1, 2, 3, 4], false), ([0, 1, 5, 6, 7], true)] {
                for a in 0..5 {
                    for b in a + 1..5 {
                        if a == 0 && b == 1 {
                            continue;
                        }
                        let id = edge_id(vertices[a], vertices[b]).unwrap() as i32;
                        contradiction.assign(if blue { -id } else { id })?;
                    }
                }
            }
            let mut bad_roots = vec![contradiction.clone(), contradiction.clone()];
            bad_roots[0].assign(858)?;
            bad_roots[1].assign(-858)?;
            // A=(0,1) libre. A roja bloquea ambos colores de B=(0,2):
            // el retroceso debe probar A azul y marcar su alternativa consumida.
            let mut flip = Graph43::default();
            for (vertices, blue) in [([0, 1, 2, 3, 4], false), ([0, 2, 5, 6, 7], true)] {
                for a in 0..5 {
                    for b in a + 1..5 {
                        let edge = (vertices[a], vertices[b]);
                        if edge == (0, 2) || (!blue && edge == (0, 1)) {
                            continue;
                        }
                        let id = edge_id(edge.0, edge.1).unwrap() as i32;
                        flip.assign(if blue { -id } else { id })?;
                    }
                }
            }
            for lookahead in [false, true] {
                let config = SearchConfig {
                    lookahead,
                    ..prefixed
                };
                let snapshot = self.journal_snapshot(&empty, config, &[1])?;
                Self::check_journal_snapshot(&snapshot, &roots)?;
                if snapshot
                    .frames
                    .iter()
                    .any(|frames| frames.len() != 1 || frames[0] & 0x03ff != 0)
                {
                    return Err(Error::Device(format!(
                        "sufijo léxico no conserva el prefijo permutado: {snapshot:?}"
                    )));
                }
                checks += 1;
                let snapshot = self.journal_snapshot(&contradiction, config, &[16])?;
                Self::check_journal_snapshot(&snapshot, &bad_roots)?;
                if snapshot.counters[2] != 2
                    || snapshot
                        .controls
                        .iter()
                        .any(|control| control[1] != 0 || control[3] != 3)
                {
                    return Err(Error::Device(format!(
                        "agotamiento del journal alteró la raíz o dejó frames vivos: {snapshot:?}"
                    )));
                }
                checks += 1;
                let config = SearchConfig {
                    lookahead,
                    ..common
                };
                let burst = self.journal_snapshot(&empty, config, &[40])?;
                let tiny = self.journal_snapshot(&empty, config, &[1; 40])?;
                Self::check_journal_snapshot(&burst, std::slice::from_ref(&empty))?;
                Self::check_journal_snapshot(&tiny, std::slice::from_ref(&empty))?;
                if burst != tiny || burst.frames[0].len() != 40 {
                    return Err(Error::Device(format!(
                        "checkpoint del journal >32 frames cambió el DFS: {burst:?} vs {tiny:?}"
                    )));
                }
                checks += 1;
                let config = SearchConfig {
                    lookahead,
                    order_seed: 0,
                    ..common
                };
                let first = self.journal_snapshot(&flip, config, &[1])?;
                Self::check_journal_snapshot(&first, std::slice::from_ref(&flip))?;
                if first.frames[0] != [0] {
                    return Err(Error::Device(format!(
                        "primera polaridad del journal no es A roja libre: {first:?}"
                    )));
                }
                let burst = self.journal_snapshot(&flip, config, &[3])?;
                let tiny = self.journal_snapshot(&flip, config, &[1; 3])?;
                Self::check_journal_snapshot(&burst, std::slice::from_ref(&flip))?;
                Self::check_journal_snapshot(&tiny, std::slice::from_ref(&flip))?;
                if burst != tiny || burst.frames[0] != [0x0c00] {
                    return Err(Error::Device(format!(
                        "el journal no probó el complemento azul de A: {burst:?} vs {tiny:?}"
                    )));
                }
                checks += 2;
            }
            let mut rollback = Graph43::default();
            for (vertices, blue) in [
                ([0, 1, 3, 4, 5], false),
                ([0, 2, 8, 9, 10], false),
                ([0, 2, 11, 12, 13], true),
            ] {
                for a in 0..5 {
                    for b in a + 1..5 {
                        if a == 0 && b == 1 {
                            continue;
                        }
                        let id = edge_id(vertices[a], vertices[b]).unwrap() as i32;
                        rollback.assign(if blue { -id } else { id })?;
                    }
                }
            }
            let config = SearchConfig {
                order_seed: 0,
                lookahead: true,
                ..common
            };
            let forced = self.journal_snapshot(&rollback, config, &[1])?;
            Self::check_journal_snapshot(&forced, std::slice::from_ref(&rollback))?;
            if forced.frames[0] != [0x1c00] || forced.counters[4] != 1 {
                return Err(Error::Device(format!(
                    "frame forzado del journal no conserva color/alternativa: {forced:?}"
                )));
            }
            checks += 1;
            let tiny = self.journal_snapshot(&rollback, config, &[1; 8])?;
            let burst = self.journal_snapshot(&rollback, config, &[8])?;
            Self::check_journal_snapshot(&tiny, std::slice::from_ref(&rollback))?;
            Self::check_journal_snapshot(&burst, std::slice::from_ref(&rollback))?;
            if tiny != burst
                || tiny.controls[0][1] != 0
                || tiny.controls[0][3] != 3
                || tiny.counters[4] != 1
                || tiny.counters[3] != 0
            {
                return Err(Error::Device(format!(
                    "rollback forzado del journal cambió la raíz: {tiny:?} vs {burst:?}"
                )));
            }
            checks += 1;
            let mut full_red = Graph43::default();
            for id in 1..=EDGES as i32 {
                full_red.assign(id)?;
            }
            for lookahead in [false, true] {
                let snapshot = self.journal_snapshot(
                    &full_red,
                    SearchConfig {
                        lookahead,
                        ..common
                    },
                    &[1],
                )?;
                Self::check_journal_snapshot(&snapshot, std::slice::from_ref(&full_red))?;
                if snapshot.counters[2] != 1 || snapshot.controls[0][3] != 1 {
                    return Err(Error::Device(format!(
                        "capacidad cero del journal no refutó la raíz completa: {snapshot:?}"
                    )));
                }
                checks += 1;
            }
            Ok(checks)
        }
        pub fn solve(
            &self,
            graph: &Graph43,
            config: SearchConfig,
            universes: u32,
        ) -> Result<SearchReport, Error> {
            #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
            if config.neuro_mock || config.neuro_trt {
                return self.solve_neuro(graph, config, universes);
            }
            #[cfg(feature = "neuro-dataset")]
            {
                self.solve_main::<false>(graph, config, universes, None)
            }
            #[cfg(not(feature = "neuro-dataset"))]
            {
                self.solve_main::<false>(graph, config, universes)
            }
        }

        #[cfg(feature = "neuro-dataset")]
        pub fn solve_with_dataset(
            &self,
            graph: &Graph43,
            config: SearchConfig,
            universes: u32,
            exporter: &mut dataset::DatasetExporter,
            sample_every: u32,
            horizon: u32,
        ) -> Result<SearchReport, Error> {
            use std::sync::atomic::{AtomicU64, Ordering};
            use std::time::{SystemTime, UNIX_EPOCH};
            static RUN_COUNTER: AtomicU64 = AtomicU64::new(0);
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|_| Error::Input("reloj anterior a Unix epoch".into()))?
                .as_nanos() as u64;
            // Monotonic within this process even if the system clock goes back;
            // timestamp and PID distinguish independent collection processes.
            let previous = RUN_COUNTER
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |previous| {
                    Some(timestamp.max(previous.saturating_add(1)))
                })
                .expect("fetch_update closure always returns Some");
            let run_id =
                timestamp.max(previous.saturating_add(1)) ^ (u64::from(std::process::id()) << 32);
            let known_star = (graph.rows[0] | graph.rows[VERTICES]) == (VERTEX_MASK & !1);
            let degree = graph.rows[0].count_ones() as u8;
            let canonical_degree = if known_star && (18..=21).contains(&degree) {
                degree
            } else {
                255
            };
            let session = DatasetSession {
                exporter,
                sample_every,
                horizon,
                run_id,
                canonical_degree,
                captured: 0,
                gpu_dropped: 0,
                labeling_time: Duration::ZERO,
            };
            self.solve_main::<true>(graph, config, universes, Some(session))
        }

        fn solve_main<const CAPTURE: bool>(
            &self,
            graph: &Graph43,
            config: SearchConfig,
            universes: u32,
            #[cfg(feature = "neuro-dataset")] mut dataset_session: Option<DatasetSession<'_>>,
        ) -> Result<SearchReport, Error> {
            let start = Instant::now();
            let mut buffers = self.allocate_search_buffers(graph, config, universes)?;
            #[cfg(feature = "neuro-dataset")]
            let mut dataset_buffers = if CAPTURE {
                Some(self.allocate_dataset_buffers(universes)?)
            } else {
                None
            };
            let (mut epochs, mut launches, mut batch_warps) = (0_u64, 0, config.batch_warps);
            let mut launch_steps = config.steps.min(8);
            let mut launch_time = Duration::ZERO;
            let mut max_launch = Duration::ZERO;
            let mut min_batch_warps = u32::MAX;
            let mut max_batch_warps = 0;
            let mut max_launch_steps = 0;
            let mut heartbeat = Instant::now();
            let (mut counters, mut outcome) = ([0_u64; COUNTER_WORDS], Outcome::Unknown);
            // La selección se hace una vez; ninguna bandera extra entra en el DFS.
            let search = self.search_kernel(config);
            let epoch_limit = if config.epochs == 0 {
                "unlimited".to_owned()
            } else {
                config.epochs.to_string()
            };
            'epochs: while config.has_next_epoch(epochs) {
                epochs = epochs.saturating_add(1);
                let mut first = 0;
                while first < universes {
                    let batch = batch_warps.min(universes - first);
                    // Cada lote conserva first/batch durante TODAS sus continuaciones.
                    // La suma de chunks es steps por época, aunque cambie su quantum.
                    let mut remaining = config.steps;
                    while remaining != 0 {
                        let chunk = launch_steps.min(remaining);
                        let launch_start = Instant::now();
                        let chunk_config = SearchConfig {
                            steps: chunk,
                            ..config
                        };
                        #[cfg(feature = "neuro-dataset")]
                        if CAPTURE {
                            let sampling = dataset_buffers
                                .as_mut()
                                .ok_or_else(|| Error::Device("arena dataset ausente".into()))?;
                            let session = dataset_session
                                .as_ref()
                                .ok_or_else(|| Error::Device("sesión dataset ausente".into()))?;
                            self.launch_dataset_search(
                                &mut buffers,
                                sampling,
                                universes,
                                first,
                                batch,
                                chunk_config,
                                session.sample_every,
                            )?;
                        }
                        if !CAPTURE {
                            self.launch_search(
                                search,
                                &mut buffers,
                                universes,
                                first,
                                batch,
                                chunk_config,
                            )?;
                        }
                        launches += 1;
                        let elapsed = launch_start.elapsed();
                        #[cfg(feature = "neuro-dataset")]
                        if CAPTURE {
                            self.drain_dataset(
                                dataset_buffers
                                    .as_mut()
                                    .ok_or_else(|| Error::Device("arena dataset ausente".into()))?,
                                dataset_session.as_mut().ok_or_else(|| {
                                    Error::Device("sesión dataset ausente".into())
                                })?,
                                config,
                                universes,
                            )?;
                        }
                        launch_time += elapsed;
                        max_launch = max_launch.max(elapsed);
                        min_batch_warps = min_batch_warps.min(batch);
                        max_batch_warps = max_batch_warps.max(batch);
                        max_launch_steps = max_launch_steps.max(chunk);
                        remaining -= chunk;
                        // Calibración conservadora: quantum inicial 8, techo 512.
                        // El tamaño de lote nunca supera el solicitado por CLI.
                        if elapsed > Duration::from_millis(100) {
                            launch_steps = (launch_steps / 2).max(1);
                            batch_warps = (batch_warps / 2).max(1);
                        } else if elapsed < Duration::from_millis(10) {
                            launch_steps = (launch_steps * 2).min(config.steps).min(512);
                            batch_warps = (batch_warps * 2).min(config.batch_warps);
                        }
                        let found = self.stream.clone_dtoh(&buffers.found).map_err(device)?[0];
                        if found == 2 {
                            let rows = self.stream.clone_dtoh(&buffers.model).map_err(device)?;
                            let model = Graph43::from_rows(rows.try_into().map_err(|_| {
                                Error::Device("modelo GPU de longitud incorrecta".into())
                            })?)?;
                            if !model.extends(graph)
                                || self.check(std::slice::from_ref(&model))?
                                    != [CheckResult::CompleteSafe]
                            {
                                return Err(Error::Device(
                                    "el modelo publicado no es una extensión completa sin K5"
                                        .into(),
                                ));
                            }
                            model.model()?;
                            outcome = Outcome::Sat(model);
                            break 'epochs;
                        }
                        if found != 0 {
                            return Err(Error::Device(format!(
                                "publicación incompleta de modelo: found={found}"
                            )));
                        }
                    }
                    first += batch;
                    if heartbeat.elapsed() >= Duration::from_secs(5) {
                        let totals = self.stream.clone_dtoh(&buffers.counters).map_err(device)?;
                        if config.epochs == 0 {
                            let controls =
                                self.stream.clone_dtoh(&buffers.control).map_err(device)?;
                            reject_universe_errors(&controls)?;
                        }
                        eprintln!("c progress: t={:.1}s; epoch={epochs}/{epoch_limit}; universes_visited={first}/{universes}; transitions={}; prunes={}; completed={}; batch={batch}; quantum={launch_steps}; max_launch_ms={:.3}",
                            start.elapsed().as_secs_f64(), totals[0], totals[1], totals[2], max_launch.as_secs_f64() * 1000.0);
                        if config.lookahead {
                            eprintln!("c inference: decisions={}; forced_assignments={}; lookahead_checks={}; lookahead_conflicts={}", totals[3], totals[4], totals[5], totals[6]);
                        }
                        #[cfg(feature = "neuro-dataset")]
                        if CAPTURE {
                            if let Some(session) = dataset_session.as_ref() {
                                session.print_stats();
                            }
                        }
                        heartbeat = Instant::now();
                    }
                }
                let totals = self.stream.clone_dtoh(&buffers.counters).map_err(device)?;
                counters.copy_from_slice(&totals);
                if counters[2] > u64::from(universes) {
                    return Err(Error::Device(
                        "contador de universos agotados desbordado".into(),
                    ));
                }
                if counters[2] == u64::from(universes) {
                    let controls = self.stream.clone_dtoh(&buffers.control).map_err(device)?;
                    if controls
                        .chunks_exact(CONTROL_WORDS_PER_UNIVERSE)
                        .all(|control| matches!(control[3], 1 | 3))
                    {
                        outcome = Outcome::Unsat;
                    } else {
                        return Err(Error::Device(
                            "contador de agotamiento incompatible con los estados de los universos"
                                .into(),
                        ));
                    }
                    break;
                }
            }
            let totals = self.stream.clone_dtoh(&buffers.counters).map_err(device)?;
            counters.copy_from_slice(&totals);
            // Un error de estado no debe quedar oculto como presupuesto agotado.
            // Se descarga al salir y en latidos continuos, nunca por paso DFS.
            let controls = self.stream.clone_dtoh(&buffers.control).map_err(device)?;
            reject_universe_errors(&controls)?;
            // cudaFreeAsync puede conservar la arena en el pool hasta sincronizar.
            // Liberarla antes del informe evita falsas faltas de VRAM en la clase siguiente.
            drop(buffers);
            #[cfg(feature = "neuro-dataset")]
            if CAPTURE {
                if let Some(session) = dataset_session.as_ref() {
                    session.print_stats();
                }
                drop(dataset_buffers);
            }
            self.stream.synchronize().map_err(device)?;
            Ok(SearchReport {
                outcome,
                universes,
                epochs,
                launches,
                transitions: counters[0],
                prunes: counters[1],
                completed: counters[2],
                decisions: counters[3],
                forced_assignments: counters[4],
                lookahead_checks: counters[5],
                lookahead_conflicts: counters[6],
                policy_queries: 0,
                policy_responses: 0,
                policy_fallbacks: 0,
                elapsed: start.elapsed(),
                launch_time,
                max_launch,
                min_batch_warps,
                max_batch_warps,
                max_launch_steps,
            })
        }
        #[cfg(feature = "neuro-dataset")]
        fn allocate_dataset_buffers(&self, universes: u32) -> Result<DatasetBuffers, Error> {
            let capacity = DATASET_CAPACITY as usize;
            let needed = u64::from(universes) * 8
                + capacity as u64
                    * (86 * 8 + 8 * 4 + 1024 * 2 + 32 * 2 + 4 + 64 * 8 + 64 * 903 * 2)
                + 8;
            self.reserve(needed)?;
            println!("c dataset arena: capacity={DATASET_CAPACITY}; arena_bytes={needed}; probe_journal_bytes={}", capacity * 64 * EDGES * 2);
            Ok(DatasetBuffers {
                sequence: self
                    .stream
                    .alloc_zeros(universes as usize)
                    .map_err(device)?,
                rows: self
                    .stream
                    .alloc_zeros(capacity * ROW_WORDS)
                    .map_err(device)?,
                meta: self.stream.alloc_zeros(capacity * 8).map_err(device)?,
                control: self.stream.alloc_zeros(2).map_err(device)?,
                x: self.stream.alloc_zeros(capacity * 1024).map_err(device)?,
                ids: self.stream.alloc_zeros(capacity * 32).map_err(device)?,
                valid: self.stream.alloc_zeros(capacity).map_err(device)?,
                cost: self.stream.alloc_zeros(capacity * 64).map_err(device)?,
                status: self.stream.alloc_zeros(capacity * 64).map_err(device)?,
                journal: self
                    .stream
                    .alloc_zeros(capacity * 64 * EDGES)
                    .map_err(device)?,
            })
        }

        #[cfg(feature = "neuro-dataset")]
        #[allow(unsafe_code)]
        fn launch_dataset_search(
            &self,
            buffers: &mut SearchBuffers,
            sampling: &mut DatasetBuffers,
            universes: u32,
            first: u32,
            count: u32,
            config: SearchConfig,
            sample_every: u32,
        ) -> Result<(), Error> {
            let (journal, capacity) = match &mut buffers.trail {
                TrailBuffers::Explicit { journal, capacity } => (journal, *capacity),
                _ => return Err(Error::Device("dataset requiere journal explícito".into())),
            };
            let mut call = self.stream.launch_builder(&self.search_dataset);
            call.arg(&buffers.base)
                .arg(&mut buffers.rows)
                .arg(journal)
                .arg(&mut buffers.control)
                .arg(&universes)
                .arg(&first)
                .arg(&count)
                .arg(&config.prefix_bits)
                .arg(&config.order_seed)
                .arg(&config.steps)
                .arg(&capacity)
                .arg(&mut buffers.found)
                .arg(&mut buffers.model)
                .arg(&mut buffers.counters)
                .arg(&mut sampling.sequence)
                .arg(&mut sampling.rows)
                .arg(&mut sampling.meta)
                .arg(&mut sampling.control)
                .arg(&sample_every)
                .arg(&DATASET_CAPACITY);
            // SAFETY: exact journal-smart ABI followed by six sampling arguments;
            // fixed capture arrays hold 32 parents, while sequence has U entries.
            unsafe {
                call.launch(LaunchConfig {
                    grid_dim: (count.div_ceil(8), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(device)?;
            self.stream.synchronize().map_err(device)
        }

        #[cfg(feature = "neuro-dataset")]
        #[allow(unsafe_code)]
        fn launch_dataset_labels(
            &self,
            sampling: &mut DatasetBuffers,
            count: u32,
            horizon: u32,
        ) -> Result<(), Error> {
            if count > DATASET_CAPACITY || count == 0 || horizon == 0 {
                return Err(Error::Input(
                    "lote de sondas dataset fuera del contrato".into(),
                ));
            }
            let mut pack = self.stream.launch_builder(&self.dataset_pack);
            pack.arg(&sampling.rows)
                .arg(&sampling.meta)
                .arg(&count)
                .arg(&DATASET_CAPACITY)
                .arg(&mut sampling.x)
                .arg(&mut sampling.ids)
                .arg(&mut sampling.valid);
            // SAFETY: immutable compact snapshots, fixed X[32][32][32] and IDs;
            // this stream orders packing before independent candidate probes.
            unsafe {
                pack.launch(LaunchConfig {
                    grid_dim: (DATASET_CAPACITY.div_ceil(8), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(device)?;
            let mut probe = self.stream.launch_builder(&self.dataset_probe);
            probe
                .arg(&sampling.rows)
                .arg(&sampling.ids)
                .arg(&sampling.valid)
                .arg(&count)
                .arg(&horizon)
                .arg(&mut sampling.journal)
                .arg(&mut sampling.cost)
                .arg(&mut sampling.status);
            // SAFETY: each of count*64 warps owns 903 global u16 journal entries
            // and a single cost/status. No live search buffer is an argument.
            unsafe {
                probe.launch(LaunchConfig {
                    grid_dim: ((count * 64).div_ceil(8), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(device)?;
            self.stream.synchronize().map_err(device)
        }

        #[cfg(feature = "neuro-dataset")]
        fn drain_dataset(
            &self,
            sampling: &mut DatasetBuffers,
            session: &mut DatasetSession<'_>,
            config: SearchConfig,
            universes: u32,
        ) -> Result<(), Error> {
            // Only two tiny counters cross PCIe on non-sampling chunks.
            let control = self.stream.clone_dtoh(&sampling.control).map_err(device)?;
            let count = control[0].min(DATASET_CAPACITY);
            if control[1] != control[0].saturating_sub(DATASET_CAPACITY) {
                return Err(Error::Device(
                    "contadores de captura dataset incompatibles".into(),
                ));
            }
            session.captured += u64::from(count);
            session.gpu_dropped += u64::from(control[1]);
            if count != 0 {
                let label_start = Instant::now();
                self.launch_dataset_labels(sampling, count, session.horizon)?;
                session.labeling_time += label_start.elapsed();
                let records = self.dataset_records(sampling, count, session, config, universes)?;
                session
                    .exporter
                    .try_submit(records)
                    .map_err(|error| Error::Device(format!("exportación dataset: {error}")))?;
            }
            self.stream
                .memcpy_htod(&[0_u32, 0], &mut sampling.control)
                .map_err(device)?;
            Ok(())
        }

        #[cfg(feature = "neuro-dataset")]
        fn dataset_records(
            &self,
            sampling: &DatasetBuffers,
            count: u32,
            session: &DatasetSession<'_>,
            config: SearchConfig,
            universes: u32,
        ) -> Result<Vec<dataset::DatasetRecord>, Error> {
            let rows = self.stream.clone_dtoh(&sampling.rows).map_err(device)?;
            let meta = self.stream.clone_dtoh(&sampling.meta).map_err(device)?;
            let x = self.stream.clone_dtoh(&sampling.x).map_err(device)?;
            let ids = self.stream.clone_dtoh(&sampling.ids).map_err(device)?;
            let valid = self.stream.clone_dtoh(&sampling.valid).map_err(device)?;
            let cost = self.stream.clone_dtoh(&sampling.cost).map_err(device)?;
            let status = self.stream.clone_dtoh(&sampling.status).map_err(device)?;
            let mut records = Vec::with_capacity(count as usize);
            for slot in 0..count as usize {
                let m = &meta[slot * 8..(slot + 1) * 8];
                if valid[slot] != 1
                    || m[0] >= universes
                    || m[4] >= EDGES as u32
                    || m[5] > 1
                    || m[6] != 0
                    || m[7] != 0
                {
                    return Err(Error::Device(format!(
                        "snapshot dataset {slot} inválido: {m:?}, valid={}",
                        valid[slot]
                    )));
                }
                let parent = Graph43::from_rows(
                    rows[slot * ROW_WORDS..(slot + 1) * ROW_WORDS]
                        .try_into()
                        .map_err(|_| Error::Device("filas dataset incompletas".into()))?,
                )?;
                let (a, b) = edge_vertices(m[4] as usize + 1).expect("ID dataset comprobado");
                if (parent.rows[a] | parent.rows[VERTICES + a]) >> b & 1 != 0 {
                    return Err(Error::Device(
                        "el snapshot dataset no precede a su decisión".into(),
                    ));
                }
                let candidate_ids: [u16; 32] = ids[slot * 32..(slot + 1) * 32]
                    .try_into()
                    .map_err(|_| Error::Device("IDs dataset incompletos".into()))?;
                let mut outcome = [0_u8; 64];
                for (index, &value) in status[slot * 64..(slot + 1) * 64].iter().enumerate() {
                    outcome[index] = u8::try_from(value)
                        .map_err(|_| Error::Device("estado de sonda fuera de u8".into()))?;
                }
                let mut parent_hash = 0xcbf29ce484222325_u64;
                for word in parent.rows {
                    for byte in word.to_le_bytes() {
                        parent_hash = (parent_hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
                    }
                }
                records.push(dataset::DatasetRecord {
                    run_id: session.run_id,
                    decision_seq: u64::from(m[1]) | (u64::from(m[2]) << 32),
                    universe: m[0],
                    depth: m[3],
                    chosen_edge: m[4] as u16,
                    chosen_phase: m[5] as u8,
                    canonical_degree: session.canonical_degree,
                    order_seed: config.order_seed,
                    prefix_bits: config.prefix_bits,
                    horizon: session.horizon,
                    candidate_count: candidate_ids
                        .iter()
                        .filter(|&&edge| usize::from(edge) < EDGES)
                        .count() as u32,
                    parent_hash,
                    x: x[slot * 1024..(slot + 1) * 1024]
                        .try_into()
                        .map_err(|_| Error::Device("features dataset incompletas".into()))?,
                    ids: candidate_ids,
                    cost: cost[slot * 64..(slot + 1) * 64]
                        .try_into()
                        .map_err(|_| Error::Device("costes dataset incompletos".into()))?,
                    status: outcome,
                });
            }
            Ok(records)
        }

        #[cfg(feature = "neuro-dataset")]
        pub fn validate_dataset(&self) -> Result<usize, Error> {
            let empty = Graph43::default();
            let config = SearchConfig {
                prefix_bits: 0,
                order_seed: 0,
                steps: 8,
                epochs: 1,
                batch_warps: 1,
                lookahead: true,
                explicit_trail: true,
                neuro_mock: false,
                neuro_trt: false,
                ..SearchConfig::default()
            };
            let mut regular = self.allocate_search_buffers(&empty, config, 1)?;
            let mut captured = self.allocate_search_buffers(&empty, config, 1)?;
            let mut sampling = self.allocate_dataset_buffers(1)?;
            self.launch_search(self.search_kernel(config), &mut regular, 1, 0, 1, config)?;
            self.launch_dataset_search(&mut captured, &mut sampling, 1, 0, 1, config, 1)?;
            if self.stream.clone_dtoh(&regular.rows).map_err(device)?
                != self.stream.clone_dtoh(&captured.rows).map_err(device)?
                || self.stream.clone_dtoh(&regular.control).map_err(device)?
                    != self.stream.clone_dtoh(&captured.control).map_err(device)?
                || self.stream.clone_dtoh(&regular.counters).map_err(device)?
                    != self.stream.clone_dtoh(&captured.counters).map_err(device)?
                || self.stream.clone_dtoh(&regular.found).map_err(device)?
                    != self.stream.clone_dtoh(&captured.found).map_err(device)?
            {
                return Err(Error::Device(
                    "el muestreo cambió el estado o los contadores del DFS".into(),
                ));
            }
            if let (
                TrailBuffers::Explicit { journal: plain, .. },
                TrailBuffers::Explicit {
                    journal: sampled, ..
                },
            ) = (&regular.trail, &captured.trail)
            {
                if self.stream.clone_dtoh(plain).map_err(device)?
                    != self.stream.clone_dtoh(sampled).map_err(device)?
                {
                    return Err(Error::Device(
                        "el muestreo cambió el journal del DFS".into(),
                    ));
                }
            }
            let control = self.stream.clone_dtoh(&sampling.control).map_err(device)?;
            if control != [8, 0] {
                return Err(Error::Device(format!(
                    "capture_every=1 no capturó las ocho decisiones: {control:?}"
                )));
            }
            let rows = self.stream.clone_dtoh(&sampling.rows).map_err(device)?;
            let meta = self.stream.clone_dtoh(&sampling.meta).map_err(device)?;
            self.launch_dataset_labels(&mut sampling, 8, 32)?;
            let x = self.stream.clone_dtoh(&sampling.x).map_err(device)?;
            let ids = self.stream.clone_dtoh(&sampling.ids).map_err(device)?;
            let mut checks = 5;
            for slot in 0..8 {
                let parent = Graph43::from_rows(
                    rows[slot * ROW_WORDS..(slot + 1) * ROW_WORDS]
                        .try_into()
                        .map_err(|_| Error::Device("snapshot de test incompleto".into()))?,
                )?;
                let m = &meta[slot * 8..(slot + 1) * 8];
                let edge = m[4] as usize;
                let (a, b) = edge_vertices(edge + 1)
                    .ok_or_else(|| Error::Device("ID capturado inválido".into()))?;
                if parent.free_edges() != EDGES - slot
                    || m[0] != 0
                    || m[1] != slot as u32 + 1
                    || m[2] != 0
                    || m[3] != slot as u32
                    || (parent.rows[a] | parent.rows[VERTICES + a]) >> b & 1 != 0
                    || check_reference(&parent) != CheckResult::Open
                {
                    return Err(Error::Device(format!(
                        "la captura {slot} no contiene el padre anterior a la decisión"
                    )));
                }
                for candidate in 0..32 {
                    let reference = feature_reference(&parent, m[3], ids[slot * 32 + candidate]);
                    for feature in 0..32 {
                        let actual = decode_feature_half(x[(slot * 32 + candidate) * 32 + feature]);
                        if (actual - reference[feature]).abs() > 0.0005 {
                            return Err(Error::Device(format!(
                                "feature dataset {slot}/{candidate}/{feature}: {actual} != {}",
                                reference[feature]
                            )));
                        }
                    }
                }
                checks += 2;
            }
            // A ninth missing edge closes a red K5. Adding blue leaves a live
            // branch; the second fixture has both colors blocked one edge later.
            let mut nine_red = Graph43::default();
            for a in 0..5 {
                for b in a + 1..5 {
                    if (a, b) != (0, 1) {
                        nine_red.assign(edge_id(a, b).unwrap() as i32)?;
                    }
                }
            }
            let mut refuted = Graph43::default();
            for (vertices, blue) in [
                ([0, 1, 3, 4, 5], false),
                ([0, 2, 8, 9, 10], false),
                ([0, 2, 11, 12, 13], true),
            ] {
                for a in 0..5 {
                    for b in a + 1..5 {
                        if (a, b) != (0, 1) {
                            let literal = edge_id(vertices[a], vertices[b]).unwrap() as i32;
                            refuted.assign(if blue { -literal } else { literal })?;
                        }
                    }
                }
            }
            let fixtures = [empty, nine_red, refuted];
            let mut fixture_rows = vec![0; DATASET_CAPACITY as usize * ROW_WORDS];
            let mut fixture_meta = vec![0; DATASET_CAPACITY as usize * 8];
            for (slot, parent) in fixtures.iter().enumerate() {
                fixture_rows[slot * ROW_WORDS..(slot + 1) * ROW_WORDS]
                    .copy_from_slice(parent.rows());
                fixture_meta[slot * 8 + 1] = slot as u32 + 1;
            }
            self.stream
                .memcpy_htod(&fixture_rows, &mut sampling.rows)
                .map_err(device)?;
            self.stream
                .memcpy_htod(&fixture_meta, &mut sampling.meta)
                .map_err(device)?;
            for horizon in [32, 64] {
                self.launch_dataset_labels(&mut sampling, fixtures.len() as u32, horizon)?;
                let ids = self.stream.clone_dtoh(&sampling.ids).map_err(device)?;
                let cost = self.stream.clone_dtoh(&sampling.cost).map_err(device)?;
                let status = self.stream.clone_dtoh(&sampling.status).map_err(device)?;
                let valid = self.stream.clone_dtoh(&sampling.valid).map_err(device)?;
                for (slot, parent) in fixtures.iter().enumerate() {
                    if valid[slot] != 1 {
                        return Err(Error::Device("fixture dataset inválido".into()));
                    }
                    for candidate in 0..32 {
                        for color in 0..2 {
                            let index = slot * 64 + candidate * 2 + color;
                            let expected = dataset_probe_reference(
                                parent,
                                ids[slot * 32 + candidate],
                                color as u8,
                                horizon,
                            )?;
                            if (cost[index], status[index] as u8) != expected {
                                return Err(Error::Device(format!("sonda dataset {slot}/{candidate}/{color} H{horizon}: ({},{}) != {expected:?}", cost[index], status[index])));
                            }
                            checks += 1;
                        }
                    }
                }
                if status[0] != u32::from(dataset::STATUS_CENSORED)
                    || cost[0] != horizon
                    || status[64] != u32::from(dataset::STATUS_IMMEDIATE_K5)
                    || cost[64] != 1
                    || status[129] != u32::from(dataset::STATUS_REFUTED)
                    || cost[129] != 3
                {
                    return Err(Error::Device(
                        "las fixtures no cubren K5 inmediato, refutación exacta y censura".into(),
                    ));
                }
            }
            // Deliberately saturate the fixed GPU capture arena; exact DFS state
            // still evolves, and every discarded parent has an explicit count.
            self.stream
                .memcpy_htod(&[0_u32, 0], &mut sampling.control)
                .map_err(device)?;
            self.launch_dataset_search(
                &mut captured,
                &mut sampling,
                1,
                0,
                1,
                SearchConfig {
                    steps: 128,
                    ..config
                },
                1,
            )?;
            let control = self.stream.clone_dtoh(&sampling.control).map_err(device)?;
            if control[0] <= DATASET_CAPACITY || control[1] != control[0] - DATASET_CAPACITY {
                return Err(Error::Device(format!(
                    "overflow de captura sin contabilizar: {control:?}"
                )));
            }
            drop(regular);
            drop(captured);
            drop(sampling);
            self.stream.synchronize().map_err(device)?;
            Ok(checks + 1)
        }

        #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
        fn solve_neuro(
            &self,
            graph: &Graph43,
            config: SearchConfig,
            universes: u32,
        ) -> Result<SearchReport, Error> {
            let start = Instant::now();
            let stream = self.context.new_stream().map_err(device)?;
            let mut buffers =
                self.allocate_search_buffers_on_stream(graph, config, universes, &stream)?;
            let engine = if config.neuro_trt {
                #[cfg(feature = "neuro-trt")]
                {
                    let (capacity, engine) =
                        self.trt_policy.borrow_mut().take().ok_or_else(|| {
                            Error::Input("cargar la política TensorRT antes de buscar".into())
                        })?;
                    if capacity != config.batch_warps {
                        self.trt_policy.borrow_mut().replace((capacity, engine));
                        return Err(Error::Input(
                            "batch_warps difiere de la capacidad del engine cargado".into(),
                        ));
                    }
                    engine
                }
                #[cfg(not(feature = "neuro-trt"))]
                return Err(Error::Input("TensorRT no compilado".into()));
            } else {
                #[cfg(feature = "neuro-mock")]
                {
                    risk::RiskEngine::mock(self.context.clone(), config.batch_warps)?
                }
                #[cfg(not(feature = "neuro-mock"))]
                return Err(Error::Input("mock no compilado".into()));
            };
            let mut neuro = self.allocate_neuro_buffers_with_engine(
                universes,
                config.batch_warps,
                stream.clone(),
                engine,
            )?;
            let (mut epochs, mut launches, mut batch_warps) = (0_u64, 0, config.batch_warps);
            let mut launch_steps = config.steps.min(8);
            let mut launch_time = Duration::ZERO;
            let mut max_launch = Duration::ZERO;
            let mut min_batch_warps = u32::MAX;
            let mut max_batch_warps = 0;
            let mut max_launch_steps = 0;
            let mut heartbeat = Instant::now();
            let (mut counters, mut outcome) = ([0_u64; COUNTER_WORDS], Outcome::Unknown);
            // La selección se hace una vez; ninguna bandera extra entra en el DFS.

            let epoch_limit = if config.epochs == 0 {
                "unlimited".to_owned()
            } else {
                config.epochs.to_string()
            };
            'epochs: while config.has_next_epoch(epochs) {
                epochs = epochs.saturating_add(1);
                let mut first = 0;
                while first < universes {
                    let batch = batch_warps.min(universes - first);
                    // Cada lote conserva first/batch durante TODAS sus continuaciones.
                    // La suma de chunks es steps por época, aunque cambie su quantum.
                    let mut remaining = config.steps;
                    while remaining != 0 {
                        let chunk = launch_steps.min(remaining);
                        let launch_start = Instant::now();
                        self.launch_neuro_chunk(
                            &mut buffers,
                            &mut neuro,
                            universes,
                            first,
                            batch,
                            SearchConfig {
                                steps: chunk,
                                ..config
                            },
                        )?;
                        launches += 1;
                        let elapsed = launch_start.elapsed();
                        launch_time += elapsed;
                        max_launch = max_launch.max(elapsed);
                        min_batch_warps = min_batch_warps.min(batch);
                        max_batch_warps = max_batch_warps.max(batch);
                        max_launch_steps = max_launch_steps.max(chunk);
                        remaining -= chunk;
                        // Calibración conservadora: quantum inicial 8, techo 512.
                        // El tamaño de lote nunca supera el solicitado por CLI.
                        if elapsed > Duration::from_millis(100) {
                            launch_steps = (launch_steps / 2).max(1);
                            batch_warps = (batch_warps / 2).max(1);
                        } else if elapsed < Duration::from_millis(10) {
                            launch_steps = (launch_steps * 2).min(config.steps).min(512);
                            batch_warps = (batch_warps * 2).min(config.batch_warps);
                        }
                        let found = stream.clone_dtoh(&buffers.found).map_err(device)?[0];
                        if found == 2 {
                            let rows = stream.clone_dtoh(&buffers.model).map_err(device)?;
                            let model = Graph43::from_rows(rows.try_into().map_err(|_| {
                                Error::Device("modelo GPU de longitud incorrecta".into())
                            })?)?;
                            if !model.extends(graph)
                                || self.check(std::slice::from_ref(&model))?
                                    != [CheckResult::CompleteSafe]
                            {
                                return Err(Error::Device(
                                    "el modelo publicado no es una extensión completa sin K5"
                                        .into(),
                                ));
                            }
                            model.model()?;
                            outcome = Outcome::Sat(model);
                            break 'epochs;
                        }
                        if found != 0 {
                            return Err(Error::Device(format!(
                                "publicación incompleta de modelo: found={found}"
                            )));
                        }
                    }
                    first += batch;
                    if heartbeat.elapsed() >= Duration::from_secs(5) {
                        let totals = stream.clone_dtoh(&buffers.counters).map_err(device)?;
                        if config.epochs == 0 {
                            let controls = stream.clone_dtoh(&buffers.control).map_err(device)?;
                            reject_universe_errors(&controls)?;
                        }
                        eprintln!("c progress: t={:.1}s; epoch={epochs}/{epoch_limit}; universes_visited={first}/{universes}; transitions={}; prunes={}; completed={}; batch={batch}; quantum={launch_steps}; max_launch_ms={:.3}",
                            start.elapsed().as_secs_f64(), totals[0], totals[1], totals[2], max_launch.as_secs_f64() * 1000.0);
                        if config.lookahead {
                            eprintln!("c inference: decisions={}; forced_assignments={}; lookahead_checks={}; lookahead_conflicts={}", totals[3], totals[4], totals[5], totals[6]);
                        }
                        let policy_totals = stream.clone_dtoh(&neuro.counters).map_err(device)?;
                        eprintln!(
                            "c neuro-policy: queries={}; responses={}; fallbacks={}",
                            policy_totals[0], policy_totals[1], policy_totals[2]
                        );
                        heartbeat = Instant::now();
                    }
                }
                let totals = stream.clone_dtoh(&buffers.counters).map_err(device)?;
                counters.copy_from_slice(&totals);
                if counters[2] > u64::from(universes) {
                    return Err(Error::Device(
                        "contador de universos agotados desbordado".into(),
                    ));
                }
                if counters[2] == u64::from(universes) {
                    let controls = stream.clone_dtoh(&buffers.control).map_err(device)?;
                    if controls
                        .chunks_exact(CONTROL_WORDS_PER_UNIVERSE)
                        .all(|control| matches!(control[3], 1 | 3))
                    {
                        outcome = Outcome::Unsat;
                    } else {
                        return Err(Error::Device(
                            "contador de agotamiento incompatible con los estados de los universos"
                                .into(),
                        ));
                    }
                    break;
                }
            }
            let totals = stream.clone_dtoh(&buffers.counters).map_err(device)?;
            counters.copy_from_slice(&totals);
            // Un error de estado no debe quedar oculto como presupuesto agotado.
            // Se descarga al salir y en latidos continuos, nunca por paso DFS.
            let controls = stream.clone_dtoh(&buffers.control).map_err(device)?;
            reject_universe_errors(&controls)?;
            // cudaFreeAsync puede conservar la arena en el pool hasta sincronizar.
            // Liberarla antes del informe evita falsas faltas de VRAM en la clase siguiente.
            let policy_totals = stream.clone_dtoh(&neuro.counters).map_err(device)?;
            drop(buffers);
            let NeuroBuffers {
                engine,
                ids,
                meta,
                policy,
                generation,
                counters: policy_counters,
                features_ready,
                scores_ready,
                stream: arena_stream,
                batch_capacity: _,
            } = neuro;
            drop((
                ids,
                meta,
                policy,
                generation,
                policy_counters,
                features_ready,
                scores_ready,
                arena_stream,
            ));
            stream.synchronize().map_err(device)?;
            #[cfg(feature = "neuro-trt")]
            if config.neuro_trt {
                self.trt_policy
                    .borrow_mut()
                    .replace((config.batch_warps, engine));
            } else {
                drop(engine);
            }
            #[cfg(not(feature = "neuro-trt"))]
            drop(engine);
            Ok(SearchReport {
                outcome,
                universes,
                epochs,
                launches,
                transitions: counters[0],
                prunes: counters[1],
                completed: counters[2],
                decisions: counters[3],
                forced_assignments: counters[4],
                lookahead_checks: counters[5],
                lookahead_conflicts: counters[6],
                policy_queries: policy_totals[0],
                policy_responses: policy_totals[1],
                policy_fallbacks: policy_totals[2],
                elapsed: start.elapsed(),
                launch_time,
                max_launch,
                min_batch_warps,
                max_batch_warps,
                max_launch_steps,
            })
        }
        #[cfg(feature = "neuro-mock")]
        pub fn validate_neuro(&self) -> Result<usize, Error> {
            let checks = risk::RiskEngine::validate_mock_contracts(self.context.clone())?;
            Ok(checks + self.validate_neuro_features()? + self.validate_neuro_choices()?)
        }
        #[cfg(feature = "neuro-mock")]
        fn validate_neuro_features(&self) -> Result<usize, Error> {
            let stream = self.context.new_stream().map_err(device)?;
            let config = SearchConfig {
                prefix_bits: 0,
                steps: 1,
                epochs: 1,
                batch_warps: 4,
                explicit_trail: true,
                lookahead: true,
                neuro_mock: true,
                neuro_trt: false,
                ..SearchConfig::default()
            };
            let mut dense = Graph43::default();
            for index in 0..EDGES {
                if [0, 50, 100, 850, 902].contains(&index) {
                    continue;
                }
                let literal = (index + 1) as i32;
                dense.assign(if index % 2 == 0 { literal } else { -literal })?;
            }
            let graphs = [
                Graph43::default(),
                canonical_star(18)?,
                dense,
                Graph43::default(),
            ];
            let depths = [0, 7, 5, 0];
            let generations = [0, 0x1234_5678_9abc_def0_u64, 5, 7];
            let mut buffers =
                self.allocate_search_buffers_on_stream(&Graph43::default(), config, 4, &stream)?;
            let mut neuro = self.allocate_neuro_buffers(4, 4, stream.clone())?;
            let flat: Vec<_> = graphs.iter().flat_map(|graph| graph.rows).collect();
            let controls: Vec<_> = depths
                .iter()
                .flat_map(|&depth| [1, depth, 0, 0, 0])
                .collect();
            stream
                .memcpy_htod(&flat, &mut buffers.rows)
                .map_err(device)?;
            stream
                .memcpy_htod(&controls, &mut buffers.control)
                .map_err(device)?;
            stream
                .memcpy_htod(&vec![0_u32, 1, 0].repeat(4), &mut neuro.policy)
                .map_err(device)?;
            stream
                .memcpy_htod(&generations, &mut neuro.generation)
                .map_err(device)?;
            self.launch_neuro_pack(&buffers, &mut neuro, 4, 1, 3)?;
            stream.synchronize().map_err(device)?;
            let x = stream
                .clone_dtoh(neuro.engine.input_mut())
                .map_err(device)?;
            let ids = stream.clone_dtoh(&neuro.ids).map_err(device)?;
            let meta = stream.clone_dtoh(&neuro.meta).map_err(device)?;
            for slot in 0..4 {
                if slot == 3 {
                    if meta[12..16] != [u32::MAX, 0, 0, 0]
                        || ids[96..128].iter().any(|&id| id != EDGES as u16)
                        || x[3072..4096].iter().any(|&bits| bits != 0)
                    {
                        return Err(Error::Device(
                            "padding de slot neuro no inicializado".into(),
                        ));
                    }
                    continue;
                }
                let universe = slot + 1;
                let expected_meta = [
                    universe as u32,
                    generations[universe] as u32,
                    (generations[universe] >> 32) as u32,
                    depths[universe],
                ];
                if meta[slot * 4..slot * 4 + 4] != expected_meta {
                    return Err(Error::Device(format!(
                        "meta neuro slot {slot} != snapshot: {:?}",
                        &meta[slot * 4..slot * 4 + 4]
                    )));
                }
                let expected_ids: Vec<_> = (0..EDGES)
                    .filter(|&index| {
                        let (a, b) = edge_vertices(index + 1).unwrap();
                        (graphs[universe].rows[a] | graphs[universe].rows[VERTICES + a]) >> b & 1
                            == 0
                    })
                    .take(32)
                    .map(|index| index as u16)
                    .collect();
                for candidate in 0..32 {
                    let id = expected_ids.get(candidate).copied().unwrap_or(EDGES as u16);
                    if ids[slot * 32 + candidate] != id {
                        return Err(Error::Device(format!(
                            "candidato neuro slot {slot}/{candidate}: ID incorrecto"
                        )));
                    }
                    let expected = feature_reference(&graphs[universe], depths[universe], id);
                    let row = &x[(slot * 32 + candidate) * 32..(slot * 32 + candidate + 1) * 32];
                    for (feature, (&bits, &value)) in row.iter().zip(&expected).enumerate() {
                        let actual = decode_feature_half(bits);
                        if !actual.is_finite()
                            || (actual - value).abs() > 0.0005
                            || (feature >= 20 && bits != 0)
                            || (id == EDGES as u16 && bits != 0)
                        {
                            return Err(Error::Device(format!("feature neuro {slot}/{candidate}/{feature}: {actual} != {value}, half={bits:#06x}")));
                        }
                    }
                }
            }
            Ok(97)
        }
        #[cfg(feature = "neuro-mock")]
        fn validate_neuro_choices(&self) -> Result<usize, Error> {
            let config = SearchConfig {
                prefix_bits: 0,
                steps: 1,
                epochs: 1,
                batch_warps: 2,
                explicit_trail: true,
                lookahead: true,
                neuro_mock: true,
                neuro_trt: false,
                policy_interval: 32,
                ..SearchConfig::default()
            };
            let empty = Graph43::default();
            let stream = self.context.new_stream().map_err(device)?;
            let mut buffers = self.allocate_search_buffers_on_stream(&empty, config, 1, &stream)?;
            let mut neuro = self.allocate_neuro_buffers(1, 2, stream.clone())?;
            self.launch_neuro_chunk(&mut buffers, &mut neuro, 1, 0, 1, config)?;
            let policy = stream.clone_dtoh(&neuro.policy).map_err(device)?;
            let controls = stream.clone_dtoh(&buffers.control).map_err(device)?;
            if policy[1] != 2 || policy[2] != (31 | 0x0400) || controls[1] != 0 {
                return Err(Error::Device(format!(
                    "mock no dejó READY edge31 azul con grafo congelado: {policy:?}/{controls:?}"
                )));
            }
            self.launch_neuro_chunk(&mut buffers, &mut neuro, 1, 0, 1, config)?;
            self.launch_neuro_chunk(&mut buffers, &mut neuro, 1, 0, 1, config)?;
            let journal = match &buffers.trail {
                TrailBuffers::Explicit { journal, .. } => {
                    stream.clone_dtoh(journal).map_err(device)?
                }
                TrailBuffers::Implicit(_) => unreachable!(),
            };
            let controls = stream.clone_dtoh(&buffers.control).map_err(device)?;
            let totals = stream.clone_dtoh(&neuro.counters).map_err(device)?;
            if journal[..2] != [31 | 0x0400, 0] || controls[1] != 2 || totals != [1, 1, 0] {
                return Err(Error::Device(format!("mock/R32 no conservó edge31 azul y respaldo edge0: {:?}/{controls:?}/{totals:?}", &journal[..2])));
            }
            drop(neuro);
            drop(buffers);
            stream.synchronize().map_err(device)?;

            // El mock prefiere azul para edge31=(0,32), pero cerraría un K5:
            // el oráculo lo convierte en rojo forzado con alternativa consumida.
            let mut forced = Graph43::default();
            let vertices = [0, 32, 34, 35, 36];
            for a in 0..5 {
                for b in a + 1..5 {
                    if (a, b) != (0, 1) {
                        forced.assign(-(edge_id(vertices[a], vertices[b]).unwrap() as i32))?;
                    }
                }
            }
            let mut buffers =
                self.allocate_search_buffers_on_stream(&forced, config, 1, &stream)?;
            let mut neuro = self.allocate_neuro_buffers(1, 2, stream.clone())?;
            self.launch_neuro_chunk(&mut buffers, &mut neuro, 1, 0, 1, config)?;
            self.launch_neuro_chunk(&mut buffers, &mut neuro, 1, 0, 1, config)?;
            let journal = match &buffers.trail {
                TrailBuffers::Explicit { journal, .. } => {
                    stream.clone_dtoh(journal).map_err(device)?
                }
                TrailBuffers::Implicit(_) => unreachable!(),
            };
            let rows = stream.clone_dtoh(&buffers.rows).map_err(device)?;
            if journal[0] != 0x181f || rows[0] >> 32 & 1 != 1 || rows[43] >> 32 & 1 != 0 {
                return Err(Error::Device(format!(
                    "la política saltó legalidad exacta: frame={:#06x}",
                    journal[0]
                )));
            }
            drop(neuro);
            drop(buffers);
            stream.synchronize().map_err(device)?;

            // Una elección no léxica debe retroceder usando el ID/color del
            // journal: conflicto edge3, pop del forzado edge0, flip de edge31.
            let rollback_base = parse_partial_dimacs("v -2 -3 5 6 -7 -8 -9 -43 -44 45 46 47 -73 -84 -113 -152 163 164 -165 -166 -167 201 -274 -275 -309 0")?;
            if check_reference(&rollback_base) != CheckResult::Open {
                return Err(Error::Device(
                    "base del retroceso no léxico contiene K5".into(),
                ));
            }
            let mut buffers =
                self.allocate_search_buffers_on_stream(&rollback_base, config, 1, &stream)?;
            let mut neuro = self.allocate_neuro_buffers(1, 2, stream.clone())?;
            self.launch_neuro_search(&mut buffers, &mut neuro, 1, 0, 1, config)?;
            self.launch_neuro_pack(&buffers, &mut neuro, 1, 0, 1)?;
            stream.synchronize().map_err(device)?;
            let policy = stream.clone_dtoh(&neuro.policy).map_err(device)?;
            let control = stream.clone_dtoh(&buffers.control).map_err(device)?;
            let ids = stream.clone_dtoh(&neuro.ids).map_err(device)?;
            let rows = stream.clone_dtoh(&buffers.rows).map_err(device)?;
            let initial_graph =
                Graph43::from_rows(rows.try_into().map_err(|_| {
                    Error::Device("snapshot inicial del retroceso incompleto".into())
                })?)?;
            if policy[1] != 1
                || control != [1, 0, 0, 0, 0]
                || ids[24] != 31
                || initial_graph != rollback_base
            {
                return Err(Error::Device(format!("snapshot inicial del retroceso no léxico inválido: {policy:?}/{control:?}/row24={}", ids[24])));
            }
            let mut scores = vec![1.0_f32; 2 * 32 * 2];
            scores[24 * 2 + 1] = -1.0;
            let scores = stream.clone_htod(&scores).map_err(device)?;
            self.launch_neuro_apply(&buffers, &mut neuro, 1, 0, 1, Some(&scores))?;
            stream.synchronize().map_err(device)?;
            let policy = stream.clone_dtoh(&neuro.policy).map_err(device)?;
            if policy[1] != 2 || policy[2] != 0x041f {
                return Err(Error::Device(format!(
                    "override row24 azul no seleccionó edge31: {policy:?}"
                )));
            }
            let expected_stages: &[(&[u16], u32)] = &[
                (&[0x041f], 0),
                (&[0x041f, 0x1800], 0),
                (&[0x041f, 0x1800], 1),
                (&[0x041f], 1),
                (&[0x081f], 0),
            ];
            let mut final_snapshot = None;
            for (stage, &(expected_frames, expected_backtrack)) in
                expected_stages.iter().enumerate()
            {
                self.launch_neuro_search(&mut buffers, &mut neuro, 1, 0, 1, config)?;
                stream.synchronize().map_err(device)?;
                let control = stream.clone_dtoh(&buffers.control).map_err(device)?;
                let rows = stream.clone_dtoh(&buffers.rows).map_err(device)?;
                let counters = stream.clone_dtoh(&buffers.counters).map_err(device)?;
                let journal = match &buffers.trail {
                    TrailBuffers::Explicit { journal, .. } => {
                        stream.clone_dtoh(journal).map_err(device)?
                    }
                    TrailBuffers::Implicit(_) => unreachable!(),
                };
                let expected_control = [1, expected_frames.len() as u32, 0, 0, expected_backtrack];
                if control != expected_control
                    || journal[..expected_frames.len()] != *expected_frames
                {
                    return Err(Error::Device(format!("retroceso no léxico etapa {stage}: control={control:?}, frames={:?}; esperado {expected_control:?}/{expected_frames:?}", &journal[..expected_frames.len()])));
                }
                let snapshot = JournalSnapshot {
                    graphs: vec![Graph43::from_rows(rows.try_into().map_err(|_| {
                        Error::Device("filas del retroceso incompletas".into())
                    })?)?],
                    frames: vec![journal[..expected_frames.len()].to_vec()],
                    controls: vec![expected_control],
                    counters: counters.try_into().map_err(|_| {
                        Error::Device("contadores del retroceso incompletos".into())
                    })?,
                };
                Self::check_journal_snapshot(&snapshot, std::slice::from_ref(&rollback_base))?;
                final_snapshot = Some(snapshot);
            }
            let final_snapshot = final_snapshot.expect("cinco etapas de retroceso");
            let mut expected_graph = rollback_base;
            expected_graph.assign(32)?;
            let policy = stream.clone_dtoh(&neuro.policy).map_err(device)?;
            let generation = stream.clone_dtoh(&neuro.generation).map_err(device)?;
            let policy_totals = stream.clone_dtoh(&neuro.counters).map_err(device)?;
            if final_snapshot.graphs != [expected_graph]
                || final_snapshot.counters != [6, 1, 0, 2, 1, 5, 1]
                || policy[..2] != [2, 0]
                || generation != [1]
                || policy_totals != [1, 1, 0]
            {
                return Err(Error::Device(format!("retroceso no léxico dejó estado/alternativa incorrectos: {final_snapshot:?}/{policy:?}/{generation:?}/{policy_totals:?}")));
            }
            drop(neuro);
            drop(buffers);
            drop(scores);
            stream.synchronize().map_err(device)?;

            // Cada respuesta defectuosa debe desbloquear WAIT y avanzar usando
            // respaldo, sin declarar conflicto, agotamiento ni UNSAT.
            for fault in 0..7 {
                let mut base = empty.clone();
                if fault == 2 {
                    base.assign(1)?;
                }
                let mut buffers =
                    self.allocate_search_buffers_on_stream(&base, config, 1, &stream)?;
                let mut neuro = self.allocate_neuro_buffers(1, 2, stream.clone())?;
                self.launch_neuro_search(&mut buffers, &mut neuro, 1, 0, 1, config)?;
                self.launch_neuro_pack(&buffers, &mut neuro, 1, 0, 1)?;
                stream.synchronize().map_err(device)?;
                let mut ids = stream.clone_dtoh(&neuro.ids).map_err(device)?;
                let mut meta = stream.clone_dtoh(&neuro.meta).map_err(device)?;
                let mut scores = vec![1.0_f32; 2 * 32 * 2];
                match fault {
                    0 => scores[0] = f32::NAN,
                    1 => scores[1] = f32::INFINITY,
                    2 => ids[0] = 0, // ID obsoleto: arista ya asignada en la base.
                    3 => ids[1] = ids[0],
                    4 => meta[1] = meta[1].wrapping_add(1),
                    5 => meta[0] = u32::MAX,
                    _ => ids[..32].fill(EDGES as u16),
                }
                stream.memcpy_htod(&ids, &mut neuro.ids).map_err(device)?;
                stream.memcpy_htod(&meta, &mut neuro.meta).map_err(device)?;
                let scores = stream.clone_htod(&scores).map_err(device)?;
                self.launch_neuro_apply(&buffers, &mut neuro, 1, 0, 1, Some(&scores))?;
                stream.synchronize().map_err(device)?;
                let policy = stream.clone_dtoh(&neuro.policy).map_err(device)?;
                if policy[1] != 2 || policy[2] != 0xffff {
                    return Err(Error::Device(format!(
                        "respuesta defectuosa {fault} no activó READY respaldo: {policy:?}"
                    )));
                }
                self.launch_neuro_search(&mut buffers, &mut neuro, 1, 0, 1, config)?;
                stream.synchronize().map_err(device)?;
                let control = stream.clone_dtoh(&buffers.control).map_err(device)?;
                let totals = stream.clone_dtoh(&neuro.counters).map_err(device)?;
                let counters = stream.clone_dtoh(&buffers.counters).map_err(device)?;
                if control[1] != 1
                    || control[3] != 0
                    || control[4] != 0
                    || counters[2] != 0
                    || totals != [1, 1, 1]
                {
                    return Err(Error::Device(format!("respuesta defectuosa {fault} alteró búsqueda: {control:?}/{totals:?}/{counters:?}")));
                }
                drop(neuro);
                drop(buffers);
                drop(scores);
                stream.synchronize().map_err(device)?;
            }
            Ok(11)
        }
        #[cfg(feature = "neuro-mock")]
        fn allocate_neuro_buffers(
            &self,
            universes: u32,
            batch_capacity: u32,
            stream: Arc<CudaStream>,
        ) -> Result<NeuroBuffers, Error> {
            let engine = risk::RiskEngine::mock(self.context.clone(), batch_capacity)?;
            self.allocate_neuro_buffers_with_engine(universes, batch_capacity, stream, engine)
        }
        #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
        fn allocate_neuro_buffers_with_engine(
            &self,
            universes: u32,
            batch_capacity: u32,
            stream: Arc<CudaStream>,
            engine: risk::RiskEngine,
        ) -> Result<NeuroBuffers, Error> {
            // Estado persistente U*(3*u32+u64), un slot B*(X+Y+IDs+meta).
            let needed = u64::from(universes) * 20 + u64::from(batch_capacity) * 2384 + 24;
            self.reserve(needed)?;
            Ok(NeuroBuffers {
                ids: stream
                    .alloc_zeros(batch_capacity as usize * 32)
                    .map_err(device)?,
                meta: stream
                    .alloc_zeros(batch_capacity as usize * 4)
                    .map_err(device)?,
                policy: stream.alloc_zeros(universes as usize * 3).map_err(device)?,
                generation: stream.alloc_zeros(universes as usize).map_err(device)?,
                counters: stream.alloc_zeros(3).map_err(device)?,
                features_ready: self.context.new_event(None).map_err(device)?,
                scores_ready: self.context.new_event(None).map_err(device)?,
                stream,
                engine,
                batch_capacity,
            })
        }
        #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
        fn launch_neuro_chunk(
            &self,
            buffers: &mut SearchBuffers,
            neuro: &mut NeuroBuffers,
            universes: u32,
            first: u32,
            count: u32,
            config: SearchConfig,
        ) -> Result<(), Error> {
            self.launch_neuro_search(buffers, neuro, universes, first, count, config)?;
            self.launch_neuro_pack(buffers, neuro, universes, first, count)?;
            neuro.features_ready.record(&neuro.stream).map_err(device)?;
            let inference_stream = neuro.engine.inference_stream().clone();
            inference_stream
                .wait(&neuro.features_ready)
                .map_err(device)?;
            neuro.engine.enqueue()?;
            neuro
                .scores_ready
                .record(&inference_stream)
                .map_err(device)?;
            neuro.stream.wait(&neuro.scores_ready).map_err(device)?;
            self.launch_neuro_apply(buffers, neuro, universes, first, count, None)?;
            // Único slot: este límite existente de timing acaba su consumo antes
            // de sobrescribir X/Y/IDs/meta en el siguiente chunk.
            neuro.stream.synchronize().map_err(device)
        }
        #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
        #[allow(unsafe_code)]
        fn launch_neuro_search(
            &self,
            buffers: &mut SearchBuffers,
            neuro: &mut NeuroBuffers,
            universes: u32,
            first: u32,
            count: u32,
            config: SearchConfig,
        ) -> Result<(), Error> {
            if !(config.neuro_mock || config.neuro_trt)
                || !config.explicit_trail
                || !config.lookahead
                || count == 0
                || count > neuro.batch_capacity
                || first.checked_add(count).is_none_or(|end| end > universes)
            {
                return Err(Error::Input("lote neuro fuera del contrato".into()));
            }
            let (journal, capacity) = match &mut buffers.trail {
                TrailBuffers::Explicit { journal, capacity } => (journal, *capacity),
                TrailBuffers::Implicit(_) => {
                    return Err(Error::Device("neuro requiere journal explícito".into()))
                }
            };
            let mut call = neuro.stream.launch_builder(&self.search_neuro);
            call.arg(&buffers.base)
                .arg(&mut buffers.rows)
                .arg(journal)
                .arg(&mut buffers.control)
                .arg(&universes)
                .arg(&first)
                .arg(&count)
                .arg(&config.prefix_bits)
                .arg(&config.order_seed)
                .arg(&config.steps)
                .arg(&capacity)
                .arg(&mut buffers.found)
                .arg(&mut buffers.model)
                .arg(&mut buffers.counters)
                .arg(&mut neuro.policy)
                .arg(&mut neuro.generation)
                .arg(&config.policy_interval)
                .arg(&mut neuro.counters);
            // SAFETY: ABI journal más policy[U*3], generation[U], intervalo
            // validado y contadores[3]. El warp WAIT sale sin esperar otro kernel.
            unsafe {
                call.launch(LaunchConfig {
                    grid_dim: (count.div_ceil(8), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(device)?;
            Ok(())
        }
        #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
        #[allow(unsafe_code)]
        fn launch_neuro_pack(
            &self,
            buffers: &SearchBuffers,
            neuro: &mut NeuroBuffers,
            universes: u32,
            first: u32,
            count: u32,
        ) -> Result<(), Error> {
            let mut call = neuro.stream.launch_builder(&self.pack_features);
            call.arg(&buffers.rows)
                .arg(&buffers.control)
                .arg(&neuro.policy)
                .arg(&neuro.generation)
                .arg(&universes)
                .arg(&first)
                .arg(&count)
                .arg(&neuro.batch_capacity)
                .arg(neuro.engine.input_mut())
                .arg(&mut neuro.ids)
                .arg(&mut neuro.meta);
            // SAFETY: X[B*32*32] half bits, IDs[B*32], meta[B*4]. El grid
            // cubre también slots de relleno; los universos WAIT no mutan.
            unsafe {
                call.launch(LaunchConfig {
                    grid_dim: (neuro.batch_capacity.div_ceil(8), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(device)?;
            Ok(())
        }
        #[cfg(any(feature = "neuro-mock", feature = "neuro-trt"))]
        #[allow(unsafe_code)]
        fn launch_neuro_apply(
            &self,
            buffers: &SearchBuffers,
            neuro: &mut NeuroBuffers,
            universes: u32,
            first: u32,
            count: u32,
            scores_override: Option<&CudaSlice<f32>>,
        ) -> Result<(), Error> {
            let scores = match scores_override {
                Some(scores) => scores,
                None => neuro.engine.output(),
            };
            let mut call = neuro.stream.launch_builder(&self.apply_scores);
            call.arg(&buffers.rows)
                .arg(&buffers.control)
                .arg(&mut neuro.policy)
                .arg(&neuro.generation)
                .arg(&universes)
                .arg(&first)
                .arg(&count)
                .arg(&neuro.ids)
                .arg(&neuro.meta)
                .arg(scores)
                .arg(&mut neuro.counters);
            // SAFETY: scores_ready ordena Y[B*32*2] FP32. Snapshot e IDs se
            // validan en GPU; inválidos producen READY de respaldo, nunca UNSAT.
            unsafe {
                call.launch(LaunchConfig {
                    grid_dim: (count.div_ceil(8), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(device)?;
            Ok(())
        }
        #[allow(unsafe_code)]
        fn launch_search(
            &self,
            search: &CudaFunction,
            buffers: &mut SearchBuffers,
            universes: u32,
            first: u32,
            count: u32,
            config: SearchConfig,
        ) -> Result<(), Error> {
            let journal_capacity = match &buffers.trail {
                TrailBuffers::Implicit(_) => None,
                TrailBuffers::Explicit { capacity, .. } => Some(*capacity),
            };
            if journal_capacity.is_some() != config.explicit_trail {
                return Err(Error::Device(
                    "arena y kernel de trail incompatibles".into(),
                ));
            }
            let mut call = self.stream.launch_builder(search);
            call.arg(&buffers.base).arg(&mut buffers.rows);
            match &mut buffers.trail {
                TrailBuffers::Implicit(alt) => {
                    call.arg(alt);
                }
                TrailBuffers::Explicit { journal, .. } => {
                    call.arg(journal);
                }
            }
            call.arg(&mut buffers.control)
                .arg(&universes)
                .arg(&first)
                .arg(&count)
                .arg(&config.prefix_bits)
                .arg(&config.order_seed)
                .arg(&config.steps);
            if let Some(capacity) = journal_capacity.as_ref() {
                call.arg(capacity);
            }
            call.arg(&mut buffers.found)
                .arg(&mut buffers.model)
                .arg(&mut buffers.counters);
            // SAFETY: firma exacta del kernel seleccionado. Buffers U*86 u64,
            // U*32 u32 o U*max(capacidad,1) u16, y U*5 u32; first+count<=U.
            // La ABI explícita añade capacidad u32 tras steps. Un warp por universo,
            // atómicos en modelo/contadores compartidos. Este stream espera cada lanzamiento.
            unsafe {
                call.launch(LaunchConfig {
                    grid_dim: (count.div_ceil(8), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(device)?;
            self.stream.synchronize().map_err(device)
        }
    }
}

#[cfg(not(feature = "gpu"))]
mod backend {
    use super::*;
    pub enum Device {}
    impl Device {
        pub fn open(_ordinal: usize) -> Result<(Self, String), Error> {
            Err(Error::NotCompiled)
        }
        pub fn validate_kernel_resources(&self) -> Result<(), Error> {
            match *self {}
        }
        pub fn validate_explicit_trail(&self) -> Result<usize, Error> {
            match *self {}
        }
        pub fn check(&self, _graphs: &[Graph43]) -> Result<Vec<CheckResult>, Error> {
            match *self {}
        }
        pub fn allowed_colours(
            &self,
            _graphs: &[Graph43],
            _edges: &[(usize, usize)],
        ) -> Result<Vec<u32>, Error> {
            match *self {}
        }
        pub fn solve(
            &self,
            _graph: &Graph43,
            _config: SearchConfig,
            _universes: u32,
        ) -> Result<SearchReport, Error> {
            match *self {}
        }
    }
}

/// Referencia independiente O(43^5) sólo para auditorías y tests, nunca para ramificar.
pub fn check_reference(graph: &Graph43) -> CheckResult {
    for colour in 0..2 {
        for a in 0..VERTICES - 4 {
            for b in a + 1..VERTICES - 3 {
                if graph.rows[colour * VERTICES + a] >> b & 1 == 0 {
                    continue;
                }
                for c in b + 1..VERTICES - 2 {
                    if [a, b]
                        .iter()
                        .any(|&v| graph.rows[colour * VERTICES + v] >> c & 1 == 0)
                    {
                        continue;
                    }
                    for d in c + 1..VERTICES - 1 {
                        if [a, b, c]
                            .iter()
                            .any(|&v| graph.rows[colour * VERTICES + v] >> d & 1 == 0)
                        {
                            continue;
                        }
                        for e in d + 1..VERTICES {
                            if [a, b, c, d]
                                .iter()
                                .all(|&v| graph.rows[colour * VERTICES + v] >> e & 1 != 0)
                            {
                                return CheckResult::Bad {
                                    blue: colour != 0,
                                    vertices: [a, b, c, d, e],
                                };
                            }
                        }
                    }
                }
            }
        }
    }
    if graph.free_edges() == 0 {
        CheckResult::CompleteSafe
    } else {
        CheckResult::Open
    }
}

/// Independent CPU probe used only by the explicit GPU dataset self-test.
#[cfg(feature = "neuro-dataset")]
fn dataset_probe_reference(
    parent: &Graph43,
    edge: u16,
    color: u8,
    horizon: u32,
) -> Result<(u32, u8), Error> {
    use dataset::{
        STATUS_CENSORED, STATUS_IMMEDIATE_K5, STATUS_PADDING, STATUS_REFUTED, STATUS_SAT,
    };
    if usize::from(edge) >= EDGES {
        return Ok((0, STATUS_PADDING));
    }
    fn allowed(graph: &Graph43, edge: usize) -> Result<u8, Error> {
        let (a, b) = edge_vertices(edge + 1).expect("reference edge validated");
        let mut mask = 0;
        for color in 0..2 {
            if graph.rows[color * VERTICES + a].count_ones() >= 24
                || graph.rows[color * VERTICES + b].count_ones() >= 24
            {
                continue;
            }
            let mut child = graph.clone();
            let literal = (edge + 1) as i32;
            child.assign(if color == 0 { literal } else { -literal })?;
            if !matches!(check_reference(&child), CheckResult::Bad { .. }) {
                mask |= 1 << color;
            }
        }
        Ok(mask)
    }
    fn clear(graph: &mut Graph43, edge: usize) {
        let (a, b) = edge_vertices(edge + 1).expect("reference edge validated");
        for color in 0..2 {
            graph.rows[color * VERTICES + a] &= !(1 << b);
            graph.rows[color * VERTICES + b] &= !(1 << a);
        }
    }
    let mut graph = parent.clone();
    let literal = (usize::from(edge) + 1) as i32;
    graph.assign(if color == 0 { literal } else { -literal })?;
    if matches!(check_reference(&graph), CheckResult::Bad { .. }) {
        return Ok((1, STATUS_IMMEDIATE_K5));
    }
    if allowed(parent, usize::from(edge))? & (1 << color) == 0 {
        return Ok((1, STATUS_REFUTED));
    }
    let mut trail: Vec<(usize, u8, bool, bool)> = Vec::new();
    let mut backtrack = false;
    let mut work = 1;
    while work < horizon {
        work += 1;
        let (edge, color, alternate, forced) = if backtrack {
            let Some((edge, color, alternate, forced)) = trail.pop() else {
                return Ok((work, STATUS_REFUTED));
            };
            clear(&mut graph, edge);
            if alternate || forced {
                continue;
            }
            let color = color ^ 1;
            if allowed(&graph, edge)? & (1 << color) == 0 {
                continue;
            }
            (edge, color, true, false)
        } else {
            let next = (0..EDGES).find(|&edge| {
                let (a, b) = edge_vertices(edge + 1).expect("reference enumerated edge");
                (graph.rows[a] | graph.rows[VERTICES + a]) >> b & 1 == 0
            });
            let Some(edge) = next else {
                if check_reference(&graph) != CheckResult::CompleteSafe {
                    return Err(Error::Device(
                        "CPU dataset reference reached invalid model".into(),
                    ));
                }
                return Ok((work, STATUS_SAT));
            };
            let mask = allowed(&graph, edge)?;
            if mask == 0 {
                backtrack = true;
                continue;
            }
            let forced = mask != 3;
            (edge, u8::from(mask == 2), forced, forced)
        };
        let literal = (edge + 1) as i32;
        graph.assign(if color == 0 { literal } else { -literal })?;
        trail.push((edge, color, alternate, forced));
        backtrack = false;
    }
    Ok((horizon, STATUS_CENSORED))
}

/// Validación CUDA explícita: no se ejecuta al arrancar una búsqueda normal.
pub fn self_test(device: &NativeRamsey43) -> Result<usize, Error> {
    device.device.validate_kernel_resources()?;
    let mut graphs = vec![Graph43::default()];
    for (vertices, blue) in [([0, 1, 2, 3, 4], false), ([32, 34, 38, 41, 42], true)] {
        let mut graph = Graph43::default();
        for a in 0..5 {
            for b in a + 1..5 {
                let id = edge_id(vertices[a], vertices[b]).expect("vértices de test") as i32;
                graph.assign(if blue { -id } else { id })?;
            }
        }
        graphs.push(graph);
    }
    let mut seed = 0x82bd_920a_7fc5_1189_u64;
    for sample in 0..48 {
        let mut graph = Graph43::default();
        for a in 0..8 {
            for b in a + 1..8 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                // Alterna sesgos de color: aparecen K5 y fragmentos abiertos.
                // Mantener unas aristas desconocidas prueba que azul no es ~rojo.
                let choice = seed % 16;
                if choice != 0 {
                    let id = edge_id(a + 32, b + 32).expect("vértices de test") as i32;
                    let red = (choice <= 11) ^ (sample % 2 != 0);
                    graph.assign(if red { id } else { -id })?;
                }
            }
        }
        graphs.push(graph);
    }
    let mut full_red = Graph43::default();
    for literal in 1..=EDGES as i32 {
        full_red.assign(literal)?;
    }
    graphs.push(full_red.clone());
    let results = device.check(&graphs)?;
    for (index, (graph, actual)) in graphs.iter().zip(results).enumerate() {
        let expected = check_reference(graph);
        let matches = match (&actual, &expected) {
            (CheckResult::Bad { blue: a, .. }, CheckResult::Bad { blue: b, .. }) => a == b,
            _ => actual == expected,
        };
        if !matches {
            return Err(Error::Device(format!(
                "self-test checker {index}: {actual:?} != {expected:?}"
            )));
        }
    }
    // x_(0,1)=rojo completa un K5 rojo; azul completa un K5 azul diferente.
    // Las dos ramas se refutan aunque las restantes aristas de K43 sean libres.
    let mut contradiction = Graph43::default();
    for (vertices, blue) in [([0, 1, 2, 3, 4], false), ([0, 1, 5, 6, 7], true)] {
        for a in 0..5 {
            for b in a + 1..5 {
                if a == 0 && b == 1 {
                    continue;
                }
                let id = edge_id(vertices[a], vertices[b]).expect("arista de test") as i32;
                contradiction.assign(if blue { -id } else { id })?;
            }
        }
    }
    for prefix_bits in [0, 1] {
        for epochs in [0, 32] {
            let report = device.solve(
                &contradiction,
                SearchConfig {
                    prefix_bits,
                    steps: 1,
                    epochs,
                    batch_warps: 1,
                    order_seed: 0,
                    lookahead: false,
                    explicit_trail: false,
                    neuro_mock: false,
                    neuro_trt: false,
                    policy_interval: 32,
                },
            )?;
            if report.outcome != Outcome::Unsat || report.completed != u64::from(report.universes) {
                return Err(Error::Device(format!(
                    "self-test DFS/reanudación prefix={prefix_bits}, epochs={epochs}: {report:?}"
                )));
            }
        }
    }
    let unfinished = device.solve(
        &Graph43::default(),
        SearchConfig {
            prefix_bits: 0,
            steps: 1,
            epochs: 1,
            batch_warps: 1,
            order_seed: 0,
            lookahead: false,
            explicit_trail: false,
            neuro_mock: false,
            neuro_trt: false,
            policy_interval: 32,
        },
    )?;
    if unfinished.outcome != Outcome::Unknown {
        return Err(Error::Device(
            "un presupuesto corto produjo un veredicto no autorizado".into(),
        ));
    }
    let impossible = device.solve(
        &full_red,
        SearchConfig {
            prefix_bits: 0,
            steps: 1,
            epochs: 2,
            batch_warps: 1,
            order_seed: 0,
            lookahead: false,
            explicit_trail: false,
            neuro_mock: false,
            neuro_trt: false,
            policy_interval: 32,
        },
    )?;
    if impossible.outcome != Outcome::Unsat {
        return Err(Error::Device("K43 rojo no fue refutado en CUDA".into()));
    }
    let mut one_bad_prefix = Graph43::default();
    for a in 0..5 {
        for b in a + 1..5 {
            if a != 0 || b != 1 {
                one_bad_prefix.assign(edge_id(a, b).expect("arista de test") as i32)?;
            }
        }
    }
    let isolated = device.solve(
        &one_bad_prefix,
        SearchConfig {
            prefix_bits: 1,
            steps: 1,
            epochs: 1,
            batch_warps: 2,
            order_seed: 0,
            lookahead: false,
            explicit_trail: false,
            neuro_mock: false,
            neuro_trt: false,
            policy_interval: 32,
        },
    )?;
    if isolated.outcome != Outcome::Unknown || isolated.completed != 1 || isolated.prunes != 1 {
        return Err(Error::Device(format!(
            "self-test aislamiento entre warps: {isolated:?}"
        )));
    }
    let mix = |mut value: u32| {
        value ^= value >> 16;
        value = value.wrapping_mul(0x7feb_352d);
        value ^= value >> 15;
        value = value.wrapping_mul(0x846c_a68b);
        value ^ (value >> 16)
    };
    // (32,42): ambas filas están en la banca alta. (1,33): el MISMO lane
    // posee ambos extremos, en bancas distintas; ninguna actualización puede perderse.
    for (edge, red_extra, blue_extra) in [
        ((32, 42), [0, 1, 2], [3, 4, 5]),
        ((1, 33), [2, 3, 4], [5, 6, 7]),
    ] {
        let edge_index = edge_id(edge.0, edge.1).expect("arista de test") as u32 - 1;
        let order_seed = (1..=1_000_000_u32)
            .find(|&seed| mix(seed) % EDGES as u32 == edge_index)
            .ok_or_else(|| {
                Error::Device("no se encontró semilla para la primera arista del test".into())
            })?;
        let mut high_contradiction = Graph43::default();
        for (extra, blue) in [(red_extra, false), (blue_extra, true)] {
            let vertices = [edge.0, edge.1, extra[0], extra[1], extra[2]];
            for a in 0..5 {
                for b in a + 1..5 {
                    if a == 0 && b == 1 {
                        continue;
                    }
                    let id = edge_id(vertices[a].min(vertices[b]), vertices[a].max(vertices[b]))
                        .expect("arista de test") as i32;
                    high_contradiction.assign(if blue { -id } else { id })?;
                }
            }
        }
        if check_reference(&high_contradiction) != CheckResult::Open
            || device.check(std::slice::from_ref(&high_contradiction))? != [CheckResult::Open]
        {
            return Err(Error::Device(format!(
                "base del test de arista {edge:?} ya está refutada"
            )));
        }
        for prefix_bits in [0, 1] {
            let report = device.solve(
                &high_contradiction,
                SearchConfig {
                    prefix_bits,
                    order_seed,
                    steps: 1,
                    epochs: if prefix_bits == 0 { 8 } else { 1 },
                    batch_warps: 2,
                    lookahead: false,
                    explicit_trail: false,
                    neuro_mock: false,
                    neuro_trt: false,
                    policy_interval: 32,
                },
            )?;
            if report.outcome != Outcome::Unsat
                || report.prunes != 2
                || report.completed != u64::from(report.universes)
            {
                return Err(Error::Device(format!("self-test arista {edge:?}, prefix={prefix_bits}, seed={order_seed}: {report:?}")));
            }
        }
    }
    // Misma continuación DFS con distinto particionado del presupuesto.
    // Un solo universo elimina carreras entre posibles modelos ganadores.
    let common = SearchConfig {
        prefix_bits: 0,
        order_seed: 43,
        steps: 100,
        epochs: 2,
        batch_warps: 1,
        lookahead: false,
        explicit_trail: false,
        neuro_mock: false,
        neuro_trt: false,
        policy_interval: 32,
    };
    let burst = device.solve(&Graph43::default(), common)?;
    let tiny = device.solve(
        &Graph43::default(),
        SearchConfig {
            steps: 4,
            epochs: 50,
            ..common
        },
    )?;
    if burst.transitions != 200
        || burst.transitions != tiny.transitions
        || burst.prunes != tiny.prunes
        || burst.completed != tiny.completed
        || burst.outcome != tiny.outcome
        || burst.max_batch_warps != 1
        || tiny.max_batch_warps != 1
    {
        return Err(Error::Device(format!(
            "presupuesto fragmentado cambió el DFS: {burst:?} frente a {tiny:?}"
        )));
    }
    let checks = graphs.len()
        + 15
        + self_test_lookahead(device)?
        + device.device.validate_explicit_trail()?;
    #[cfg(feature = "neuro-mock")]
    let checks = checks + device.device.validate_neuro()?;
    #[cfg(feature = "neuro-dataset")]
    let checks = checks + device.device.validate_dataset()?;
    Ok(checks)
}

fn self_test_lookahead(device: &NativeRamsey43) -> Result<usize, Error> {
    // Enumeración independiente de TODAS las coloraciones de nueve aristas:
    // sólo las dos monocromáticas deben forzar la arista ausente.
    let mut graphs = Vec::new();
    let mut edges = Vec::new();
    let mut expected = Vec::new();
    for vertices in [[0, 1, 2, 3, 4], [32, 34, 38, 41, 42], [1, 33, 2, 3, 4]] {
        let edge = (vertices[0].min(vertices[1]), vertices[0].max(vertices[1]));
        for bits in 0_u32..512 {
            let mut graph = Graph43::default();
            let mut position = 0;
            for a in 0..5 {
                for b in a + 1..5 {
                    if a == 0 && b == 1 {
                        continue;
                    }
                    let id = edge_id(vertices[a].min(vertices[b]), vertices[a].max(vertices[b]))
                        .expect("arista de test") as i32;
                    graph.assign(if bits >> position & 1 != 0 { id } else { -id })?;
                    position += 1;
                }
            }
            graphs.push(graph);
            edges.push(edge);
            expected.push(if bits == 0 {
                1
            } else if bits == 511 {
                2
            } else {
                3
            });
        }
    }
    // K4 aislado no fuerza ninguna conexión particular al quinto vértice.
    let mut k4 = Graph43::default();
    for a in 1..5 {
        for b in a + 1..5 {
            k4.assign(edge_id(a, b).unwrap() as i32)?;
        }
    }
    graphs.push(k4);
    edges.push((0, 1));
    expected.push(3);
    for blue in [false, true] {
        let mut degree24 = Graph43::default();
        for b in 2..=25 {
            let id = edge_id(0, b).unwrap() as i32;
            degree24.assign(if blue { -id } else { id })?;
        }
        graphs.push(degree24);
        edges.push((0, 1));
        expected.push(if blue { 1 } else { 2 });
    }
    let actual = device.allowed_colours(&graphs, &edges)?;
    if let Some((index, (actual, expected))) = actual
        .iter()
        .zip(&expected)
        .enumerate()
        .find(|(_, (a, e))| a != e)
    {
        return Err(Error::Device(format!(
            "self-test lookahead {index}: {actual} != {expected}"
        )));
    }

    // Primera libre (0,1) forzada azul; siguiente (0,2) tiene ambos colores
    // refutados. Debe deshacer la fuerza sin intentar rojo, incluso al suspender.
    let mut rollback = Graph43::default();
    for (vertices, blue) in [
        ([0, 1, 3, 4, 5], false),
        ([0, 2, 8, 9, 10], false),
        ([0, 2, 11, 12, 13], true),
    ] {
        for a in 0..5 {
            for b in a + 1..5 {
                if a == 0 && b == 1 {
                    continue;
                }
                let id = edge_id(vertices[a], vertices[b]).unwrap() as i32;
                rollback.assign(if blue { -id } else { id })?;
            }
        }
    }
    let common = SearchConfig {
        prefix_bits: 0,
        order_seed: 0,
        steps: 1,
        epochs: 8,
        batch_warps: 1,
        lookahead: true,
        explicit_trail: false,
        neuro_mock: false,
        neuro_trt: false,
        policy_interval: 32,
    };
    let tiny = device.solve(&rollback, common)?;
    let burst = device.solve(
        &rollback,
        SearchConfig {
            steps: 8,
            epochs: 1,
            ..common
        },
    )?;
    for report in [&tiny, &burst] {
        if report.outcome != Outcome::Unsat
            || report.completed != 1
            || report.forced_assignments != 1
            || report.decisions != 0
            || report.lookahead_checks != 2
            || report.lookahead_conflicts != 1
            || report.transitions != 4
        {
            return Err(Error::Device(format!(
                "self-test trail forzado/rollback: {report:?}"
            )));
        }
    }
    let mut checks = graphs.len() + 2;
    // La primera libre bloquea ambos colores: un conflicto, sin decisión.
    let mut contradiction = Graph43::default();
    for (vertices, blue) in [([0, 1, 2, 3, 4], false), ([0, 1, 5, 6, 7], true)] {
        for a in 0..5 {
            for b in a + 1..5 {
                if a == 0 && b == 1 {
                    continue;
                }
                let id = edge_id(vertices[a], vertices[b]).unwrap() as i32;
                contradiction.assign(if blue { -id } else { id })?;
            }
        }
    }
    if device.allowed_colours(std::slice::from_ref(&contradiction), &[(0, 1)])? != [0] {
        return Err(Error::Device(
            "doble prohibición no detectada anticipadamente".into(),
        ));
    }
    checks += 1;
    for prefix_bits in [0, 1] {
        let report = device.solve(
            &contradiction,
            SearchConfig {
                prefix_bits,
                ..common
            },
        )?;
        if report.outcome != Outcome::Unsat
            || report.completed != u64::from(report.universes)
            || report.decisions != 0
            || report.prunes != u64::from(report.universes)
        {
            return Err(Error::Device(format!(
                "self-test conflicto temprano/prefijo: {report:?}"
            )));
        }
        checks += 1;
    }
    // Un solo prefijo invalido no debe eliminar al universo vecino.
    let mut nine_red = Graph43::default();
    for a in 0..5 {
        for b in a + 1..5 {
            if a != 0 || b != 1 {
                nine_red.assign(edge_id(a, b).unwrap() as i32)?;
            }
        }
    }
    let isolated = device.solve(
        &nine_red,
        SearchConfig {
            prefix_bits: 1,
            steps: 1,
            epochs: 1,
            batch_warps: 2,
            ..common
        },
    )?;
    if isolated.outcome != Outcome::Unknown || isolated.completed != 1 || isolated.prunes != 1 {
        return Err(Error::Device(format!(
            "self-test aislamiento de prefijos smart: {isolated:?}"
        )));
    }
    checks += 1;
    let mut degree25 = Graph43::default();
    for b in 1..=25 {
        degree25.assign(edge_id(0, b).unwrap() as i32)?;
    }
    if check_reference(&degree25) != CheckResult::Open {
        return Err(Error::Device(
            "test de grado 25 contiene un K5 inesperado".into(),
        ));
    }
    let pruned = device.solve(&degree25, common)?;
    if pruned.outcome != Outcome::Unsat
        || pruned.lookahead_conflicts != 1
        || pruned.transitions != 0
    {
        return Err(Error::Device(format!(
            "self-test limite necesario de grado: {pruned:?}"
        )));
    }
    checks += 1;
    // Igual presupuesto, incluyendo inferencias posteriores al checkpoint.
    let burst = device.solve(
        &canonical_star(18)?,
        SearchConfig {
            order_seed: 43,
            steps: 100,
            epochs: 2,
            ..common
        },
    )?;
    let tiny = device.solve(
        &canonical_star(18)?,
        SearchConfig {
            order_seed: 43,
            steps: 4,
            epochs: 50,
            ..common
        },
    )?;
    if burst.transitions != 200
        || burst.transitions != tiny.transitions
        || burst.prunes != tiny.prunes
        || burst.decisions != tiny.decisions
        || burst.forced_assignments != tiny.forced_assignments
        || burst.lookahead_checks != tiny.lookahead_checks
        || burst.lookahead_conflicts != tiny.lookahead_conflicts
        || burst.outcome != tiny.outcome
        || burst.completed != tiny.completed
    {
        return Err(Error::Device(format!(
            "self-test checkpoint smart: {burst:?} vs {tiny:?}"
        )));
    }
    Ok(checks + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_stars_have_exactly_the_required_fixed_row() {
        for degree in 18..=21 {
            let graph = canonical_star(degree).unwrap();
            graph.validate().unwrap();
            assert_eq!(graph.rows[0].count_ones(), degree);
            assert_eq!(graph.rows[VERTICES].count_ones(), 42 - degree);
            assert_eq!(graph.free_edges(), 861);
            for b in 1..VERTICES {
                assert_eq!(graph.rows[0] >> b & 1 != 0, b <= degree as usize);
            }
            let file = std::fs::read_to_string(format!("results/k43_star_{degree}.sat")).unwrap();
            assert_eq!(parse_partial_dimacs(&file).unwrap(), graph);
        }
        for degree in [0, 17, 22, 42, u32::MAX] {
            assert!(canonical_star(degree).is_err());
        }
    }
    #[test]
    fn lexical_ids_are_bijective_including_high_vertices() {
        let mut next = 1;
        for a in 0..VERTICES {
            for b in a + 1..VERTICES {
                assert_eq!(edge_id(a, b), Some(next));
                assert_eq!(edge_vertices(next), Some((a, b)));
                next += 1;
            }
        }
        assert_eq!(next, 904);
        assert_eq!(edge_vertices(0), None);
        assert_eq!(edge_vertices(904), None);
        assert_eq!(edge_id(42, 42), None);
        assert_eq!(edge_id(0, 43), None);
    }
    #[test]
    fn partial_dimacs_rejects_auxiliaries_and_malformed_terminators() {
        let graph = parse_partial_dimacs("c profundidad 2\nv 1 -903\nv 0\n").unwrap();
        assert_eq!(graph.free_edges(), EDGES - 2);
        for invalid in [
            "v 0 1",
            "v 1 0\nv 2 0",
            "v 1 -1 0",
            "v 904 0",
            "v -2583 0",
            "v -2147483648 0",
            "v nan 0",
            "v 1",
            "p cnf 903 1",
            "c no modelo",
        ] {
            assert!(parse_partial_dimacs(invalid).is_err(), "{invalid}");
        }
        assert_eq!(parse_partial_dimacs("v 0").unwrap(), Graph43::default());
    }
    #[test]
    fn row_contract_rejects_corruption() {
        for bad in [0, 1, 2, 3] {
            let mut rows = [0_u64; ROW_WORDS];
            match bad {
                0 => rows[0] = 1,
                1 => rows[0] = 1_u64 << 43,
                2 => rows[0] = 2,
                _ => {
                    rows[0] = 2;
                    rows[1] = 1;
                    rows[43] = 2;
                    rows[44] = 1;
                }
            }
            assert!(Graph43::from_rows(rows).is_err());
        }
    }
    #[test]
    fn unknown_edges_do_not_become_blue_and_high_vertices_are_checked() {
        assert_eq!(check_reference(&Graph43::default()), CheckResult::Open);
        let vertices = [32, 34, 38, 41, 42];
        for blue in [false, true] {
            let mut graph = Graph43::default();
            for a in 0..5 {
                for b in a + 1..5 {
                    let id = edge_id(vertices[a], vertices[b]).unwrap() as i32;
                    graph.assign(if blue { -id } else { id }).unwrap();
                }
            }
            assert_eq!(check_reference(&graph), CheckResult::Bad { blue, vertices });
            assert_eq!(graph.free_edges(), EDGES - 10);
        }
    }
    #[test]
    fn exhaustive_five_vertex_colourings_only_refute_the_two_monochromatic_models() {
        for bits in 0_u32..1 << 10 {
            let mut graph = Graph43::default();
            let mut edge = 0;
            for a in 0..5 {
                for b in a + 1..5 {
                    let id = edge_id(a + 38, b + 38).unwrap() as i32;
                    graph
                        .assign(if bits >> edge & 1 != 0 { id } else { -id })
                        .unwrap();
                    edge += 1;
                }
            }
            match check_reference(&graph) {
                CheckResult::Bad { blue, vertices } => {
                    assert!(bits == 0 || bits == 1023);
                    assert_eq!(blue, bits == 0);
                    assert_eq!(vertices, [38, 39, 40, 41, 42]);
                }
                CheckResult::Open => assert!(bits != 0 && bits != 1023),
                CheckResult::CompleteSafe => {
                    panic!("893 aristas desconocidas no constituyen un modelo completo")
                }
            }
        }
    }
    #[test]
    fn partial_models_are_not_publishable() {
        assert!(Graph43::default().model().is_err());
        let mut graph = Graph43::default();
        for literal in 1..=EDGES as i32 {
            graph
                .assign(if literal % 2 == 0 { -literal } else { literal })
                .unwrap();
        }
        let model = graph.model().unwrap();
        assert_eq!(model.len(), 903);
        assert_eq!(
            parse_partial_dimacs(&format!(
                "v {} 0",
                model
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" ")
            ))
            .unwrap(),
            graph
        );
    }
    #[test]
    fn prefix_partition_and_launch_budgets_are_validated() {
        let empty = Graph43::default();
        assert_eq!(SearchConfig::default().validate(&empty).unwrap(), 16);
        let continuous = SearchConfig {
            epochs: 0,
            ..SearchConfig::default()
        };
        assert!(continuous.validate(&empty).is_ok());
        assert!(continuous.has_next_epoch(u64::MAX));
        assert!(SearchConfig::default().has_next_epoch(15));
        assert!(!SearchConfig::default().has_next_epoch(16));
        for policy_interval in [0, 65_537] {
            assert!(SearchConfig {
                policy_interval,
                ..SearchConfig::default()
            }
            .validate(&empty)
            .is_err());
        }
        for steps in [100, 500, 65_536] {
            assert!(SearchConfig {
                steps,
                ..SearchConfig::default()
            }
            .validate(&empty)
            .is_ok());
        }
        assert!(SearchConfig {
            steps: 65_537,
            ..SearchConfig::default()
        }
        .validate(&empty)
        .is_err());
        assert!(SearchConfig {
            prefix_bits: 21,
            ..SearchConfig::default()
        }
        .validate(&empty)
        .is_err());
        assert!(SearchConfig {
            steps: 0,
            ..SearchConfig::default()
        }
        .validate(&empty)
        .is_err());
        assert!(SearchConfig {
            batch_warps: 513,
            ..SearchConfig::default()
        }
        .validate(&empty)
        .is_err());
        let mut full = Graph43::default();
        for id in 1..=EDGES as i32 {
            full.assign(id).unwrap();
        }
        assert!(SearchConfig::default().validate(&full).is_err());
        assert_eq!(
            SearchConfig {
                prefix_bits: 0,
                ..SearchConfig::default()
            }
            .validate(&full)
            .unwrap(),
            1
        );
    }
    #[cfg(not(feature = "gpu"))]
    #[test]
    fn no_cuda_feature_means_no_runtime_driver_loading() {
        assert!(matches!(NativeRamsey43::new(0), Err(Error::NotCompiled)));
    }
    #[cfg(feature = "neuro-mock")]
    #[test]
    fn feature_half_reference_decodes_ieee_boundaries() {
        assert_eq!(decode_feature_half(0), 0.0);
        assert_eq!(decode_feature_half(0x3c00), 1.0);
        assert_eq!(decode_feature_half(0x3800), 0.5);
        assert_eq!(decode_feature_half(1), 2.0_f32.powi(-24));
        assert_eq!(decode_feature_half(0x0400), 2.0_f32.powi(-14));
        assert!(decode_feature_half(0x7c00).is_infinite());
        assert!(decode_feature_half(0x7e00).is_nan());
        let features = feature_reference(&Graph43::default(), 0, 0);
        assert_eq!(features[2], 1.0);
        assert_eq!(features[12], 1.0);
        assert_eq!(features[13], 1.0);
        assert_eq!(features[16], 41.0 / 42.0);
        assert!(features[20..].iter().all(|&value| value == 0.0));
    }
}
