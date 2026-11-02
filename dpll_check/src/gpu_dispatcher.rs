// Mejora 3: puente con la GPU para la fase "conquer" de cube and conquer.
//
// La CPU parte el problema en cubos y este modulo los recibe por lotes. La tarjeta hace de
// prefiltro: un nucleo CUDA (bcp_filter.cu) aplica propagacion unitaria a cada cubo, un
// hilo por cubo. Si llega a una contradiccion el cubo es `Unsat`; si deja todas las
// clausulas satisfechas es `Sat`, con el modelo comprobado en la CPU; lo demas vuelve como
// `Unknown` y es trabajo del CDCL. En la GPU no hay busqueda ni aprendizaje.
//
// Todo lo que toca CUDA esta tras la caracteristica `gpu` de Cargo. Sin ella el modulo
// compila igual, `GpuConqueror::new` devuelve `GpuError::NotCompiled` y el binario no
// enlaza nada de NVIDIA. Este modulo no conoce el motor ni el motor lo conoce a el.
//
// El unico `unsafe` es el lanzamiento del nucleo, en `backend::Device::launch`: el crate
// lo prohibe en todo lo demas.
use std::fmt;

pub type Clause = Vec<i32>;

// Veredicto de un cubo. `Unknown` es una respuesta legitima: lo que la GPU no decide
// vuelve a la CPU, y la GPU nunca afirma nada que no haya comprobado.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SolveResult {
    // Modelo de la formula que cumple el cubo, en literales DIMACS.
    Sat(Vec<i32>),
    Unsat,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuError {
    // El binario se compilo sin la caracteristica `gpu`.
    NotCompiled,
    // No hay driver, no hay tarjeta o no se deja abrir.
    Unavailable(String),
    // El plan de buffers no cabe en la VRAM libre.
    OutOfMemory { needed: u64, free: u64 },
    // Lo que se quiere subir no cabe en lo reservado.
    Capacity { what: &'static str, needed: usize, reserved: usize },
    // Un literal 0 o fuera de rango.
    Malformed(String),
    Driver(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCompiled => write!(out, "este binario se compilo sin soporte de GPU: recompila con `cargo build --release --features gpu`"),
            Self::Unavailable(reason) => write!(out, "no hay GPU CUDA disponible: {reason}"),
            Self::OutOfMemory { needed, free } => write!(out, "los buffers piden {needed} bytes de VRAM y hay {free} libres"),
            Self::Capacity { what, needed, reserved } => write!(out, "{what}: hacen falta {needed} y hay {reserved} reservados"),
            Self::Malformed(reason) => write!(out, "entrada no valida: {reason}"),
            Self::Driver(reason) => write!(out, "error del driver CUDA: {reason}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuInfo {
    pub ordinal: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_bytes: u64,
    // VRAM libre al abrir el dispositivo, antes de reservar nada.
    pub free_bytes: u64,
}

// Lista de clausulas (o de cubos) aplanada como la leera el nucleo: los literales de la
// lista i son literals[starts[i]..starts[i + 1]]. Un literal es 2 * (variable - 1), mas 1
// si es negativo: el mismo bit de signo que usa el CDCL.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Flat {
    pub starts: Vec<u32>,
    pub literals: Vec<u32>,
    // Mayor identificador de variable que aparece.
    pub variables: u32,
}

fn encode(literal: i32) -> Result<u32, GpuError> {
    if literal == 0 || literal == i32::MIN {
        return Err(GpuError::Malformed(format!("literal {literal}")));
    }
    Ok((literal.unsigned_abs() - 1) << 1 | u32::from(literal < 0))
}

pub fn flatten(lists: &[Vec<i32>]) -> Result<Flat, GpuError> {
    let total: usize = lists.iter().map(Vec::len).sum();
    if u32::try_from(total).is_err() || u32::try_from(lists.len()).is_err() {
        return Err(GpuError::Malformed(format!("{total} literales no caben en indices de 32 bits")));
    }
    let mut flat = Flat { starts: Vec::with_capacity(lists.len() + 1), literals: Vec::with_capacity(total), variables: 0 };
    flat.starts.push(0);
    for list in lists {
        for &literal in list {
            flat.literals.push(encode(literal)?);
            flat.variables = flat.variables.max(literal.unsigned_abs());
        }
        flat.starts.push(flat.literals.len() as u32);
    }
    Ok(flat)
}

// `lists` aplanada seria exactamente `flat`. Recorre sin reservar memoria: es como se
// decide, lote a lote, que la formula ya esta en VRAM.
fn matches(flat: &Flat, lists: &[Vec<i32>]) -> bool {
    flat.starts.len() == lists.len() + 1
        && lists.iter().zip(flat.starts.windows(2)).all(|(list, bounds)| {
            let stored = &flat.literals[bounds[0] as usize..bounds[1] as usize];
            stored.len() == list.len() && list.iter().zip(stored).all(|(&literal, &code)| encode(literal) == Ok(code))
        })
}

// Un modelo denso (la variable i en la posicion i - 1) cumple el cubo y todas las clausulas.
fn satisfies(formula: &[Clause], cube: &[i32], model: &[i32]) -> bool {
    let holds = |literal: &i32| model.get(literal.unsigned_abs() as usize - 1) == Some(literal);
    cube.iter().all(holds) && formula.iter().all(|clause| clause.iter().any(|literal| holds(literal)))
}

// Lo que hace el nucleo, en la CPU y con sus mismas reglas: es su especificacion y el
// testigo de los tests. La propagacion unitaria llega al mismo punto fijo en cualquier
// orden, asi que los veredictos tienen que coincidir uno a uno. `variables` es el ancho
// del modelo; lo que queda sin asignar va a falso.
pub fn bcp_reference(formula: &[Clause], cube: &[i32], variables: usize) -> SolveResult {
    // 0 sin asignar, 1 cierta, -1 falsa.
    let mut values = vec![0_i8; variables + 1];
    let value = |values: &[i8], literal: i32| values[literal.unsigned_abs() as usize] * literal.signum() as i8;
    for &literal in cube {
        match value(&values, literal) {
            -1 => return SolveResult::Unsat,
            0 => values[literal.unsigned_abs() as usize] = literal.signum() as i8,
            _ => {}
        }
    }
    loop {
        let (mut changed, mut open) = (false, false);
        for clause in formula {
            // Primer literal libre, y si antes de uno cierto aparece otro distinto.
            let (mut free, mut second, mut satisfied) = (None, false, false);
            for &literal in clause {
                match value(&values, literal) {
                    1 => {
                        satisfied = true;
                        break;
                    }
                    0 if free.is_none() => free = Some(literal),
                    0 if free != Some(literal) => {
                        second = true;
                        break;
                    }
                    _ => {}
                }
            }
            match free {
                _ if satisfied => {}
                None => return SolveResult::Unsat,
                Some(_) if second => open = true,
                Some(literal) => {
                    values[literal.unsigned_abs() as usize] = literal.signum() as i8;
                    changed = true;
                }
            }
        }
        if !changed {
            if open {
                return SolveResult::Unknown;
            }
            return SolveResult::Sat((1..=variables as i32).map(|id| if values[id as usize] == 1 { id } else { -id }).collect());
        }
    }
}

// Lo que se reserva en VRAM al abrir el dispositivo, en elementos. Un lote que no quepa
// se rechaza entero: quien llama lo trocea.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferPlan {
    pub variables: usize,
    pub clauses: usize,
    pub literals: usize,
    // Cubos por lote, y literales de cubo por lote.
    pub cubes: usize,
    pub cube_literals: usize,
}

// Buffers de VRAM, todos de palabras de 32 bits, en el orden en que se reservan.
const LITERALS: usize = 0;
const CLAUSE_STARTS: usize = 1;
const CUBE_LITERALS: usize = 2;
const CUBE_STARTS: usize = 3;
const ASSIGNMENT: usize = 4;
const VERDICTS: usize = 5;
const BUFFERS: usize = 6;

// Veredictos del nucleo, como los define bcp_filter.cu. PENDING no sale de este modulo:
// es un cubo cuyo lanzamiento agoto sus pasadas y hay que relanzar.
const UNKNOWN: u32 = 0;
const UNSAT: u32 = 1;
const SAT: u32 = 2;
const PENDING: u32 = 3;
// Trabajo de un hilo en un lanzamiento, en literales leidos como mucho: de aqui sale
// cuantas pasadas densas caben antes de devolver el control.
const ROUND_WORK: usize = 1 << 24;

impl BufferPlan {
    // Plan para una formula concreta y lotes de `cubes` cubos de hasta `depth` literales.
    pub fn for_formula(formula: &[Clause], cubes: usize, depth: usize) -> Self {
        Self {
            variables: formula.iter().flatten().map(|literal| literal.unsigned_abs()).max().unwrap_or(0) as usize,
            clauses: formula.len(),
            literals: formula.iter().map(Vec::len).sum(),
            cubes,
            cube_literals: cubes.saturating_mul(depth),
        }
    }
    // Estado de un cubo en el nucleo: dos bits por variable (sin asignar, cierta, falsa).
    pub fn words_per_cube(&self) -> usize {
        self.variables.div_ceil(16).max(1)
    }
    // Palabras de cada buffer: literales e inicios de la formula, literales e inicios del
    // lote, asignaciones de trabajo y veredictos. Ninguno queda vacio.
    fn sizes(&self) -> [usize; BUFFERS] {
        [
            self.literals.max(1), self.clauses.saturating_add(1),
            self.cube_literals.max(1), self.cubes.saturating_add(1),
            self.cubes.saturating_mul(self.words_per_cube()).max(1), self.cubes.max(1),
        ]
    }
    pub fn bytes(&self) -> u64 {
        self.sizes().iter().fold(0_u64, |total, &words| total.saturating_add((words as u64).saturating_mul(4)))
    }
}

#[cfg(feature = "gpu")]
mod backend {
    use super::{BufferPlan, GpuError, GpuInfo, PENDING, VERDICTS};
    use cudarc::driver::sys::CUdevice_attribute;
    use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    // El nucleo viaja dentro del binario y se compila con nvrtc al abrir la tarjeta.
    const KERNEL: &str = include_str!("bcp_filter.cu");
    const THREADS_PER_BLOCK: u32 = 256;
    // Bloques por multiprocesador de la tarjeta en cada lanzamiento. Medido en una RTX 4060
    // con R(5,5) sobre K_36: con dos se alcanza el ritmo maximo (32 000 cubos por segundo de
    // una pasada); con mas, los lanzamientos se alargan y el ritmo no sube.
    const BLOCKS_PER_PROCESSOR: u32 = 2;
    // Windows reinicia el driver de una tarjeta que pasa unos dos segundos sin responder.
    // Un lanzamiento que supere este limite hace que los siguientes lleven menos cubos.
    const LAUNCH_LIMIT: Duration = Duration::from_millis(1000);

    pub struct Device {
        // El contexto tiene que vivir lo que vivan los buffers.
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        kernel: CudaFunction,
        buffers: Vec<CudaSlice<u32>>,
        // Cubos, uno por hilo, que lleva cada lanzamiento, y lo minimo a lo que se baja:
        // un bloque por multiprocesador. Por debajo de eso un lanzamiento ya no se acorta,
        // porque lo que dura es la pasada de un hilo.
        pub(super) chunk: u32,
        floor: u32,
    }

    fn driver(error: DriverError) -> GpuError {
        GpuError::Driver(error.to_string())
    }

    impl Device {
        pub fn open(ordinal: usize, plan: &BufferPlan) -> Result<(Self, GpuInfo), GpuError> {
            // Con carga dinamica, cudarc entra en panico si no encuentra la biblioteca del
            // driver: aqui eso es "no hay GPU", no un fallo del programa.
            let context = std::panic::catch_unwind(|| CudaContext::new(ordinal))
                .map_err(|_| GpuError::Unavailable("no se encuentra el driver de NVIDIA".into()))?
                .map_err(|error| GpuError::Unavailable(error.to_string()))?;
            let (free, total) = context.mem_get_info().map_err(driver)?;
            let info = GpuInfo {
                ordinal,
                name: context.name().map_err(driver)?,
                compute_capability: context.compute_capability().map_err(driver)?,
                total_bytes: total as u64,
                free_bytes: free as u64,
            };
            let needed = plan.bytes();
            if needed > info.free_bytes {
                return Err(GpuError::OutOfMemory { needed, free: info.free_bytes });
            }
            // Igual que con el driver: sin la biblioteca de nvrtc no hay con que compilar.
            let ptx = std::panic::catch_unwind(|| cudarc::nvrtc::compile_ptx(KERNEL))
                .map_err(|_| GpuError::Unavailable("no se encuentra nvrtc (CUDA Toolkit) para compilar el nucleo".into()))?
                .map_err(|error| GpuError::Driver(format!("el nucleo no compila: {error:?}")))?;
            let kernel = context.load_module(ptx).map_err(driver)?.load_function("bcp_filter").map_err(driver)?;
            let stream = context.default_stream();
            let buffers = plan.sizes().iter()
                .map(|&words| stream.alloc_zeros::<u32>(words).map_err(driver))
                .collect::<Result<Vec<_>, _>>()?;
            stream.synchronize().map_err(driver)?;
            // Un hilo tarda lo mismo en su pasada vaya solo o acompanado, mientras haya
            // multiprocesadores libres: se llena la tarjeta en cada lanzamiento.
            let processors = context.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).map_err(driver)?;
            let floor = (processors.max(1) as u32).saturating_mul(THREADS_PER_BLOCK);
            let chunk = floor.saturating_mul(BLOCKS_PER_PROCESSOR);
            Ok((Self { context, stream, kernel, buffers, chunk, floor }, info))
        }
        // Copia `data` al principio del buffer; que quepa lo comprueba quien llama.
        pub fn upload(&mut self, buffer: usize, data: &[u32]) -> Result<(), GpuError> {
            if data.is_empty() {
                return Ok(());
            }
            let mut view = self.buffers[buffer].slice_mut(..data.len());
            self.stream.memcpy_htod(data, &mut view).map_err(driver)
        }
        pub fn download(&self, buffer: usize, start: usize, words: usize) -> Result<Vec<u32>, GpuError> {
            if words == 0 {
                return Ok(Vec::new());
            }
            self.stream.clone_dtoh(&self.buffers[buffer].slice(start..start + words)).map_err(driver)
        }
        pub fn free_bytes(&self) -> Result<u64, GpuError> {
            self.context.mem_get_info().map(|(free, _)| free as u64).map_err(driver)
        }
        // Lanza el nucleo sobre `count` cubos del lote a partir de `first` y espera a que
        // termine. `phase` son las pasadas que esos cubos llevan ya: con cero cada hilo
        // carga su cubo; con mas, sigue donde se quedo.
        #[allow(unsafe_code)]
        fn launch(&mut self, clauses: u32, words: u32, rounds: u32, first: u32, count: u32, phase: u32) -> Result<(), GpuError> {
            let fresh = u32::from(phase == 0);
            let [literals, clause_starts, cube_literals, cube_starts, assignment, verdicts] = &mut self.buffers[..] else {
                unreachable!("Los buffers son los seis del plan");
            };
            // En el orden de los parametros de bcp_filter.
            let mut call = self.stream.launch_builder(&self.kernel);
            call.arg(&*clause_starts).arg(&*literals).arg(&clauses)
                .arg(&*cube_starts).arg(&*cube_literals)
                .arg(assignment).arg(&words).arg(verdicts)
                .arg(&first).arg(&count).arg(&fresh).arg(&rounds).arg(&phase);
            let config = LaunchConfig {
                grid_dim: (count.div_ceil(THREADS_PER_BLOCK), 1, 1), block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: los trece argumentos son, en orden y tipo, los parametros de bcp_filter
            // (cinco punteros a palabras de 32 bits y ocho enteros de 32 bits). El nucleo solo
            // indexa dentro de lo reservado porque `GpuConqueror` lo ha comprobado antes de
            // llegar aqui: la formula y el lote subidos caben en el plan, sus inicios los
            // genera `flatten` (crecientes y dentro de los literales subidos), toda variable
            // es menor que 16 * words y first + count no pasa de los cubos del lote. Cada
            // hilo escribe solo en la fila de estado y en el veredicto de su cubo. Los
            // buffers viven en `self` y el lanzamiento se espera antes de volver.
            unsafe { call.launch(config) }.map_err(driver)?;
            self.stream.synchronize().map_err(driver)
        }
        // Veredictos de los `cubes` primeros cubos del lote subido. El lote va en trozos de
        // `chunk` cubos, y lo que no llega al punto fijo en las pasadas de un lanzamiento
        // se relanza hasta que llega.
        pub fn filter(&mut self, clauses: u32, words: u32, rounds: u32, cubes: u32) -> Result<Vec<u32>, GpuError> {
            let mut first = 0;
            while first < cubes {
                let count = self.chunk.min(cubes - first);
                let mut phase = 0;
                loop {
                    let start = Instant::now();
                    self.launch(clauses, words, rounds, first, count, phase)?;
                    // Las pasadas por lanzamiento ya son las minimas: lo unico que acorta
                    // uno largo es que la tarjeta lleve menos hilos a la vez.
                    if start.elapsed() > LAUNCH_LIMIT {
                        self.chunk = (self.chunk / 2).max(self.floor);
                    }
                    phase += rounds;
                    if !self.download(VERDICTS, first as usize, count as usize)?.contains(&PENDING) {
                        break;
                    }
                }
                first += count;
            }
            self.download(VERDICTS, 0, cubes as usize)
        }
    }
}

#[cfg(not(feature = "gpu"))]
mod backend {
    use super::{BufferPlan, GpuError, GpuInfo};

    // Sin la caracteristica `gpu` no hay dispositivo que abrir: el tipo no tiene valores.
    pub enum Device {}

    impl Device {
        pub fn open(_ordinal: usize, _plan: &BufferPlan) -> Result<(Self, GpuInfo), GpuError> {
            Err(GpuError::NotCompiled)
        }
        pub fn upload(&mut self, _buffer: usize, _data: &[u32]) -> Result<(), GpuError> {
            match *self {}
        }
        pub fn download(&self, _buffer: usize, _start: usize, _words: usize) -> Result<Vec<u32>, GpuError> {
            match *self {}
        }
        pub fn free_bytes(&self) -> Result<u64, GpuError> {
            match *self {}
        }
        pub fn filter(&mut self, _clauses: u32, _words: u32, _rounds: u32, _cubes: u32) -> Result<Vec<u32>, GpuError> {
            match *self {}
        }
    }
}

// Controlador de la tarjeta: el dispositivo abierto, sus buffers y lo que hay subido.
pub struct GpuConqueror {
    device: backend::Device,
    info: GpuInfo,
    plan: BufferPlan,
    // Copia en el anfitrion de la formula que hay en VRAM.
    loaded: Option<Flat>,
    // Por que el ultimo lote volvio entero sin decidir, si fue un fallo del puente.
    last_error: Option<GpuError>,
}

impl GpuConqueror {
    // Abre la primera tarjeta CUDA, compila el nucleo y reserva en VRAM los buffers del
    // plan, a cero.
    pub fn new(plan: BufferPlan) -> Result<Self, GpuError> {
        let (device, info) = backend::Device::open(0, &plan)?;
        Ok(Self { device, info, plan, loaded: None, last_error: None })
    }
    pub fn info(&self) -> &GpuInfo {
        &self.info
    }
    pub fn plan(&self) -> &BufferPlan {
        &self.plan
    }
    pub fn last_error(&self) -> Option<&GpuError> {
        self.last_error.as_ref()
    }
    // VRAM libre ahora mismo, con los buffers ya reservados.
    pub fn free_bytes(&self) -> Result<u64, GpuError> {
        self.device.free_bytes()
    }
    fn fits(what: &'static str, needed: usize, reserved: usize) -> Result<(), GpuError> {
        if needed <= reserved { Ok(()) } else { Err(GpuError::Capacity { what, needed, reserved }) }
    }
    // Deja la formula en VRAM. Si ya es la que hay, no se vuelve a subir.
    pub fn load_formula(&mut self, formula: &[Clause]) -> Result<(), GpuError> {
        if self.loaded.as_ref().is_some_and(|flat| matches(flat, formula)) {
            return Ok(());
        }
        let flat = flatten(formula)?;
        Self::fits("variables", flat.variables as usize, self.plan.variables)?;
        Self::fits("clausulas", formula.len(), self.plan.clauses)?;
        Self::fits("literales", flat.literals.len(), self.plan.literals)?;
        // Si la subida se queda a medias, lo que hay en VRAM no es ninguna formula.
        self.loaded = None;
        self.device.upload(LITERALS, &flat.literals)?;
        self.device.upload(CLAUSE_STARTS, &flat.starts)?;
        self.loaded = Some(flat);
        Ok(())
    }
    // Relee de VRAM la formula subida y la compara con la copia del anfitrion.
    pub fn verify_formula(&self) -> Result<bool, GpuError> {
        let Some(flat) = &self.loaded else { return Ok(false) };
        Ok(self.device.download(LITERALS, 0, flat.literals.len())? == flat.literals
            && self.device.download(CLAUSE_STARTS, 0, flat.starts.len())? == flat.starts)
    }
    fn stage(&mut self, formula: &[Clause], cubes: &[Vec<i32>]) -> Result<(), GpuError> {
        self.load_formula(formula)?;
        let batch = flatten(cubes)?;
        Self::fits("cubos del lote", cubes.len(), self.plan.cubes)?;
        Self::fits("literales de cubo", batch.literals.len(), self.plan.cube_literals)?;
        Self::fits("variables de los cubos", batch.variables as usize, self.plan.variables)?;
        self.device.upload(CUBE_LITERALS, &batch.literals)?;
        self.device.upload(CUBE_STARTS, &batch.starts)
    }
    fn conquer(&mut self, formula: &[Clause], cubes: &[Vec<i32>]) -> Result<Vec<SolveResult>, GpuError> {
        self.stage(formula, cubes)?;
        let (clauses, literals) = self.loaded.as_ref().map_or((0, 0), |flat| (flat.starts.len() - 1, flat.literals.len()));
        let words = self.plan.words_per_cube();
        let rounds = (ROUND_WORK / literals.max(1)).clamp(1, 64);
        let verdicts = self.device.filter(clauses as u32, words as u32, rounds as u32, cubes.len() as u32)?;
        verdicts.iter().zip(cubes).enumerate().map(|(index, (&verdict, cube))| match verdict {
            UNKNOWN => Ok(SolveResult::Unknown),
            UNSAT => Ok(SolveResult::Unsat),
            SAT => {
                let state = self.device.download(ASSIGNMENT, index * words, words)?;
                let model: Vec<i32> = (0..self.plan.variables).map(|variable| {
                    let id = variable as i32 + 1;
                    if state[variable >> 4] >> ((variable & 15) << 1) & 3 == 1 { id } else { -id }
                }).collect();
                // Un modelo solo sale de aqui si la CPU lo ha comprobado contra la formula.
                if satisfies(formula, cube, &model) {
                    Ok(SolveResult::Sat(model))
                } else {
                    Err(GpuError::Driver(format!("el modelo que la GPU da para el cubo {index} no cumple la formula")))
                }
            }
            other => Err(GpuError::Driver(format!("veredicto {other} sin sentido en el cubo {index}"))),
        }).collect()
    }
    // Filtra en la GPU un lote de cubos de `formula`: un veredicto por cubo, en el mismo
    // orden. Un cubo es una lista de literales DIMACS que se suponen ciertos.
    //
    // `Unsat`: la propagacion unitaria refuta el cubo. `Sat`: deja satisfechas todas las
    // clausulas, y el modelo que acompana cumple la formula y el cubo. `Unknown`: ni lo
    // uno ni lo otro; hay que buscar. Si el puente falla (el lote no cabe, error del
    // driver) todo el lote vuelve `Unknown` y `last_error` dice por que: la CPU se queda
    // con el.
    pub fn dispatch_cubes(&mut self, formula: &[Clause], cubes: &[Vec<i32>]) -> Vec<SolveResult> {
        let outcome = self.conquer(formula, cubes);
        self.last_error = outcome.as_ref().err().cloned();
        outcome.unwrap_or_else(|_| vec![SolveResult::Unknown; cubes.len()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_are_flattened_with_the_sign_bit_of_the_engine() {
        let flat = flatten(&[vec![1, -2], vec![], vec![-7, 3, 1]]).unwrap();
        assert_eq!(flat.starts, [0, 2, 2, 5]);
        assert_eq!(flat.literals, [0, 3, 13, 4, 0]);
        assert_eq!(flat.variables, 7);
        assert_eq!(flatten(&[]).unwrap(), Flat { starts: vec![0], literals: vec![], variables: 0 });
        // La variable mas alta posible cabe en una palabra.
        assert_eq!(flatten(&[vec![-i32::MAX]]).unwrap().literals, [u32::MAX - 2]);
        assert!(matches!(flatten(&[vec![1, 0]]), Err(GpuError::Malformed(_))));
        assert!(matches!(flatten(&[vec![i32::MIN]]), Err(GpuError::Malformed(_))));
    }

    #[test]
    fn a_stored_formula_is_recognised_without_flattening_it_again() {
        let formula = vec![vec![1, -2], vec![3], vec![]];
        let flat = flatten(&formula).unwrap();
        assert!(matches(&flat, &formula));
        // Mismos literales repartidos de otra forma, un signo cambiado, una clausula de mas.
        assert!(!matches(&flat, &[vec![1], vec![-2, 3], vec![]]));
        assert!(!matches(&flat, &[vec![1, 2], vec![3], vec![]]));
        assert!(!matches(&flat, &[vec![1, -2], vec![3], vec![], vec![]]));
        assert!(!matches(&flat, &[vec![1, -2], vec![3]]));
    }

    #[test]
    fn the_plan_counts_what_the_formula_needs_in_vram() {
        // R(5,5) sobre K_43: 903 variables, 1 925 196 clausulas de 10 literales.
        let plan = BufferPlan { variables: 903, clauses: 1_925_196, literals: 19_251_960, cubes: 1 << 16, cube_literals: 32 << 16 };
        assert_eq!(plan.words_per_cube(), 57);
        let words = 19_251_960 + 1_925_197 + (32 << 16) + (1 << 16) + 1 + 57 * (1 << 16) + (1 << 16);
        assert_eq!(plan.bytes(), 4 * words);
        let small = BufferPlan::for_formula(&[vec![1, -17], vec![5]], 8, 4);
        assert_eq!((small.variables, small.clauses, small.literals, small.cubes, small.cube_literals), (17, 2, 3, 8, 32));
        assert_eq!(small.words_per_cube(), 2);
        // Un plan vacio reserva una palabra por buffer, y uno desmesurado no desborda.
        assert_eq!(BufferPlan::for_formula(&[], 0, 0).bytes(), 4 * BUFFERS as u64);
        assert!(BufferPlan::for_formula(&[vec![1]], usize::MAX, usize::MAX).bytes() > 1 << 40);
    }

    // 2-coloraciones de las aristas de K_n sin triangulo de un solo color: R(3,3) = 6.
    fn triangles(n: i32) -> Vec<Vec<i32>> {
        let edge = |i: i32, j: i32| (i - 1) * n - (i - 1) * i / 2 + (j - i);
        let mut formula = Vec::new();
        for a in 1..=n {
            for b in a + 1..=n {
                for c in b + 1..=n {
                    let triangle = [edge(a, b), edge(a, c), edge(b, c)];
                    formula.push(triangle.to_vec());
                    formula.push(triangle.iter().map(|&variable| -variable).collect());
                }
            }
        }
        formula
    }

    // Todas las asignaciones completas de `variables` variables, como cubos.
    fn total_cubes(variables: i32) -> Vec<Vec<i32>> {
        (0..1_u32 << variables).map(|bits| {
            (1..=variables).map(|id| if bits >> (id - 1) & 1 == 1 { id } else { -id }).collect()
        }).collect()
    }

    #[test]
    fn the_reference_propagates_to_a_fixpoint_and_names_what_it_proves() {
        let formula = vec![vec![1, -2, 3], vec![-1, 2], vec![-3]];
        // x1 fuerza x2 y la unidad niega x3: todo satisfecho.
        assert_eq!(bcp_reference(&formula, &[1], 3), SolveResult::Sat(vec![1, 2, -3]));
        // Sin x1 y con x2 la primera clausula pide x3, que la unidad niega.
        assert_eq!(bcp_reference(&formula, &[-1, 2], 3), SolveResult::Unsat);
        assert_eq!(bcp_reference(&formula, &[], 3), SolveResult::Unknown);
        // Un cubo que se contradice, la clausula vacia y la formula vacia.
        assert_eq!(bcp_reference(&formula, &[2, -2], 3), SolveResult::Unsat);
        assert_eq!(bcp_reference(&[vec![]], &[], 2), SolveResult::Unsat);
        assert_eq!(bcp_reference(&[], &[2], 2), SolveResult::Sat(vec![-1, 2]));
        // Un literal repetido sigue siendo una unidad; una tautologia no obliga a nada.
        assert_eq!(bcp_reference(&[vec![1, 1], vec![2, -2]], &[], 2), SolveResult::Unknown);
        assert_eq!(bcp_reference(&[vec![1, 1], vec![-1, 2]], &[], 2), SolveResult::Sat(vec![1, 2]));
        // K_5 admite exactamente 12 coloraciones sin triangulo monocromatico; K_6, ninguna.
        let count = |n: i32, variables: i32| {
            let formula = triangles(n);
            total_cubes(variables).iter().filter(|cube| matches!(bcp_reference(&formula, cube, variables as usize), SolveResult::Sat(_))).count()
        };
        assert_eq!((count(5, 10), count(6, 15)), (12, 0));
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn without_the_feature_there_is_no_device_to_open() {
        let plan = BufferPlan::for_formula(&[vec![1, 2]], 4, 2);
        assert!(matches!(GpuConqueror::new(plan), Err(GpuError::NotCompiled)));
    }

    // Abre la tarjeta o, en una maquina sin GPU CUDA, deja pasar el test sin probar nada.
    #[cfg(feature = "gpu")]
    fn open(plan: BufferPlan) -> Option<GpuConqueror> {
        match GpuConqueror::new(plan) {
            Ok(conqueror) => Some(conqueror),
            Err(GpuError::Unavailable(reason)) => {
                eprintln!("sin GPU CUDA, test omitido: {reason}");
                None
            }
            Err(error) => panic!("{error}"),
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn the_formula_and_a_batch_reach_vram_and_each_cube_gets_its_verdict() {
        let formula = vec![vec![1, -2, 3], vec![-1, 2], vec![-3]];
        let cubes = vec![vec![1], vec![-1, 2], vec![]];
        let Some(mut conqueror) = open(BufferPlan::for_formula(&formula, 4, 2)) else { return };
        let info = conqueror.info().clone();
        assert!(!info.name.is_empty() && info.total_bytes >= info.free_bytes && info.free_bytes > 0);
        assert_eq!(conqueror.verify_formula(), Ok(false));
        assert_eq!(
            conqueror.dispatch_cubes(&formula, &cubes),
            [SolveResult::Sat(vec![1, 2, -3]), SolveResult::Unsat, SolveResult::Unknown],
        );
        assert_eq!(conqueror.last_error(), None);
        // Lo que hay en VRAM es lo que se subio, palabra por palabra.
        assert_eq!(conqueror.verify_formula(), Ok(true));
        let batch = flatten(&cubes).unwrap();
        assert_eq!(conqueror.device.download(CUBE_LITERALS, 0, batch.literals.len()).unwrap(), batch.literals);
        assert_eq!(conqueror.device.download(CUBE_STARTS, 0, batch.starts.len()).unwrap(), batch.starts);
        // Un lote que no cabe vuelve entero sin decidir y dice por que.
        let wide = vec![vec![1]; 5];
        assert_eq!(conqueror.dispatch_cubes(&formula, &wide), vec![SolveResult::Unknown; 5]);
        assert_eq!(conqueror.last_error(), Some(&GpuError::Capacity { what: "cubos del lote", needed: 5, reserved: 4 }));
        // Otra formula sustituye a la anterior; una que no cabe se rechaza.
        assert_eq!(conqueror.load_formula(&[vec![2, 3], vec![-1]]), Ok(()));
        assert_eq!(conqueror.verify_formula(), Ok(true));
        assert!(matches!(conqueror.load_formula(&[vec![4]]), Err(GpuError::Capacity { what: "variables", .. })));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn the_kernel_agrees_with_the_reference_on_random_formulas() {
        let plan = BufferPlan { variables: 48, clauses: 256, literals: 1024, cubes: 512, cube_literals: 4096 };
        let Some(mut conqueror) = open(plan) else { return };
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = |bound: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % bound
        };
        // Unknown, Unsat, Sat.
        let mut seen = [0_usize; 3];
        for _ in 0..60 {
            let variables = 4 + next(45);
            let literal = |next: &mut dyn FnMut(u64) -> u64| {
                let id = 1 + next(variables) as i32;
                if next(2) == 0 { id } else { -id }
            };
            let clauses = 1 + next(4 * variables);
            let formula: Vec<Vec<i32>> = (0..clauses).map(|_| (0..1 + next(4)).map(|_| literal(&mut next)).collect()).collect();
            let cubes: Vec<Vec<i32>> = (0..512).map(|_| (0..next(8)).map(|_| literal(&mut next)).collect()).collect();
            let results = conqueror.dispatch_cubes(&formula, &cubes);
            assert_eq!(conqueror.last_error(), None);
            for (cube, result) in cubes.iter().zip(&results) {
                assert_eq!(*result, bcp_reference(&formula, cube, plan.variables), "formula {formula:?}, cube {cube:?}");
                seen[match result { SolveResult::Unknown => 0, SolveResult::Unsat => 1, SolveResult::Sat(_) => 2 }] += 1;
            }
        }
        // Los tres veredictos aparecen de sobra: la comparacion no es vacia.
        assert!(seen.iter().all(|&count| count > 500), "{seen:?}");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn long_chains_span_several_launches_and_wide_batches_several_chunks() {
        // x1 y la cadena x_i -> x_(i+1) escrita en espiral: los eslabones impares hacia
        // atras y los pares hacia delante. Ni recorriendo las clausulas en un sentido ni en
        // el otro se avanza mas de un eslabon por pasada, y hacen falta muchas mas pasadas
        // de las que caben en un lanzamiento.
        let links = 300;
        let link = |id: i32| if id == 1 { vec![1] } else { vec![1 - id, id] };
        let formula: Vec<Vec<i32>> = (1..=links).rev().filter(|id| id % 2 == 1)
            .chain((1..=links).filter(|id| id % 2 == 0)).map(link).collect();
        let all_true: Vec<i32> = (1..=links).collect();
        // Negar cualquier eslabon rompe la cadena; afirmarlo la deja entera.
        let cubes: Vec<Vec<i32>> = (0..5000).map(|index| {
            let id = index % links + 1;
            vec![if index % 2 == 0 { -id } else { id }]
        }).collect();
        let Some(mut conqueror) = open(BufferPlan::for_formula(&formula, cubes.len(), 1)) else { return };
        let results = conqueror.dispatch_cubes(&formula, &cubes);
        assert_eq!(conqueror.last_error(), None);
        for (cube, result) in cubes.iter().zip(&results) {
            let expected = if cube[0] < 0 { SolveResult::Unsat } else { SolveResult::Sat(all_true.clone()) };
            assert_eq!(*result, expected, "cube {cube:?}");
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn exhaustive_cubes_reproduce_the_ramsey_number_of_triangles() {
        // Un cubo por coloracion completa: la GPU decide cada una sin ayuda.
        for (n, variables, colourings) in [(5, 10, 12), (6, 15, 0)] {
            let formula = triangles(n);
            let cubes = total_cubes(variables);
            let Some(mut conqueror) = open(BufferPlan::for_formula(&formula, cubes.len(), variables as usize)) else { return };
            let results = conqueror.dispatch_cubes(&formula, &cubes);
            assert_eq!(conqueror.last_error(), None);
            assert!(results.iter().all(|result| *result != SolveResult::Unknown));
            assert_eq!(results.iter().filter(|result| matches!(result, SolveResult::Sat(_))).count(), colourings);
            // Cada modelo es su propio cubo.
            assert!(cubes.iter().zip(&results).all(|(cube, result)| match result {
                SolveResult::Sat(model) => model == cube,
                _ => true,
            }));
        }
    }

    // Banco, no test: ritmo del nucleo con R(5,5) sobre K_n y cubos de dos literales, que
    // no propagan nada: cada hilo hace exactamente una pasada densa.
    //   GPU_BENCH_N=36 cargo test --release --features gpu --bin hunt throughput -- --ignored --nocapture
    #[cfg(feature = "gpu")]
    #[test]
    #[ignore]
    fn throughput_on_a_ramsey_formula() {
        let n: i32 = std::env::var("GPU_BENCH_N").ok().and_then(|text| text.parse().ok()).unwrap_or(30);
        let edge = |i: i32, j: i32| (i - 1) * n - (i - 1) * i / 2 + (j - i);
        let mut formula = Vec::new();
        for a in 1..=n { for b in a + 1..=n { for c in b + 1..=n { for d in c + 1..=n { for e in d + 1..=n {
            let clique = [a, b, c, d, e];
            let edges: Vec<i32> = (0..5).flat_map(|i| (i + 1..5).map(move |j| (i, j))).map(|(i, j)| edge(clique[i], clique[j])).collect();
            formula.push(edges.iter().map(|&variable| -variable).collect());
            formula.push(edges);
        } } } } }
        let variables = n * (n - 1) / 2;
        let cubes: Vec<Vec<i32>> = (1..=variables).flat_map(|a| (a + 1..=variables).map(move |b| vec![a, -b])).take(1 << 15).collect();
        let Some(mut conqueror) = open(BufferPlan::for_formula(&formula, cubes.len(), 2)) else { return };
        eprintln!("K_{n}: {} clausulas, {} cubos; lanzamiento por defecto de {} hilos", formula.len(), cubes.len(), conqueror.device.chunk);
        conqueror.load_formula(&formula).unwrap();
        for threads in [256, 1536, 6144, 12288, 24576, 49152] {
            conqueror.device.chunk = threads;
            let batch = &cubes[..cubes.len().min(2 * threads as usize)];
            let start = std::time::Instant::now();
            let results = conqueror.dispatch_cubes(&formula, batch);
            let seconds = start.elapsed().as_secs_f64();
            assert_eq!(conqueror.last_error(), None);
            assert!(results.iter().all(|result| *result == SolveResult::Unknown));
            eprintln!(
                "  {threads:>6} hilos por lanzamiento: {:>6} cubos en {:>8.1} ms, {:>8.0} cubos/s, {:.0} ms por lanzamiento; queda en {}",
                batch.len(), seconds * 1000.0, batch.len() as f64 / seconds, seconds * 1000.0 / 2.0, conqueror.device.chunk,
            );
        }
        // Un punado contra la referencia, para que el banco no mida un nucleo roto.
        for cube in cubes.iter().step_by(4099) {
            assert_eq!(bcp_reference(&formula, cube, variables as usize), SolveResult::Unknown);
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn a_plan_larger_than_the_card_is_refused_before_reserving() {
        let Some(conqueror) = open(BufferPlan::for_formula(&[vec![1]], 1, 1)) else { return };
        let total = conqueror.info().total_bytes;
        drop(conqueror);
        let huge = BufferPlan::for_formula(&[vec![1]], (total / 4) as usize + 1, 1);
        assert!(matches!(GpuConqueror::new(huge), Err(GpuError::OutOfMemory { .. })));
    }
}
