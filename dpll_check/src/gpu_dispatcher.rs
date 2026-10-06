// Mejora 3, fase 1: puente con la GPU para la fase "conquer" de cube and conquer.
//
// La CPU parte el problema en cubos y este modulo es quien los recibe por lotes para
// resolverlos en la tarjeta. En esta fase solo existe el puente: se abre el dispositivo
// CUDA, se reserva en VRAM lo que el nucleo va a necesitar, la formula y los cubos viajan
// hasta alli y queda fijada la firma de `dispatch_cubes`. Todavia no hay nucleo: ningun
// cubo se decide en la GPU y todos vuelven como `SolveResult::Unknown`.
//
// Todo lo que toca CUDA esta tras la caracteristica `gpu` de Cargo. Sin ella el modulo
// compila igual, `GpuConqueror::new` devuelve `GpuError::NotCompiled` y el binario no
// enlaza nada de NVIDIA. Este modulo no conoce el motor ni el motor lo conoce a el.
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
const BUFFERS: usize = 6;

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
    use super::{BufferPlan, GpuError, GpuInfo};
    use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DriverError};
    use std::sync::Arc;

    pub struct Device {
        // El contexto tiene que vivir lo que vivan los buffers.
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        buffers: Vec<CudaSlice<u32>>,
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
            let stream = context.default_stream();
            let buffers = plan.sizes().iter()
                .map(|&words| stream.alloc_zeros::<u32>(words).map_err(driver))
                .collect::<Result<Vec<_>, _>>()?;
            stream.synchronize().map_err(driver)?;
            Ok((Self { context, stream, buffers }, info))
        }
        // Copia `data` al principio del buffer; que quepa lo comprueba quien llama.
        pub fn upload(&mut self, buffer: usize, data: &[u32]) -> Result<(), GpuError> {
            if data.is_empty() {
                return Ok(());
            }
            let mut view = self.buffers[buffer].slice_mut(..data.len());
            self.stream.memcpy_htod(data, &mut view).map_err(driver)
        }
        pub fn download(&self, buffer: usize, words: usize) -> Result<Vec<u32>, GpuError> {
            if words == 0 {
                return Ok(Vec::new());
            }
            self.stream.clone_dtoh(&self.buffers[buffer].slice(..words)).map_err(driver)
        }
        pub fn free_bytes(&self) -> Result<u64, GpuError> {
            self.context.mem_get_info().map(|(free, _)| free as u64).map_err(driver)
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
        pub fn download(&self, _buffer: usize, _words: usize) -> Result<Vec<u32>, GpuError> {
            match *self {}
        }
        pub fn free_bytes(&self) -> Result<u64, GpuError> {
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
    // Abre la primera tarjeta CUDA y reserva en VRAM los buffers del plan, a cero.
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
        Ok(self.device.download(LITERALS, flat.literals.len())? == flat.literals
            && self.device.download(CLAUSE_STARTS, flat.starts.len())? == flat.starts)
    }
    fn stage(&mut self, formula: &[Clause], cubes: &[Vec<i32>]) -> Result<Flat, GpuError> {
        self.load_formula(formula)?;
        let batch = flatten(cubes)?;
        Self::fits("cubos del lote", cubes.len(), self.plan.cubes)?;
        Self::fits("literales de cubo", batch.literals.len(), self.plan.cube_literals)?;
        Self::fits("variables de los cubos", batch.variables as usize, self.plan.variables)?;
        self.device.upload(CUBE_LITERALS, &batch.literals)?;
        self.device.upload(CUBE_STARTS, &batch.starts)?;
        Ok(batch)
    }
    // Resuelve en la GPU un lote de cubos de `formula`: un veredicto por cubo, en el mismo
    // orden. Un cubo es una lista de literales DIMACS que se suponen ciertos.
    //
    // Fase 1: no hay nucleo. La formula y el lote se suben a VRAM y todos los cubos
    // vuelven `Unknown`. Si el puente falla (el lote no cabe, error del driver) tambien
    // vuelven `Unknown`, y `last_error` dice por que: la CPU se queda con ellos.
    pub fn dispatch_cubes(&mut self, formula: &[Clause], cubes: &[Vec<i32>]) -> Vec<SolveResult> {
        self.last_error = self.stage(formula, cubes).err();
        vec![SolveResult::Unknown; cubes.len()]
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
    fn the_formula_and_a_batch_reach_vram_and_no_cube_is_decided_yet() {
        let formula = vec![vec![1, -2, 3], vec![-1, 2], vec![-3]];
        let cubes = vec![vec![1], vec![-1, 2], vec![]];
        let Some(mut conqueror) = open(BufferPlan::for_formula(&formula, 4, 2)) else { return };
        let info = conqueror.info().clone();
        assert!(!info.name.is_empty() && info.total_bytes >= info.free_bytes && info.free_bytes > 0);
        assert_eq!(conqueror.verify_formula(), Ok(false));
        assert_eq!(conqueror.dispatch_cubes(&formula, &cubes), vec![SolveResult::Unknown; 3]);
        assert_eq!(conqueror.last_error(), None);
        // Lo que hay en VRAM es lo que se subio, palabra por palabra.
        assert_eq!(conqueror.verify_formula(), Ok(true));
        let batch = flatten(&cubes).unwrap();
        assert_eq!(conqueror.device.download(CUBE_LITERALS, batch.literals.len()).unwrap(), batch.literals);
        assert_eq!(conqueror.device.download(CUBE_STARTS, batch.starts.len()).unwrap(), batch.starts);
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
    fn a_plan_larger_than_the_card_is_refused_before_reserving() {
        let Some(conqueror) = open(BufferPlan::for_formula(&[vec![1]], 1, 1)) else { return };
        let total = conqueror.info().total_bytes;
        drop(conqueror);
        let huge = BufferPlan::for_formula(&[vec![1]], (total / 4) as usize + 1, 1);
        assert!(matches!(GpuConqueror::new(huge), Err(GpuError::OutOfMemory { .. })));
    }
}
