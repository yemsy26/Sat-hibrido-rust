use std::collections::{HashSet, VecDeque};
use super::proof::ProofLog;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
pub type Clause = Vec<i32>;
pub type Formula = Vec<Clause>;
#[derive(Clone, Copy, Debug, Default)]
pub struct SolveStats {
    pub decisions: usize,
    pub cube_decisions: usize,
    pub propagations: usize,
    pub conflicts: usize,
    pub learned_clauses: usize,
    pub backjumps: usize,
    pub nonchronological_backjumps: usize,
    pub restarts: usize,
    pub deleted_clauses: usize,
    pub database_reductions: usize,
    pub reignitions: usize,
    pub reignition_flips: usize,
    pub reignition_ms: f64,
    pub exported_clauses: usize,
    pub imported_clauses: usize,
    pub dropped_exports: usize,
    // Siempre 0 desde la Fase 12b: solo se importa en el nivel 0.
    pub import_backjumps: usize,
}
impl SolveStats {
    fn add_search(&mut self, other: Self) {
        self.decisions += other.decisions;
        self.cube_decisions += other.cube_decisions;
        self.propagations += other.propagations;
        self.conflicts += other.conflicts;
        self.learned_clauses += other.learned_clauses;
        self.backjumps += other.backjumps;
        self.nonchronological_backjumps += other.nonchronological_backjumps;
        self.restarts += other.restarts;
        self.deleted_clauses += other.deleted_clauses;
        self.database_reductions += other.database_reductions;
        self.reignitions += other.reignitions;
        self.reignition_flips += other.reignition_flips;
        self.reignition_ms += other.reignition_ms;
        self.exported_clauses += other.exported_clauses;
        self.imported_clauses += other.imported_clauses;
        self.dropped_exports += other.dropped_exports;
        self.import_backjumps += other.import_backjumps;
    }
}
#[derive(Clone, Copy, Debug)]
pub struct OracleConfig {
    // Tope de flips por variable en cada ejecución; 0 desactiva el oráculo.
    pub flips_per_variable: usize,
    pub seed: u64,
}
impl Default for OracleConfig {
    fn default() -> Self {
        Self { flips_per_variable: 16, seed: 0x243f_6a88_85a3_08d3 }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct ReignitionConfig {
    // Conflictos mínimos entre ejecuciones del oráculo; 0 lo desactiva.
    pub interval: usize,
    pub oracle: OracleConfig,
}
impl Default for ReignitionConfig {
    fn default() -> Self {
        Self { interval: 500, oracle: OracleConfig::default() }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct SwarmConfig {
    pub workers: usize,
    pub cubes_per_worker: usize,
    // Ventana de cláusulas retenidas en el anillo; 0 desactiva el intercambio.
    pub share_capacity: usize,
    // Conflictos mínimos entre sincronizaciones; se importa solo al reiniciar.
    pub share_interval: usize,
    pub reignition: ReignitionConfig,
    pub share_enabled: bool,
}
impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            workers: std::thread::available_parallelism().map_or(1, |count| count.get()),
            cubes_per_worker: 4,
            share_capacity: 8192,
            share_interval: 64,
            reignition: ReignitionConfig::default(),
            share_enabled: true,
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct SwarmStats {
    pub workers: usize,
    pub cubes_generated: usize,
    pub cubes_started: usize,
    pub cubes_unsat: usize,
    pub cubes_cancelled: usize,
    // Cubos que un prefiltro refutó antes de llegar a ningún worker.
    pub cubes_filtered: usize,
    pub total: SolveStats,
    // Snapshot auditable: IDs originales; nunca contiene cláusulas vacías.
    pub shared_clauses: Vec<Vec<i32>>,
}
// Telemetría del enjambre: lo que un observador externo puede leer mientras dura la búsqueda.
// Los workers solo suman, sin orden entre ellos ni con el lector, y la búsqueda nunca lee
// estos contadores. A mitad de búsqueda `conflicts` puede ir hasta PROGRESS_BATCH - 1 por
// detrás de cada worker; al terminar es la suma exacta de los conflictos de todos los cubos.
#[derive(Debug, Default)]
pub struct SwarmProgress {
    pub conflicts: AtomicU64,
    pub cubes_total: AtomicU64,
    // Cubos cerrados por los workers; los del prefiltro se cuentan aparte.
    pub cubes_closed: AtomicU64,
    pub cubes_filtered: AtomicU64,
    pub gpu_filtered: AtomicU64,
}
// Veredicto de un prefiltro de cubos: lo que demuestra de un cubo sin buscar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CubeVerdict {
    Unknown,
    // El cubo no tiene ninguna extensión que cumpla la fórmula.
    Unsat,
    // Un modelo: pasa a ser la respuesta del enjambre.
    Sat(Vec<i32>),
}
// Prefiltro de cubos: recibe los cubos recién generados, en literales DIMACS, y devuelve un
// veredicto por cubo en el mismo orden. El motor no sabe quién lo implementa, y quien lo
// aporta responde de lo que afirma: `Unsat` ha de valer para la fórmula del enjambre o para
// una parte de ella, y el modelo de `Sat` no se comprueba aquí.
pub type CubeFilter<'a> = &'a mut dyn FnMut(&[Vec<i32>]) -> Vec<CubeVerdict>;
// Un worker publica sus conflictos por lotes: una escritura atómica cada PROGRESS_BATCH
// conflictos, para no disputarse la línea de caché del contador en cada uno.
const PROGRESS_BATCH: usize = 64;
struct Beacon<'a> {
    progress: &'a SwarmProgress,
    // Conflictos del cubo en curso ya sumados al contador común.
    published: usize,
}
impl Beacon<'_> {
    fn publish(&mut self, conflicts: usize, flush: bool) {
        let pending = conflicts - self.published;
        if pending >= PROGRESS_BATCH || (flush && pending != 0) {
            self.progress.conflicts.fetch_add(pending as u64, Ordering::Relaxed);
            self.published = conflicts;
        }
    }
}
// Solo se exportan cláusulas glue; una importada entra con LBD 3 y debe ganarse la permanencia.
const EXPORT_LBD: u32 = 2;
const IMPORT_LBD: u32 = 3;
const EXPORT_BATCH: usize = 64;
const KNOWN_LIMIT: usize = 1 << 20;
// Lote inmutable de cláusulas codificadas como [n, literales...].
struct Batch {
    owner: u32,
    count: usize,
    words: Arc<[u32]>,
}
struct CloudLog {
    batches: VecDeque<Batch>,
    // Secuencia del próximo lote; el más antiguo retenido es `next - batches.len()`.
    next: u64,
    clauses: usize,
}
// Anillo de lotes sin tope de vida: `capacity` es la ventana de cláusulas retenidas.
pub struct CloudPool { log: Mutex<CloudLog>, capacity: usize }
impl CloudPool {
    pub fn new(capacity: usize) -> Self {
        Self { log: Mutex::new(CloudLog { batches: VecDeque::new(), next: 0, clauses: 0 }), capacity }
    }
    // El candado solo cubre un push de puntero y la expulsión de los lotes más antiguos.
    fn publish(&self, batch: Batch, wait: bool) -> bool {
        let guard = if wait { self.log.lock().ok() } else { self.log.try_lock().ok() };
        let Some(mut log) = guard else { return false; };
        log.clauses += batch.count;
        log.batches.push_back(batch);
        log.next += 1;
        while log.clauses > self.capacity {
            let Some(oldest) = log.batches.pop_front() else { break; };
            log.clauses -= oldest.count;
        }
        true
    }
    // Lotes ajenos desde `cursor`; un lector rezagado continúa en el más antiguo retenido.
    fn harvest(&self, owner: u32, cursor: &mut u64, batches: &mut Vec<Arc<[u32]>>) {
        let Ok(log) = self.log.try_lock() else { return; };
        let first = log.next - log.batches.len() as u64;
        let skip = cursor.saturating_sub(first) as usize;
        batches.extend(log.batches.iter().skip(skip)
            .filter(|batch| batch.owner != owner).map(|batch| Arc::clone(&batch.words)));
        *cursor = log.next;
    }
    fn snapshot(&self, variables: &[u32]) -> Vec<Vec<i32>> {
        let Ok(log) = self.log.lock() else { return Vec::new(); };
        log.batches.iter().flat_map(|batch| clauses_of(&batch.words)).map(|clause| {
            clause.iter().map(|&literal| {
                let variable = variables[(literal >> 1) as usize] as i32;
                if literal & 1 == 0 { variable } else { -variable }
            }).collect()
        }).collect()
    }
}
fn clauses_of(words: &[u32]) -> impl Iterator<Item = &[u32]> {
    let mut rest = words;
    std::iter::from_fn(move || {
        let (&size, tail) = rest.split_first()?;
        let (clause, tail) = tail.split_at(size as usize);
        rest = tail;
        Some(clause)
    })
}
// Huella independiente del orden de los literales: el filtro no necesita canonizar.
fn fingerprint(clause: &[u32]) -> u64 {
    clause.iter().fold(clause.len() as u64, |sum, &literal| {
        let mut mixed = u64::from(literal).wrapping_add(0x9e37_79b9_7f4a_7c15);
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        sum.wrapping_add(mixed ^ (mixed >> 31))
    })
}
// Contexto de intercambio de un worker: sobrevive entre cubos, así que nada se reimporta.
struct Sharing<'a> {
    pool: &'a CloudPool,
    owner: u32,
    interval: u64,
    cursor: u64,
    last_sync: u64,
    known: HashSet<u64>,
    outbox: Vec<u32>,
    pending: usize,
    inbox: Vec<u32>,
}
impl<'a> Sharing<'a> {
    fn new(pool: &'a CloudPool, owner: u32, interval: u64) -> Self {
        Self {
            pool, owner, interval, cursor: 0, last_sync: 0, known: HashSet::new(),
            outbox: Vec::new(), pending: 0, inbox: Vec::new(),
        }
    }
    fn remember(&mut self, clause: &[u32]) -> bool {
        // Olvidar el filtro solo permite algún duplicado; nunca afecta a la corrección.
        if self.known.len() >= KNOWN_LIMIT {
            self.known.clear();
        }
        self.known.insert(fingerprint(clause))
    }
    fn export(&mut self, clause: &[u32], stats: &mut SolveStats) {
        if !self.remember(clause) {
            return;
        }
        self.outbox.push(clause.len() as u32);
        self.outbox.extend_from_slice(clause);
        self.pending += 1;
        if self.pending >= EXPORT_BATCH.min(self.pool.capacity) {
            self.flush(stats, false);
        }
    }
    fn flush(&mut self, stats: &mut SolveStats, wait: bool) {
        if self.pending == 0 {
            return;
        }
        // Un lote nunca excede el anillo: se conservan las cláusulas más recientes.
        let count = self.pending.min(self.pool.capacity);
        let mut start = 0;
        for _ in count..self.pending {
            start += 1 + self.outbox[start] as usize;
        }
        let batch = Batch { owner: self.owner, count, words: Arc::from(&self.outbox[start..]) };
        if self.pool.publish(batch, wait) {
            stats.exported_clauses += count;
            stats.dropped_exports += self.pending - count;
            self.outbox.clear();
            self.pending = 0;
        }
    }
    // Publica lo pendiente y deja en `inbox` las cláusulas ajenas aún no vistas.
    fn exchange(&mut self, conflicts: u64, stats: &mut SolveStats) {
        self.flush(stats, false);
        let mut batches = Vec::new();
        self.pool.harvest(self.owner, &mut self.cursor, &mut batches);
        for words in batches {
            for clause in clauses_of(&words) {
                if self.remember(clause) {
                    self.inbox.push(clause.len() as u32);
                    self.inbox.extend_from_slice(clause);
                }
            }
        }
        self.last_sync = conflicts;
    }
}
enum SearchOutcome { Sat(Vec<i32>), Unsat, Cancelled }
struct SearchControl<'a, 'p> {
    cube: &'a [u32],
    cancel: Option<&'a AtomicBool>,
    sharing: Option<&'a mut Sharing<'p>>,
    // Traza DRAT de la búsqueda en serie; el enjambre no la emite.
    proof: Option<&'a mut dyn ProofLog>,
    // Telemetría del enjambre; sin observador no se publica nada.
    progress: Option<Beacon<'a>>,
}
// Pesos enteros de probSAT, 2^16 * 2.06^(-roturas): cuanto más rompe un flip, menos probable.
const BREAK_WEIGHTS: [u64; 16] =
    [65536, 31814, 15443, 7497, 3639, 1767, 858, 416, 202, 98, 48, 23, 11, 5, 3, 1];
// Oráculo discreto: búsqueda local estocástica sobre bits, sin aritmética flotante.
#[derive(Clone)]
struct Walker {
    literals: Vec<u32>,
    offsets: Vec<u32>,
    occurrences: Vec<u32>,
    occurrence_offsets: Vec<u32>,
    values: Vec<bool>,
    fixed: Vec<bool>,
    // Literales verdaderos de cada cláusula y su XOR: con uno solo, el XOR es ese literal.
    satisfied: Vec<u32>,
    critical: Vec<u32>,
    breaks: Vec<u32>,
    unsat: Vec<u32>,
    slots: Vec<u32>,
    // Mejor asignación vista y variables volteadas desde ella dentro de la ráfaga en curso.
    best: usize,
    best_values: Vec<bool>,
    journal: Vec<u32>,
    synced: usize,
    rng: u64,
}
impl Walker {
    fn new(variables: usize, literals: Vec<u32>, offsets: Vec<u32>, seed: u64) -> Self {
        let clauses = offsets.len() - 1;
        let mut occurrence_offsets = vec![0_u32; 2 * variables + 1];
        for &literal in &literals {
            occurrence_offsets[literal as usize + 1] += 1;
        }
        for literal in 0..2 * variables {
            occurrence_offsets[literal + 1] += occurrence_offsets[literal];
        }
        let mut cursor = occurrence_offsets.clone();
        let mut occurrences = vec![0_u32; literals.len()];
        for clause in 0..clauses {
            for &literal in &literals[offsets[clause] as usize..offsets[clause + 1] as usize] {
                occurrences[cursor[literal as usize] as usize] = clause as u32;
                cursor[literal as usize] += 1;
            }
        }
        let mut walker = Self {
            literals, offsets, occurrences, occurrence_offsets,
            values: vec![false; variables],
            fixed: vec![false; variables],
            satisfied: vec![0; clauses],
            critical: vec![0; clauses],
            breaks: vec![0; variables],
            unsat: Vec::new(),
            slots: vec![NONE; clauses],
            best: 0,
            best_values: Vec::new(),
            journal: Vec::new(),
            synced: 0,
            rng: seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1,
        };
        // Asignación inicial aleatoria.
        for variable in 0..variables {
            walker.values[variable] = walker.next() >> 63 == 1;
        }
        for clause in 0..clauses {
            let (mut count, mut xor) = (0, 0);
            for position in walker.offsets[clause] as usize..walker.offsets[clause + 1] as usize {
                let literal = walker.literals[position];
                if walker.values[variable_of(literal)] == (literal & 1 == 0) {
                    count += 1;
                    xor ^= literal;
                }
            }
            walker.satisfied[clause] = count;
            walker.critical[clause] = xor;
            match count {
                0 => walker.violate(clause),
                1 => walker.breaks[variable_of(xor)] += 1,
                _ => {}
            }
        }
        walker.settle();
        walker
    }
    fn next(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn reseed(&mut self, salt: u64) {
        self.rng = (self.rng ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15)) | 1;
    }
    // La asignación actual pasa a ser la referencia: se usa tras fijar literales de la raíz.
    fn settle(&mut self) {
        self.best = self.unsat.len();
        self.best_values.clone_from(&self.values);
        self.journal.clear();
    }
    fn violate(&mut self, clause: usize) {
        self.slots[clause] = self.unsat.len() as u32;
        self.unsat.push(clause as u32);
    }
    fn repair(&mut self, clause: usize) {
        let slot = self.slots[clause];
        let last = self.unsat.pop().expect("A violated clause is listed");
        if last as usize != clause {
            self.unsat[slot as usize] = last;
            self.slots[last as usize] = slot;
        }
        self.slots[clause] = NONE;
    }
    // Coste proporcional a las ocurrencias de la variable, no al tamaño de la fórmula.
    fn flip(&mut self, variable: usize) {
        let gained = ((variable as u32) << 1) | u32::from(self.values[variable]);
        let lost = gained ^ 1;
        self.values[variable] = !self.values[variable];
        let range = self.occurrence_offsets[gained as usize]..self.occurrence_offsets[gained as usize + 1];
        for index in range {
            let clause = self.occurrences[index as usize] as usize;
            let before = self.satisfied[clause];
            self.satisfied[clause] = before + 1;
            match before {
                0 => {
                    self.repair(clause);
                    self.breaks[variable] += 1;
                }
                // El antiguo único verdadero deja de ser crítico.
                1 => self.breaks[variable_of(self.critical[clause])] -= 1,
                _ => {}
            }
            self.critical[clause] ^= gained;
        }
        let range = self.occurrence_offsets[lost as usize]..self.occurrence_offsets[lost as usize + 1];
        for index in range {
            let clause = self.occurrences[index as usize] as usize;
            let after = self.satisfied[clause] - 1;
            self.satisfied[clause] = after;
            self.critical[clause] ^= lost;
            match after {
                0 => {
                    self.violate(clause);
                    self.breaks[variable] -= 1;
                }
                // El verdadero que queda pasa a ser crítico.
                1 => self.breaks[variable_of(self.critical[clause])] += 1,
                _ => {}
            }
        }
    }
    fn fix(&mut self, literal: u32) {
        let variable = variable_of(literal);
        if self.values[variable] != (literal & 1 == 0) {
            self.flip(variable);
        }
        self.fixed[variable] = true;
    }
    #[inline]
    fn weight(&self, variable: usize) -> u64 {
        if self.fixed[variable] {
            0
        } else {
            BREAK_WEIGHTS[(self.breaks[variable] as usize).min(BREAK_WEIGHTS.len() - 1)]
        }
    }
    // Ráfaga probSAT: cláusula violada al azar y flip con probabilidad decreciente en roturas.
    // La caminata continúa entre ráfagas; `best_values` conserva la mejor asignación vista.
    fn walk(&mut self, flips: usize, cancel: Option<&AtomicBool>) -> usize {
        let mut done = 0;
        let mut improved = false;
        self.journal.clear();
        while done < flips && !self.unsat.is_empty() {
            if done & 4095 == 4095 && cancel.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                break;
            }
            let draw = self.next();
            let clause = self.unsat[(draw % self.unsat.len() as u64) as usize] as usize;
            let (start, end) = (self.offsets[clause] as usize, self.offsets[clause + 1] as usize);
            let total: u64 = (start..end).map(|position| self.weight(variable_of(self.literals[position]))).sum();
            // Todos sus literales son falsos en la raíz: el CDCL ya tiene ese conflicto.
            if total == 0 {
                break;
            }
            let mut pick = self.next() % total;
            for position in start..end {
                let variable = variable_of(self.literals[position]);
                let weight = self.weight(variable);
                if pick < weight {
                    self.flip(variable);
                    if improved {
                        self.journal.push(variable as u32);
                    }
                    break;
                }
                pick -= weight;
            }
            done += 1;
            if self.unsat.len() < self.best {
                self.best = self.unsat.len();
                self.journal.clear();
                improved = true;
            }
        }
        if improved {
            // La mejor asignación es la actual deshaciendo los flips posteriores a ella.
            self.best_values.clone_from(&self.values);
            for &variable in &self.journal {
                self.best_values[variable as usize] = !self.best_values[variable as usize];
            }
        }
        done
    }
}
const DECAY: f64 = 0.95;
const CLEANUP_INTERVAL: u64 = 64;
const MAX_ACTIVITY: f64 = 1e100;
// Arena: [tamaño, LBD | flags, literales...]. Unidades y binarias no viven en ella.
const HEADER: usize = 2;
const LEARNED: u32 = 1 << 31;
const GARBAGE: u32 = 1 << 30;
const LOCKED: u32 = 1 << 29;
const USED: u32 = 1 << 28;
const LBD_MASK: u32 = USED - 1;
// Bit alto de `Watcher::reference` y de una razón: el resto es el otro literal de una binaria.
const BINARY: u32 = 1 << 31;
const NO_REASON: u32 = u32::MAX;
const NONE: u32 = u32::MAX;
const RESTART_MARGIN: f64 = 1.1;
const RESTART_INTERVAL: u64 = 2;
const LUBY_UNIT: u64 = 1024;
const MODE_INTERVAL: u64 = 1000;
// Tras la ignición inicial el oráculo gasta como mucho un flip por cada ORACLE_EFFORT
// literales propagados: su coste queda acotado por el trabajo del CDCL, no por V.
const ORACLE_EFFORT: u64 = 8;
#[derive(Clone, Copy)]
struct Watcher {
    blocker: u32,
    reference: u32,
}
#[derive(Clone, Copy)]
enum Conflict {
    Binary(u32, u32),
    Long(u32),
}
// Media exponencial con corrección de sesgo: la primera muestra ya es representativa.
#[derive(Clone, Copy)]
struct Ema {
    value: f64,
    biased: f64,
    alpha: f64,
    correction: f64,
}
impl Ema {
    fn new(alpha: f64) -> Self {
        Self { value: 0.0, biased: 0.0, alpha, correction: 1.0 }
    }
    fn update(&mut self, sample: f64) {
        self.biased += self.alpha * (sample - self.biased);
        self.correction *= 1.0 - self.alpha;
        self.value = self.biased / (1.0 - self.correction);
    }
}
// Cola de prioridad indexada (EVSIDS): mayor puntuación primero; a igualdad, menor índice.
#[derive(Clone)]
struct Heap {
    order: Vec<u32>,
    position: Vec<u32>,
}
impl Heap {
    fn new(variables: usize) -> Self {
        Self { order: Vec::with_capacity(variables), position: vec![NONE; variables] }
    }
    #[inline]
    fn before(scores: &[f64], a: u32, b: u32) -> bool {
        let (x, y) = (scores[a as usize], scores[b as usize]);
        x > y || (x == y && a < b)
    }
    fn contains(&self, variable: u32) -> bool {
        self.position[variable as usize] != NONE
    }
    fn insert(&mut self, variable: u32, scores: &[f64]) {
        if self.contains(variable) {
            return;
        }
        self.order.push(variable);
        self.up(self.order.len() - 1, scores);
    }
    fn increase(&mut self, variable: u32, scores: &[f64]) {
        self.up(self.position[variable as usize] as usize, scores);
    }
    fn pop(&mut self, scores: &[f64]) -> Option<u32> {
        let top = *self.order.first()?;
        let last = self.order.pop().expect("Heap is not empty");
        self.position[top as usize] = NONE;
        if top != last {
            self.order[0] = last;
            self.down(0, scores);
        }
        Some(top)
    }
    fn up(&mut self, mut index: usize, scores: &[f64]) {
        let variable = self.order[index];
        while index > 0 {
            let parent = (index - 1) / 2;
            let other = self.order[parent];
            if !Self::before(scores, variable, other) {
                break;
            }
            self.order[index] = other;
            self.position[other as usize] = index as u32;
            index = parent;
        }
        self.order[index] = variable;
        self.position[variable as usize] = index as u32;
    }
    fn down(&mut self, mut index: usize, scores: &[f64]) {
        let variable = self.order[index];
        loop {
            let mut child = 2 * index + 1;
            if child >= self.order.len() {
                break;
            }
            if child + 1 < self.order.len()
                && Self::before(scores, self.order[child + 1], self.order[child])
            {
                child += 1;
            }
            let other = self.order[child];
            if !Self::before(scores, other, variable) {
                break;
            }
            self.order[index] = other;
            self.position[other as usize] = index as u32;
            index = child;
        }
        self.order[index] = variable;
        self.position[variable as usize] = index as u32;
    }
    fn heapify(&mut self, scores: &[f64]) {
        for index in (0..self.order.len() / 2).rev() {
            self.down(index, scores);
        }
    }
}
// VMTF: lista doblemente enlazada con marcas de tiempo; `last` es el frente.
#[derive(Clone)]
struct Queue {
    previous: Vec<u32>,
    next: Vec<u32>,
    stamps: Vec<u64>,
    last: u32,
    search: u32,
    stamp: u64,
}
impl Queue {
    fn new(variables: usize) -> Self {
        Self {
            previous: vec![NONE; variables],
            next: vec![NONE; variables],
            stamps: vec![0; variables],
            last: NONE,
            search: NONE,
            stamp: 0,
        }
    }
    fn push(&mut self, variable: u32) {
        self.previous[variable as usize] = self.last;
        self.next[variable as usize] = NONE;
        if self.last != NONE {
            self.next[self.last as usize] = variable;
        }
        self.last = variable;
        self.stamp += 1;
        self.stamps[variable as usize] = self.stamp;
    }
    fn bump(&mut self, variable: u32) {
        if self.last == variable {
            return;
        }
        let (previous, next) = (self.previous[variable as usize], self.next[variable as usize]);
        if previous != NONE {
            self.next[previous as usize] = next;
        }
        // `variable` no es el frente, así que `next` existe.
        self.previous[next as usize] = previous;
        self.push(variable);
    }
    // Invariante: toda variable con marca posterior a `search` está asignada.
    fn unassign(&mut self, variable: u32) {
        if self.search == NONE
            || self.stamps[variable as usize] > self.stamps[self.search as usize]
        {
            self.search = variable;
        }
    }
}
#[inline]
fn variable_of(literal: u32) -> usize {
    (literal >> 1) as usize
}
#[derive(Clone)]
struct State {
    arena: Vec<u32>,
    watches: Vec<Vec<Watcher>>,
    // Indexado por literal: 1 verdadero, -1 falso, 0 libre.
    values: Vec<i8>,
    levels: Vec<u32>,
    reasons: Vec<u32>,
    phases: Vec<u8>,
    trail: Vec<u32>,
    trail_limits: Vec<usize>,
    head: usize,
    scores: Vec<f64>,
    increment: f64,
    heap: Heap,
    queue: Queue,
    stable: bool,
    seen: Vec<bool>,
    analyzed: Vec<u32>,
    learned: Vec<u32>,
    stack: Vec<u32>,
    level_stamps: Vec<u64>,
    level_stamp: u64,
    fast_lbd: Ema,
    slow_lbd: Ema,
    conflicts: u64,
    next_restart: u64,
    luby: (u64, u64),
    mode_span: u64,
    next_mode: u64,
    cleanup_interval: u64,
    next_cleanup: u64,
    walker: Option<Walker>,
    next_reignition: u64,
    ignited: bool,
    propagated: u64,
    reignition_mark: u64,
    root_unsat: bool,
    stats: SolveStats,
}
impl State {
    fn new(variables: usize, stats: SolveStats) -> Self {
        Self {
            arena: Vec::new(),
            watches: vec![Vec::new(); 2 * variables],
            values: vec![0; 2 * variables],
            levels: vec![0; variables],
            reasons: vec![NO_REASON; variables],
            phases: vec![0; variables],
            trail: Vec::with_capacity(variables),
            trail_limits: Vec::new(),
            head: 0,
            scores: vec![0.0; variables],
            increment: 1.0,
            heap: Heap::new(variables),
            queue: Queue::new(variables),
            stable: false,
            seen: vec![false; variables],
            analyzed: Vec::new(),
            learned: Vec::new(),
            stack: Vec::new(),
            level_stamps: vec![0; variables + 1],
            level_stamp: 0,
            fast_lbd: Ema::new(0.03),
            slow_lbd: Ema::new(1e-5),
            conflicts: 0,
            next_restart: RESTART_INTERVAL,
            luby: (1, 1),
            mode_span: MODE_INTERVAL,
            next_mode: MODE_INTERVAL,
            cleanup_interval: CLEANUP_INTERVAL,
            next_cleanup: CLEANUP_INTERVAL,
            walker: None,
            next_reignition: 0,
            ignited: false,
            propagated: 0,
            reignition_mark: 0,
            root_unsat: false,
            stats,
        }
    }
    #[inline]
    fn value(&self, literal: u32) -> i8 {
        self.values[literal as usize]
    }
    #[inline]
    fn level(&self) -> usize {
        self.trail_limits.len()
    }
    #[inline]
    fn assign(&mut self, literal: u32, reason: u32) {
        let variable = variable_of(literal);
        debug_assert_eq!(self.values[literal as usize], 0);
        self.values[literal as usize] = 1;
        self.values[(literal ^ 1) as usize] = -1;
        self.levels[variable] = self.trail_limits.len() as u32;
        self.reasons[variable] = reason;
        // Phase saving de toda asignación, no solo de las decisiones.
        self.phases[variable] = (literal & 1) as u8;
        self.trail.push(literal);
        self.stats.propagations += usize::from(reason != NO_REASON);
    }
    fn assign_root(&mut self, literal: u32) {
        debug_assert_eq!(self.level(), 0);
        match self.value(literal) {
            0 => self.assign(literal, NO_REASON),
            1 => {}
            _ => self.root_unsat = true,
        }
    }
    // Binaria inline: el watcher lleva el otro literal y la propagación no toca la arena.
    fn watch_binary(&mut self, first: u32, second: u32) {
        self.watches[first as usize].push(Watcher { blocker: second, reference: BINARY | second });
        self.watches[second as usize].push(Watcher { blocker: first, reference: BINARY | first });
    }
    fn add_long(&mut self, literals: &[u32], learned: bool, lbd: u32) -> u32 {
        debug_assert!(literals.len() > 2);
        let reference = u32::try_from(self.arena.len()).ok()
            .filter(|reference| reference & BINARY == 0)
            .expect("Clause arena exceeds 2^31 words");
        self.arena.push(literals.len() as u32);
        self.arena.push(if learned { LEARNED | lbd.min(LBD_MASK) } else { 0 });
        self.arena.extend_from_slice(literals);
        // Los vigilados son siempre las posiciones 0 y 1; cada uno bloquea al otro.
        self.watches[literals[0] as usize].push(Watcher { blocker: literals[1], reference });
        self.watches[literals[1] as usize].push(Watcher { blocker: literals[0], reference });
        reference
    }
    fn add_original(&mut self, clause: &[u32]) {
        match *clause {
            [literal] => self.assign_root(literal),
            [first, second] => self.watch_binary(first, second),
            _ => {
                self.add_long(clause, false, 0);
            }
        }
    }
    // `head` recorre el propio trail hasta su cola (`trail.len()`): no hay cola de unidades.
    fn propagate(&mut self) -> Option<Conflict> {
        while self.head < self.trail.len() {
            let false_literal = self.trail[self.head] ^ 1;
            self.head += 1;
            self.propagated += 1;
            let mut watchers = std::mem::take(&mut self.watches[false_literal as usize]);
            let mut conflict = None;
            let (mut read, mut write) = (0, 0);
            while read < watchers.len() {
                let watcher = watchers[read];
                read += 1;
                let blocker_value = self.values[watcher.blocker as usize];
                if blocker_value > 0 {
                    watchers[write] = watcher;
                    write += 1;
                    continue;
                }
                if watcher.reference & BINARY != 0 {
                    watchers[write] = watcher;
                    write += 1;
                    if blocker_value < 0 {
                        conflict = Some(Conflict::Binary(false_literal, watcher.blocker));
                        break;
                    }
                    self.assign(watcher.blocker, BINARY | false_literal);
                    continue;
                }
                let start = watcher.reference as usize + HEADER;
                // El literal falso pasa a la posición 1; la 0 queda para el implicado.
                if self.arena[start] == false_literal {
                    self.arena.swap(start, start + 1);
                }
                let first = self.arena[start];
                let first_value = self.values[first as usize];
                let moved = Watcher { blocker: first, reference: watcher.reference };
                if first_value > 0 {
                    watchers[write] = moved;
                    write += 1;
                    continue;
                }
                let end = start + self.arena[start - HEADER] as usize;
                let mut replacement = None;
                for position in start + 2..end {
                    if self.values[self.arena[position] as usize] >= 0 {
                        replacement = Some(position);
                        break;
                    }
                }
                if let Some(position) = replacement {
                    let literal = self.arena[position];
                    self.arena[start + 1] = literal;
                    self.arena[position] = false_literal;
                    // `literal` no es falso, luego nunca es la lista que se está recorriendo.
                    self.watches[literal as usize].push(moved);
                    continue;
                }
                watchers[write] = moved;
                write += 1;
                if first_value < 0 {
                    conflict = Some(Conflict::Long(watcher.reference));
                    break;
                }
                self.assign(first, watcher.reference);
            }
            // Compactación in situ: los watchers movidos a otra lista no se copian.
            let remaining = watchers.len() - read;
            watchers.copy_within(read.., write);
            watchers.truncate(write + remaining);
            self.watches[false_literal as usize] = watchers;
            if conflict.is_some() {
                return conflict;
            }
        }
        None
    }
    // En la raíz un literal verdadero satisface la cláusula y uno falso desaparece:
    // lo que queda está libre, así que adjuntarla nunca propaga ni fuerza un backjump.
    fn import(&mut self, clause: &[u32]) {
        debug_assert_eq!(self.level(), 0);
        let mut literals = std::mem::take(&mut self.learned);
        literals.clear();
        let satisfied = clause.iter().any(|&literal| match self.value(literal) {
            1 => true,
            0 => {
                literals.push(literal);
                false
            }
            _ => false,
        });
        if !satisfied {
            self.stats.imported_clauses += 1;
            match *literals {
                [] => self.root_unsat = true,
                [literal] => self.assign_root(literal),
                [first, second] => self.watch_binary(first, second),
                _ => {
                    self.add_long(&literals, true, IMPORT_LBD);
                }
            }
        }
        self.learned = literals;
    }
    fn absorb(&mut self, inbox: &mut Vec<u32>) {
        for clause in clauses_of(inbox) {
            self.import(clause);
        }
        inbox.clear();
    }
    #[inline]
    fn mark(&mut self, literal: u32, pending: &mut usize) {
        let variable = variable_of(literal);
        if self.seen[variable] || self.levels[variable] == 0 {
            return;
        }
        self.seen[variable] = true;
        self.analyzed.push(variable as u32);
        if self.levels[variable] as usize == self.trail_limits.len() {
            *pending += 1;
        } else {
            self.learned.push(literal);
        }
    }
    fn mark_clause(&mut self, reference: u32, skip: usize, pending: &mut usize) {
        let start = reference as usize + HEADER;
        let end = start + self.arena[reference as usize] as usize;
        // Una cláusula que participa en un análisis sobrevive a la siguiente reducción.
        self.arena[reference as usize + 1] |= USED;
        for position in start + skip..end {
            let literal = self.arena[position];
            self.mark(literal, pending);
        }
    }
    fn expand(&mut self, literal: u32, abstract_levels: u32) -> bool {
        let variable = variable_of(literal);
        if self.seen[variable] || self.levels[variable] == 0 {
            return true;
        }
        if self.reasons[variable] == NO_REASON
            || (1_u32 << (self.levels[variable] & 31)) & abstract_levels == 0
        {
            return false;
        }
        self.seen[variable] = true;
        self.analyzed.push(variable as u32);
        self.stack.push(literal);
        true
    }
    // Un literal sobra si su antecedente se deriva, recursivamente, de literales ya presentes.
    fn redundant(&mut self, literal: u32, abstract_levels: u32) -> bool {
        let top = self.analyzed.len();
        self.stack.clear();
        self.stack.push(literal);
        while let Some(literal) = self.stack.pop() {
            let reason = self.reasons[variable_of(literal)];
            let implied = if reason & BINARY != 0 {
                self.expand(reason ^ BINARY, abstract_levels)
            } else {
                let start = reason as usize + HEADER;
                let end = start + self.arena[reason as usize] as usize;
                (start + 1..end).all(|position| {
                    let other = self.arena[position];
                    self.expand(other, abstract_levels)
                })
            };
            if !implied {
                for index in top..self.analyzed.len() {
                    self.seen[self.analyzed[index] as usize] = false;
                }
                self.analyzed.truncate(top);
                return false;
            }
        }
        true
    }
    // 1UIP minimizado en `self.learned`; devuelve (nivel de backjump, LBD).
    fn analyze(&mut self, conflict: Conflict) -> (usize, u32) {
        debug_assert!(self.level() > 0);
        self.learned.clear();
        self.learned.push(0);
        let mut pending = 0;
        match conflict {
            Conflict::Binary(first, second) => {
                self.mark(first, &mut pending);
                self.mark(second, &mut pending);
            }
            Conflict::Long(reference) => self.mark_clause(reference, 0, &mut pending),
        }
        let mut index = self.trail.len();
        let uip = loop {
            debug_assert!(pending > 0, "Conflict has a current-level pivot");
            let literal = loop {
                index -= 1;
                let literal = self.trail[index];
                if self.seen[variable_of(literal)] {
                    break literal;
                }
            };
            pending -= 1;
            if pending == 0 {
                break literal;
            }
            let reason = self.reasons[variable_of(literal)];
            debug_assert_ne!(reason, NO_REASON, "A non-UIP pivot has an antecedent");
            if reason & BINARY != 0 {
                self.mark(reason ^ BINARY, &mut pending);
            } else {
                // El literal implicado ocupa la posición 0 de su antecedente.
                self.mark_clause(reason, 1, &mut pending);
            }
        };
        self.learned[0] = uip ^ 1;
        let bumped = self.analyzed.len();
        let abstract_levels = self.learned[1..].iter().fold(0_u32, |levels, &literal| {
            levels | 1 << (self.levels[variable_of(literal)] & 31)
        });
        let mut kept = 1;
        for index in 1..self.learned.len() {
            let literal = self.learned[index];
            if self.reasons[variable_of(literal)] == NO_REASON
                || !self.redundant(literal, abstract_levels)
            {
                self.learned[kept] = literal;
                kept += 1;
            }
        }
        self.learned.truncate(kept);
        let mut backjump_level = 0;
        if self.learned.len() > 1 {
            let mut highest = 1;
            for index in 2..self.learned.len() {
                if self.levels[variable_of(self.learned[index])]
                    > self.levels[variable_of(self.learned[highest])]
                {
                    highest = index;
                }
            }
            self.learned.swap(1, highest);
            backjump_level = self.levels[variable_of(self.learned[1])] as usize;
        }
        // LBD: niveles de decisión distintos entre los literales aprendidos.
        self.level_stamp += 1;
        let mut lbd = 0;
        for &literal in &self.learned {
            let level = self.levels[variable_of(literal)] as usize;
            if self.level_stamps[level] != self.level_stamp {
                self.level_stamps[level] = self.level_stamp;
                lbd += 1;
            }
        }
        // Se actualizan TODAS las variables vistas en la derivación, no solo las del conflicto.
        let mut analyzed = std::mem::take(&mut self.analyzed);
        if self.stable {
            for &variable in &analyzed[..bumped] {
                self.bump_score(variable);
            }
            self.increment /= DECAY;
            if self.increment > MAX_ACTIVITY {
                self.rescale();
            }
        } else {
            // Al frente conservando el orden relativo que ya tenían.
            analyzed[..bumped].sort_unstable_by_key(|&variable| self.queue.stamps[variable as usize]);
            for &variable in &analyzed[..bumped] {
                self.queue.bump(variable);
            }
        }
        for &variable in &analyzed {
            self.seen[variable as usize] = false;
        }
        analyzed.clear();
        self.analyzed = analyzed;
        (backjump_level, lbd)
    }
    // EVSIDS: crece el incremento en vez de decaer todas las puntuaciones.
    fn bump_score(&mut self, variable: u32) {
        self.scores[variable as usize] += self.increment;
        if self.scores[variable as usize] > MAX_ACTIVITY {
            self.rescale();
        }
        if self.heap.contains(variable) {
            self.heap.increase(variable, &self.scores);
        }
    }
    fn rescale(&mut self) {
        for score in &mut self.scores {
            *score /= MAX_ACTIVITY;
        }
        self.increment /= MAX_ACTIVITY;
        // El redondeo puede crear empates: se restaura el orden completo.
        self.heap.heapify(&self.scores);
    }
    fn learn(&mut self, lbd: u32) {
        let learned = std::mem::take(&mut self.learned);
        let first = learned[0];
        match learned.len() {
            1 => self.assign_root(first),
            2 => {
                self.watch_binary(first, learned[1]);
                self.assign(first, BINARY | learned[1]);
            }
            _ => {
                let reference = self.add_long(&learned, true, lbd);
                self.assign(first, reference);
            }
        }
        self.learned = learned;
        self.stats.learned_clauses += 1;
        self.conflicts += 1;
        self.fast_lbd.update(f64::from(lbd));
        self.slow_lbd.update(f64::from(lbd));
    }
    fn backjump(&mut self, level: usize) {
        if self.level() <= level {
            return;
        }
        let mark = self.trail_limits[level];
        for index in (mark..self.trail.len()).rev() {
            let literal = self.trail[index];
            self.values[literal as usize] = 0;
            self.values[(literal ^ 1) as usize] = 0;
            self.queue.unassign(literal >> 1);
            if self.stable {
                self.heap.insert(literal >> 1, &self.scores);
            }
        }
        self.trail.truncate(mark);
        self.trail_limits.truncate(level);
        self.head = mark;
    }
    fn restart_due(&self) -> bool {
        self.conflicts >= self.next_mode
            || (self.conflicts >= self.next_restart
                && (self.stable || self.fast_lbd.value > RESTART_MARGIN * self.slow_lbd.value))
    }
    // Modo enfocado: VMTF y restarts tipo Glucose. Modo estable: EVSIDS y restarts Luby.
    fn restart(&mut self, base: usize) {
        self.backjump(base);
        self.stats.restarts += 1;
        if self.conflicts >= self.next_mode {
            self.stable = !self.stable;
            self.mode_span = self.mode_span.saturating_mul(2);
            self.next_mode = self.conflicts.saturating_add(self.mode_span);
            if self.stable {
                for variable in 0..self.levels.len() as u32 {
                    if self.values[(variable << 1) as usize] == 0 {
                        self.heap.insert(variable, &self.scores);
                    }
                }
            }
        }
        self.next_restart = self.conflicts + if self.stable {
            let (u, v) = self.luby;
            self.luby = if u & u.wrapping_neg() == v { (u + 1, 1) } else { (u, 2 * v) };
            LUBY_UNIT * v
        } else {
            RESTART_INTERVAL
        };
    }
    fn decide(&mut self) -> Option<u32> {
        let variable = if self.stable {
            loop {
                let variable = self.heap.pop(&self.scores)?;
                if self.values[(variable << 1) as usize] == 0 {
                    break variable;
                }
            }
        } else {
            let mut variable = self.queue.search;
            while variable != NONE && self.values[(variable << 1) as usize] != 0 {
                variable = self.queue.previous[variable as usize];
            }
            self.queue.search = variable;
            if variable == NONE {
                return None;
            }
            variable
        };
        Some((variable << 1) | u32::from(self.phases[variable as usize]))
    }
    // El oráculo solo lee las unidades de la raíz y solo escribe fases:
    // nunca toca valores, antecedentes, watches ni el trail.
    fn reignite(&mut self, configuration: ReignitionConfig, cancel: Option<&AtomicBool>) -> bool {
        let Some(mut walker) = self.walker.take() else { return false; };
        let started = Instant::now();
        let root = self.trail_limits.first().copied().unwrap_or(self.trail.len());
        if walker.synced < root {
            for &literal in &self.trail[walker.synced..root] {
                walker.fix(literal);
            }
            walker.synced = root;
            walker.settle();
        }
        let mut budget = configuration.oracle.flips_per_variable.saturating_mul(self.phases.len());
        if self.ignited {
            let earned = (self.propagated - self.reignition_mark) / ORACLE_EFFORT;
            budget = budget.min(usize::try_from(earned).unwrap_or(usize::MAX));
        }
        self.reignition_mark = self.propagated;
        let flips = walker.walk(budget, cancel);
        // Transferencia: la mejor asignación del oráculo sobrescribe las polaridades guardadas.
        for (phase, &value) in self.phases.iter_mut().zip(&walker.best_values) {
            *phase = u8::from(!value);
        }
        let solved = walker.best == 0;
        self.walker = Some(walker);
        self.ignited = true;
        self.next_reignition = self.conflicts.saturating_add(configuration.interval as u64);
        self.stats.reignitions += 1;
        self.stats.reignition_flips += flips;
        self.stats.reignition_ms += started.elapsed().as_secs_f64() * 1000.0;
        solved
    }
    fn oracle_model(&self, variables: &[u32]) -> Vec<i32> {
        let walker = self.walker.as_ref().expect("Only the oracle reports its own model");
        signed_model(variables, walker.best_values.iter().copied())
    }
    // Candidatas: aprendidas con LBD > 2 que no son razón ni se usaron desde la última reducción.
    fn reduce_database(&mut self, mut proof: Option<(&mut (dyn ProofLog + '_), &[u32])>) {
        debug_assert_eq!(self.head, self.trail.len());
        // Intervalo aritmético: la base retenida crece con la raíz del número de conflictos.
        self.cleanup_interval += CLEANUP_INTERVAL;
        self.next_cleanup = self.conflicts + self.cleanup_interval;
        // Las razones vigentes son largas sin el bit BINARY (NO_REASON lo tiene).
        for &literal in &self.trail {
            let reason = self.reasons[variable_of(literal)];
            if reason & BINARY == 0 {
                self.arena[reason as usize + 1] |= LOCKED;
            }
        }
        let mut candidates = Vec::new();
        let mut reference = 0;
        while reference < self.arena.len() {
            let meta = self.arena[reference + 1];
            if meta & (LEARNED | LOCKED | USED) == LEARNED && meta & LBD_MASK > 2 {
                candidates.push(reference);
            }
            self.arena[reference + 1] = meta & !(LOCKED | USED);
            reference += HEADER + self.arena[reference] as usize;
        }
        if candidates.is_empty() {
            return;
        }
        // Se sacrifican primero el mayor LBD y, a igualdad, la cláusula más larga.
        candidates.sort_unstable_by(|&a, &b| {
            (self.arena[b + 1] & LBD_MASK).cmp(&(self.arena[a + 1] & LBD_MASK))
                .then_with(|| self.arena[b].cmp(&self.arena[a]))
                .then_with(|| a.cmp(&b))
        });
        let remove_count = (candidates.len() / 2).max(1);
        let mut line: Vec<i32> = Vec::new();
        for &reference in &candidates[..remove_count] {
            self.arena[reference + 1] |= GARBAGE;
            // Una razón vigente nunca es candidata: lo borrado no sostiene ninguna unidad.
            if let Some((log, variables)) = proof.as_mut() {
                let start = reference + HEADER;
                external(variables, &self.arena[start..start + self.arena[reference] as usize], &mut line);
                log.delete(&line);
            }
        }
        // La arena vieja guarda la nueva referencia de cada superviviente en su palabra de flags.
        let mut old = std::mem::take(&mut self.arena);
        let mut arena = Vec::with_capacity(old.len());
        let mut reference = 0;
        while reference < old.len() {
            let next = reference + HEADER + old[reference] as usize;
            if old[reference + 1] & GARBAGE == 0 {
                let moved = arena.len() as u32;
                arena.extend_from_slice(&old[reference..next]);
                old[reference + 1] = moved;
            }
            reference = next;
        }
        for &literal in &self.trail {
            let variable = variable_of(literal);
            let reason = self.reasons[variable];
            if reason & BINARY == 0 {
                self.reasons[variable] = old[reason as usize + 1];
            }
        }
        for watchers in &mut self.watches {
            watchers.retain(|watcher| watcher.reference & BINARY != 0);
        }
        let mut reference = 0;
        while reference < arena.len() {
            let (first, second) = (arena[reference + HEADER], arena[reference + HEADER + 1]);
            self.watches[first as usize].push(Watcher { blocker: second, reference: reference as u32 });
            self.watches[second as usize].push(Watcher { blocker: first, reference: reference as u32 });
            reference += HEADER + arena[reference] as usize;
        }
        self.arena = arena;
        self.stats.deleted_clauses += remove_count;
        self.stats.database_reductions += 1;
    }
}
fn active(config: Option<ReignitionConfig>) -> Option<ReignitionConfig> {
    config.filter(|configuration| {
        configuration.interval != 0 && configuration.oracle.flips_per_variable != 0
    })
}
fn prepare_formula(formula: &[Vec<i32>], oracle: Option<OracleConfig>) -> Option<(State, Vec<u32>)> {
    if formula.iter().any(Vec::is_empty) {
        return None;
    }
    let mut variables: Vec<u32> = formula.iter().flatten().map(|&literal| {
        assert!(literal != 0 && literal != i32::MIN, "Invalid SAT literal");
        literal.unsigned_abs()
    }).collect();
    variables.sort_unstable();
    variables.dedup();
    // Literales y referencias comparten 31 bits; NO_REASON queda fuera de ambos rangos.
    assert!(variables.len() < 1 << 30, "Too many variables");
    let mut state = State::new(variables.len(), SolveStats::default());
    let mut packed: Vec<u32> = Vec::new();
    // Copia plana para el oráculo; las unidades le llegan después por el trail raíz.
    let mut walk_literals = Vec::new();
    let mut walk_offsets = vec![0_u32];
    for clause in formula {
        packed.clear();
        packed.extend(clause.iter().map(|&literal| {
            let variable = variables.binary_search(&literal.unsigned_abs())
                .expect("Variable was indexed") as u32;
            (variable << 1) | u32::from(literal < 0)
        }));
        packed.sort_unstable();
        packed.dedup();
        // Tras ordenar, x y ¬x quedan adyacentes: la tautología no aporta nada.
        if packed.windows(2).any(|pair| pair[0] ^ 1 == pair[1]) {
            continue;
        }
        for &literal in &packed {
            state.scores[variable_of(literal)] += 1.0;
        }
        state.add_original(&packed);
        if oracle.is_some() && packed.len() > 1 {
            walk_literals.extend_from_slice(&packed);
            walk_offsets.push(u32::try_from(walk_literals.len())
                .expect("Oracle arena exceeds 2^32 literals"));
        }
    }
    if state.root_unsat {
        state.stats.conflicts = 1;
    } else if let Some(oracle) = oracle {
        state.walker = Some(Walker::new(variables.len(), walk_literals, walk_offsets, oracle.seed));
    }
    // Orden VMTF inicial: al frente la mayor puntuación y, a igualdad, el menor índice.
    let mut order: Vec<u32> = (0..variables.len() as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        state.scores[a as usize].total_cmp(&state.scores[b as usize]).then_with(|| b.cmp(&a))
    });
    for variable in order {
        state.queue.push(variable);
    }
    state.queue.search = state.queue.last;
    Some((state, variables))
}
// Literales internos a identificadores DIMACS, para las líneas de la traza.
fn external(variables: &[u32], clause: &[u32], line: &mut Vec<i32>) {
    line.clear();
    line.extend(clause.iter().map(|&literal| {
        let id = variables[(literal >> 1) as usize] as i32;
        if literal & 1 == 0 { id } else { -id }
    }));
}
pub fn solve_cdcl(formula: &[Vec<i32>]) -> (Option<Vec<i32>>, SolveStats) {
    solve_cdcl_inner(formula, None, None)
}
// Igual que `solve_cdcl_with_reignition`, registrando cada cláusula aprendida o borrada.
// Toda cláusula aprendida es RUP; un resultado UNSAT termina con la cláusula vacía.
pub fn solve_cdcl_with_proof(
    formula: &[Vec<i32>],
    config: ReignitionConfig,
    proof: &mut dyn ProofLog,
) -> (Option<Vec<i32>>, SolveStats) {
    solve_cdcl_inner(formula, Some(config), Some(proof))
}
pub fn solve_cdcl_with_reignition(
    formula: &[Vec<i32>],
    config: ReignitionConfig,
) -> (Option<Vec<i32>>, SolveStats) {
    solve_cdcl_inner(formula, Some(config), None)
}
fn solve_cdcl_inner(
    formula: &[Vec<i32>],
    config: Option<ReignitionConfig>,
    proof: Option<&mut dyn ProofLog>,
) -> (Option<Vec<i32>>, SolveStats) {
    let feedback = active(config);
    let prepared = prepare_formula(formula, feedback.map(|configuration| configuration.oracle));
    let Some((mut state, variables)) = prepared else {
        return (None, SolveStats { conflicts: 1, ..SolveStats::default() });
    };
    let control = SearchControl { cube: &[], cancel: None, sharing: None, proof, progress: None };
    let (outcome, stats) = run_state(&mut state, &variables, feedback, control);
    match outcome {
        SearchOutcome::Sat(model) => (Some(model), stats),
        SearchOutcome::Unsat => (None, stats),
        SearchOutcome::Cancelled => unreachable!("Serial execution cannot be cancelled"),
    }
}
fn signed_model(variables: &[u32], values: impl Iterator<Item = bool>) -> Vec<i32> {
    variables.iter().zip(values).map(|(&id, value)| {
        if value { id as i32 } else { -(id as i32) }
    }).collect()
}
fn extract_model(state: &State, variables: &[u32]) -> Vec<i32> {
    signed_model(variables, state.values.iter().step_by(2).map(|&value| value == 1))
}
fn run_state(
    state: &mut State,
    variables: &[u32],
    feedback: Option<ReignitionConfig>,
    mut control: SearchControl<'_, '_>,
) -> (SearchOutcome, SolveStats) {
    let outcome = search(state, variables, feedback, &mut control);
    if let Some(beacon) = control.progress.as_mut() {
        // Lo que no llenó un lote sale al soltar el cubo: el total queda exacto.
        beacon.publish(state.stats.conflicts, true);
    }
    if let Some(link) = control.sharing {
        // Lo aprendido en este cubo se publica entero antes de soltar el estado.
        link.flush(&mut state.stats, true);
    }
    (outcome, state.stats)
}
fn search(
    state: &mut State,
    variables: &[u32],
    feedback: Option<ReignitionConfig>,
    control: &mut SearchControl<'_, '_>,
) -> SearchOutcome {
    // La cancelación se consulta una vez por conflicto o decisión, nunca dentro del BCP.
    let cancel = control.cancel;
    let cancelled = || cancel.is_some_and(|flag| flag.load(Ordering::Acquire));
    if cancelled() {
        return SearchOutcome::Cancelled;
    }
    // El certificado raiz evita reprocesar una cola contradictoria y cubos posteriores.
    if state.root_unsat {
        // Dos unidades opuestas de la fórmula: la cláusula vacía es RUP.
        if let Some(proof) = control.proof.as_deref_mut() {
            proof.add(&[]);
        }
        return SearchOutcome::Unsat;
    }
    let mut line: Vec<i32> = Vec::new();
    if let Some(link) = control.sharing.as_deref_mut() {
        // Un cubo empieza en el nivel 0: mismo punto de sincronización que un restart.
        if state.level() == 0 {
            link.exchange(state.conflicts, &mut state.stats);
            state.absorb(&mut link.inbox);
        }
    }
    loop {
        if cancelled() {
            return SearchOutcome::Cancelled;
        }
        if state.root_unsat {
            // Una importación contradice la raíz: consecuencia global, UNSAT.
            state.stats.conflicts += 1;
            return SearchOutcome::Unsat;
        }
        if let Some(conflict) = state.propagate() {
            state.stats.conflicts += 1;
            if let Some(beacon) = control.progress.as_mut() {
                beacon.publish(state.stats.conflicts, false);
            }
            let level = state.level();
            if level == 0 {
                // No se publica la cláusula vacía ni se confunde cancelación con UNSAT.
                state.root_unsat = true;
                if let Some(proof) = control.proof.as_deref_mut() {
                    proof.add(&[]);
                }
                return SearchOutcome::Unsat;
            }
            let (backjump_level, lbd) = state.analyze(conflict);
            if let Some(proof) = control.proof.as_deref_mut() {
                external(variables, &state.learned, &mut line);
                proof.add(&line);
            }
            if lbd <= EXPORT_LBD {
                if let Some(link) = control.sharing.as_deref_mut() {
                    // Los cubos son decisiones: toda cláusula aprendida es consecuencia global.
                    link.export(&state.learned, &mut state.stats);
                }
            }
            state.stats.backjumps += 1;
            state.stats.nonchronological_backjumps += usize::from(level > backjump_level + 1);
            state.backjump(backjump_level);
            state.learn(lbd);
            continue;
        }
        if state.conflicts >= state.next_cleanup {
            state.reduce_database(control.proof.as_deref_mut().map(|proof| (proof, variables)));
        }
        if let Some(configuration) = feedback {
            // Ignición inicial: primer fixpoint de la búsqueda, todavía en el nivel 0.
            if !state.ignited && state.reignite(configuration, cancel) {
                return SearchOutcome::Sat(state.oracle_model(variables));
            }
        }
        if state.level() < control.cube.len() {
            let literal = control.cube[state.level()];
            let value = state.value(literal);
            if value == -1 { return SearchOutcome::Unsat; }
            // Un nivel por supuesto; niveles vacíos solo se crean en un fixpoint.
            state.trail_limits.push(state.trail.len());
            if value == 0 {
                state.stats.decisions += 1;
                state.stats.cube_decisions += 1;
                state.assign(literal, NO_REASON);
            }
            // Cubos como DECISIONES: sus literales permanecen en las cláusulas 1UIP.
            continue;
        }
        if state.restart_due() {
            // El restart conserva los supuestos del cubo salvo que haya algo que importar:
            // importar exige el nivel 0 y allí nunca fuerza un backjump.
            let mut base = control.cube.len();
            if let Some(link) = control.sharing.as_deref_mut() {
                if state.conflicts - link.last_sync >= link.interval {
                    link.exchange(state.conflicts, &mut state.stats);
                    if !link.inbox.is_empty() {
                        base = 0;
                    }
                }
            }
            state.restart(base);
            if let Some(link) = control.sharing.as_deref_mut() {
                if !link.inbox.is_empty() {
                    state.absorb(&mut link.inbox);
                }
            }
            if let Some(configuration) = feedback {
                // Re-ignición: solo aquí, con el trail recién limpiado por el restart.
                if state.conflicts >= state.next_reignition && !state.root_unsat
                    && state.reignite(configuration, cancel)
                {
                    return SearchOutcome::Sat(state.oracle_model(variables));
                }
            }
            continue;
        }
        let Some(literal) = state.decide() else {
            return SearchOutcome::Sat(extract_model(state, variables));
        };
        state.trail_limits.push(state.trail.len());
        state.stats.decisions += 1;
        state.assign(literal, NO_REASON);
    }
}
fn make_cubes(base: &State, config: SwarmConfig) -> Vec<Vec<u32>> {
    let target = config.workers.max(1).saturating_mul(config.cubes_per_worker.max(1)).clamp(1, 65_536);
    let mut candidates: Vec<usize> = (0..base.levels.len())
        .filter(|&variable| base.values[variable << 1] == 0).collect();
    candidates.sort_unstable_by(|&a, &b| {
        base.scores[b].total_cmp(&base.scores[a]).then_with(|| a.cmp(&b))
    });
    let mut depth = 0;
    while depth < candidates.len().min(16) && (1 << depth) < target { depth += 1; }
    candidates.truncate(depth);
    // Todas las combinaciones: las fases ordenan los cubos y nunca descartan ninguno.
    (0..1usize << depth).map(|mask| candidates.iter().enumerate().map(|(index, &variable)| {
        ((variable as u32) << 1) | (u32::from(base.phases[variable]) ^ ((mask >> index) & 1) as u32)
    }).collect()).collect()
}
#[derive(Default)]
struct WorkerReport {
    started: usize,
    unsat: usize,
    cancelled: usize,
    model: Option<Vec<i32>>,
    total: SolveStats,
}
fn swarm_worker(
    base: &State,
    variables: &[u32],
    cubes: &[Vec<u32>],
    next: &AtomicUsize,
    cancel: &AtomicBool,
    pool: &CloudPool,
    config: SwarmConfig,
    owner: u32,
    progress: Option<&SwarmProgress>,
) -> WorkerReport {
    let mut report = WorkerReport::default();
    let mut state = base.clone();
    // Cada worker continúa la caminata del estado base con su propio azar.
    if let Some(walker) = state.walker.as_mut() {
        walker.reseed(u64::from(owner) + 1);
    }
    let feedback = active(Some(config.reignition));
    // El contexto de intercambio vive lo que el worker, no lo que un cubo.
    let mut link = (config.share_enabled && config.share_capacity != 0)
        .then(|| Sharing::new(pool, owner, config.share_interval.max(1) as u64));
    loop {
        if cancel.load(Ordering::Acquire) { break; }
        let index = next.fetch_add(1, Ordering::Relaxed);
        let Some(cube) = cubes.get(index) else { break; };
        report.started += 1;
        if !state.root_unsat {
            // Solo se llega al siguiente cubo tras CubeUNSAT en un fixpoint.
            debug_assert_eq!(state.head, state.trail.len());
            state.backjump(0);
        }
        // Clausulas, unidades raiz, fases, actividades y oraculo sobreviven al cubo.
        state.stats = SolveStats::default();
        let control = SearchControl {
            cube, cancel: Some(cancel), sharing: link.as_mut(), proof: None,
            progress: progress.map(|progress| Beacon { progress, published: 0 }),
        };
        let (outcome, stats) = run_state(&mut state, variables, feedback, control);
        report.total.add_search(stats);
        match outcome {
            SearchOutcome::Sat(model) => {
                if cancel.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                    report.model = Some(model);
                } else { report.cancelled += 1; }
                break;
            }
            SearchOutcome::Unsat => {
                report.unsat += 1;
                if let Some(progress) = progress {
                    progress.cubes_closed.fetch_add(1, Ordering::Relaxed);
                }
            }
            SearchOutcome::Cancelled => { report.cancelled += 1; break; }
        }
    }
    report
}
pub fn solve_swarm(formula: &[Vec<i32>], config: SwarmConfig) -> (Option<Vec<i32>>, SwarmStats) {
    solve_swarm_with_progress(formula, config, None)
}
// El mismo enjambre, publicando su avance en `progress` si alguien lo mira. Con observador o
// sin él la búsqueda hace exactamente lo mismo.
pub fn solve_swarm_with_progress(
    formula: &[Vec<i32>], config: SwarmConfig, progress: Option<&SwarmProgress>,
) -> (Option<Vec<i32>>, SwarmStats) {
    solve_swarm_with_filter(formula, config, progress, None)
}
// El mismo enjambre, con un prefiltro que ve los cubos antes que los workers. Sin él, todos
// los cubos van a los workers, como siempre.
pub fn solve_swarm_with_filter(
    formula: &[Vec<i32>], config: SwarmConfig, progress: Option<&SwarmProgress>, filter: Option<CubeFilter<'_>>,
) -> (Option<Vec<i32>>, SwarmStats) {
    let mut stats = SwarmStats::default();
    let feedback = active(Some(config.reignition));
    let prepared = prepare_formula(formula, feedback.map(|configuration| configuration.oracle));
    let Some((mut base, variables)) = prepared else {
        stats.total.conflicts = 1;
        return (None, stats);
    };
    if !base.root_unsat && base.propagate().is_some() {
        base.root_unsat = true;
        base.stats.conflicts += 1;
    }
    if base.root_unsat {
        stats.total = base.stats;
        return (None, stats);
    }
    if base.trail.len() == variables.len() {
        stats.total = base.stats;
        return (Some(extract_model(&base, &variables)), stats);
    }
    if let Some(configuration) = feedback {
        // Ignición única antes de cubicar: las fases del oráculo ordenan los cubos.
        if base.reignite(configuration, None) {
            stats.total = base.stats;
            return (Some(base.oracle_model(&variables)), stats);
        }
    }
    stats.total = base.stats;
    let mut cubes = make_cubes(&base, config);
    stats.cubes_generated = cubes.len();
    if let Some(progress) = progress {
        progress.cubes_total.store(cubes.len() as u64, Ordering::Relaxed);
    }
    if let Some(filter) = filter {
        // El filtro ve los cubos en literales DIMACS. Si no contesta a todos, no ha
        // contestado a ninguno.
        let mut line = Vec::new();
        let asked: Vec<Vec<i32>> = cubes.iter().map(|cube| {
            external(&variables, cube, &mut line);
            line.clone()
        }).collect();
        let verdicts = filter(&asked);
        if verdicts.len() == cubes.len() {
            let mut open = Vec::with_capacity(cubes.len());
            for (cube, verdict) in cubes.into_iter().zip(verdicts) {
                match verdict {
                    // El modelo es la respuesta: no hace falta ningún worker.
                    CubeVerdict::Sat(model) => return (Some(model), stats),
                    CubeVerdict::Unsat => stats.cubes_filtered += 1,
                    CubeVerdict::Unknown => open.push(cube),
                }
            }
            cubes = open;
            if let Some(progress) = progress {
                progress.cubes_filtered.store(stats.cubes_filtered as u64, Ordering::Relaxed);
            }
        }
    }
    let workers = config.workers.max(1).min(cubes.len());
    let next = AtomicUsize::new(0);
    let cancel = AtomicBool::new(false);
    let pool = CloudPool::new(config.share_capacity);
    let (spawned, reports) = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for owner in 0..workers {
            if cancel.load(Ordering::Acquire) { break; }
            let (base, variables, cubes, next, cancel, pool) =
                (&base, variables.as_slice(), cubes.as_slice(), &next, &cancel, &pool);
            let worker = move || swarm_worker(base, variables, cubes,
                next, cancel, pool, config, owner as u32, progress);
            match std::thread::Builder::new().name(format!("swarm-{owner}"))
                .spawn_scoped(scope, worker) {
                Ok(handle) => handles.push(handle),
                Err(_) => break,
            }
        }
        if handles.is_empty() {
            return (0, vec![swarm_worker(&base, &variables, &cubes,
                &next, &cancel, &pool, config, 0, progress)]);
        }
        let spawned = handles.len();
        let reports = handles.into_iter().map(|handle| handle.join().expect("SAT worker panicked"))
            .collect::<Vec<_>>();
        (spawned, reports)
    });
    stats.workers = spawned;
    let mut winner = None;
    for report in reports {
        stats.cubes_started += report.started;
        stats.cubes_unsat += report.unsat;
        stats.cubes_cancelled += report.cancelled;
        stats.total.add_search(report.total);
        if report.model.is_some() { winner = report.model; }
    }
    stats.shared_clauses = pool.snapshot(&variables);
    if winner.is_none() {
        assert_eq!(
            stats.cubes_unsat + stats.cubes_filtered, stats.cubes_generated,
            "Incomplete cube coverage cannot prove UNSAT"
        );
    }
    (winner, stats)
}

#[cfg(test)]
mod deterministic_import_tests {
    use super::*;

    #[test]
    fn beacon_publishes_whole_batches_and_the_remainder_only_on_flush() {
        let progress = SwarmProgress::default();
        let mut beacon = Beacon { progress: &progress, published: 0 };
        let read = || progress.conflicts.load(Ordering::Relaxed);
        for conflicts in 1..PROGRESS_BATCH {
            beacon.publish(conflicts, false);
        }
        assert_eq!(read(), 0);
        beacon.publish(PROGRESS_BATCH, false);
        assert_eq!(read(), PROGRESS_BATCH as u64);
        beacon.publish(PROGRESS_BATCH + 5, false);
        assert_eq!(read(), PROGRESS_BATCH as u64);
        beacon.publish(PROGRESS_BATCH + 5, true);
        assert_eq!(read(), PROGRESS_BATCH as u64 + 5);
        // Nada pendiente: vaciar otra vez no suma.
        beacon.publish(PROGRESS_BATCH + 5, true);
        assert_eq!(read(), PROGRESS_BATCH as u64 + 5);
    }

    #[test]
    fn published_consequence_is_imported_and_promoted_to_root() {
        // (x1 OR x2) AND (x1 OR !x2) entails x1 without a syntactic unit.
        let formula = vec![vec![1, 2], vec![1, -2]];
        let (mut state, variables) = prepare_formula(&formula, None).unwrap();
        let positive_x1 = 0;

        let pool = CloudPool::new(8);
        let mut peer = Sharing::new(&pool, 1, 1);
        let mut peer_stats = SolveStats::default();
        peer.export(&[positive_x1], &mut peer_stats);
        peer.flush(&mut peer_stats, true);
        assert_eq!(peer_stats.exported_clauses, 1);
        let snapshot = pool.snapshot(&variables);
        assert_eq!(snapshot, vec![vec![1]]);
        for bits in 0..4 {
            let literal_true = |literal: i32| {
                (bits & (1 << (literal.unsigned_abs() - 1)) != 0) == (literal > 0)
            };
            if formula.iter().all(|clause| clause.iter().copied().any(literal_true)) {
                assert!(snapshot[0].iter().copied().any(literal_true));
            }
        }

        let mut link = Sharing::new(&pool, 0, 1);
        let control = SearchControl { cube: &[], cancel: None, sharing: Some(&mut link), proof: None, progress: None };
        let (outcome, stats) = run_state(&mut state, &variables, None, control);
        let model = match outcome {
            SearchOutcome::Sat(model) => model,
            _ => panic!("Importing a valid consequence must preserve SAT"),
        };
        assert!(model.contains(&1));
        assert_eq!(stats.imported_clauses, 1);
        assert_eq!(stats.import_backjumps, 0);
        assert_eq!(state.value(positive_x1), 1);
        assert_eq!(state.levels[0], 0);
        // La unidad importada vive en el trail raíz: no necesita cláusula antecedente.
        assert_eq!(state.reasons[0], NO_REASON);

        // Siguiente cubo del mismo worker: cursor y filtro sobreviven, nada se reimporta.
        state.backjump(0);
        state.stats = SolveStats::default();
        let control = SearchControl { cube: &[], cancel: None, sharing: Some(&mut link), proof: None, progress: None };
        let (_, stats) = run_state(&mut state, &variables, None, control);
        assert_eq!(stats.imported_clauses, 0);
        // Un lote nunca vuelve a su autor.
        peer.exchange(0, &mut peer_stats);
        assert!(peer.inbox.is_empty());
    }

    #[test]
    fn ring_keeps_the_newest_clauses_and_lagging_readers_skip_ahead() {
        let pool = CloudPool::new(2);
        let mut writer = Sharing::new(&pool, 0, 1);
        let mut stats = SolveStats::default();
        // Dos cláusulas llenan el anillo y salen como un lote; la tercera lo expulsa entero.
        for literal in [0, 2, 4] {
            writer.export(&[literal, literal + 11], &mut stats);
        }
        writer.flush(&mut stats, true);
        assert_eq!(stats.exported_clauses, 3);
        let variables: Vec<u32> = (1..=10).collect();
        assert_eq!(pool.snapshot(&variables), vec![vec![3, -8]]);

        let mut reader = Sharing::new(&pool, 1, 1);
        reader.exchange(0, &mut stats);
        let seen: Vec<Vec<u32>> = clauses_of(&reader.inbox).map(<[u32]>::to_vec).collect();
        assert_eq!(seen, vec![vec![4, 15]]);
        // El filtro reconoce la misma cláusula en cualquier orden de literales.
        assert!(!reader.remember(&[15, 4]));
    }

    fn violations(walker: &Walker, values: &[bool]) -> usize {
        (0..walker.offsets.len() - 1).filter(|&clause| {
            let range = walker.offsets[clause] as usize..walker.offsets[clause + 1] as usize;
            !walker.literals[range].iter().any(|&literal| values[variable_of(literal)] == (literal & 1 == 0))
        }).count()
    }

    #[test]
    fn walker_counters_survive_fixing_and_walking() {
        let variables = 12_usize;
        let mut seed = 0x9e37_79b9_u64;
        let mut literals = Vec::new();
        let mut offsets = vec![0_u32];
        for _ in 0..60 {
            let mut clause: Vec<u32> = (0..3).map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed % (2 * variables as u64)) as u32
            }).collect();
            clause.sort_unstable();
            clause.dedup();
            if clause.len() < 2 || clause.windows(2).any(|pair| pair[0] ^ 1 == pair[1]) {
                continue;
            }
            literals.extend_from_slice(&clause);
            offsets.push(literals.len() as u32);
        }
        let mut walker = Walker::new(variables, literals, offsets, 7);
        // x3 verdadero y x6 falso quedan fijados como unidades de la raíz.
        walker.fix(4);
        walker.fix(11);
        walker.settle();
        let start = walker.best;
        assert!(walker.walk(400, None) <= 400);
        assert!(walker.best <= start);
        assert!(walker.values[2] && !walker.values[5]);
        assert!(walker.best_values[2] && !walker.best_values[5]);
        // Recuento desde cero: verdaderos, críticos, roturas y lista de violadas.
        let mut breaks = vec![0_u32; variables];
        let mut violated = Vec::new();
        for clause in 0..walker.offsets.len() - 1 {
            let range = walker.offsets[clause] as usize..walker.offsets[clause + 1] as usize;
            let truths: Vec<u32> = walker.literals[range].iter().copied()
                .filter(|&literal| walker.values[variable_of(literal)] == (literal & 1 == 0)).collect();
            assert_eq!(walker.satisfied[clause] as usize, truths.len());
            assert_eq!(walker.critical[clause], truths.iter().fold(0, |xor, &literal| xor ^ literal));
            match truths[..] {
                [] => violated.push(clause as u32),
                [only] => breaks[variable_of(only)] += 1,
                _ => {}
            }
        }
        assert_eq!(walker.breaks, breaks);
        let mut listed = walker.unsat.clone();
        listed.sort_unstable();
        assert_eq!(listed, violated);
        // La asignación guardada es exactamente tan buena como dice `best`.
        assert_eq!(violations(&walker, &walker.best_values), walker.best);
    }

    #[test]
    fn oracle_model_ends_the_search_before_any_decision() {
        let formula: Vec<Vec<i32>> = (1..=30).step_by(3).map(|v| vec![v, v + 1, v + 2]).collect();
        let config = ReignitionConfig {
            interval: 1,
            oracle: OracleConfig { flips_per_variable: 8, seed: 5 },
        };
        let (model, stats) = solve_cdcl_with_reignition(&formula, config);
        let model = model.expect("Disjoint clauses are satisfiable");
        assert!(formula.iter().all(|clause| clause.iter().any(|literal| model.contains(literal))));
        assert_eq!((stats.decisions, stats.conflicts, stats.reignitions), (0, 0, 1));
        assert!(stats.reignition_flips <= 8 * 30);
    }
}
