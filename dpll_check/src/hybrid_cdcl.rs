use std::collections::{HashSet, VecDeque};
use super::gradient::{GradientConfig, OracleWorkspace};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
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
    pub deleted_clauses: usize,
    pub database_reductions: usize,
    pub symmetry_rows: usize,
    pub symmetry_columns: usize,
    pub symmetry_clauses: usize,
    pub reignitions: usize,
    pub reignition_epochs: usize,
    pub reignition_ms: f64,
    pub exported_clauses: usize,
    pub imported_clauses: usize,
    pub dropped_exports: usize,
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
        self.deleted_clauses += other.deleted_clauses;
        self.database_reductions += other.database_reductions;
        self.reignitions += other.reignitions;
        self.reignition_epochs += other.reignition_epochs;
        self.reignition_ms += other.reignition_ms;
        self.exported_clauses += other.exported_clauses;
        self.imported_clauses += other.imported_clauses;
        self.dropped_exports += other.dropped_exports;
        self.import_backjumps += other.import_backjumps;
    }
}
#[derive(Clone, Copy, Debug)]
pub struct ReignitionConfig {
    pub interval: usize,
    pub oracle: GradientConfig,
}
impl Default for ReignitionConfig {
    fn default() -> Self {
        Self {
            interval: 500,
            oracle: GradientConfig { max_epochs: 100, restarts: 1, ..GradientConfig::default() },
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct SwarmConfig {
    pub workers: usize,
    pub cubes_per_worker: usize,
    pub share_capacity: usize,
    pub share_interval: usize,
    pub reignition: ReignitionConfig,
    pub activity_boost: f64,
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
            activity_boost: 4.0,
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
    pub total: SolveStats,
    // Snapshot auditable: IDs originales; nunca contiene cláusulas vacías.
    pub shared_clauses: Vec<Vec<i32>>,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SmallClause {
    literals: [usize; 3],
    len: usize,
}
impl SmallClause {
    fn canonical(clause: &[usize]) -> Option<Self> {
        if clause.is_empty() || clause.len() > 3 { return None; }
        let mut literals = [0; 3];
        literals[..clause.len()].copy_from_slice(clause);
        literals[..clause.len()].sort_unstable();
        let mut len = 0;
        for index in 0..clause.len() {
            let literal = literals[index];
            if len != 0 && literals[len - 1] == literal { continue; }
            if len != 0 && literals[len - 1] == (literal ^ 1) { return None; }
            literals[len] = literal;
            len += 1;
        }
        literals[len..].fill(0);
        Some(Self { literals, len })
    }
    fn as_slice(&self) -> &[usize] { &self.literals[..self.len] }
}
#[derive(Clone, Copy)]
struct SharedClause { clause: SmallClause }
struct CloudLog { clauses: Vec<SharedClause>, known: HashSet<SmallClause> }
pub struct CloudPool { log: Mutex<CloudLog>, capacity: usize }
impl CloudPool {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.min(8192);
        Self { log: Mutex::new(CloudLog {
            clauses: Vec::with_capacity(capacity), known: HashSet::with_capacity(capacity),
        }), capacity }
    }
    fn publish(&self, clause: SmallClause) -> bool {
        let Ok(mut log) = self.log.try_lock() else { return false; };
        if log.clauses.len() == self.capacity || !log.known.insert(clause) { return false; }
        log.clauses.push(SharedClause { clause });
        true
    }
    fn harvest(&self, cursor: &mut usize) -> Vec<SharedClause> {
        let Ok(log) = self.log.try_lock() else { return Vec::new(); };
        let clauses = log.clauses[*cursor..].to_vec();
        *cursor = log.clauses.len();
        clauses
    }
    fn snapshot(&self, variables: &[u32]) -> Vec<Vec<i32>> {
        let Ok(log) = self.log.try_lock() else { return Vec::new(); };
        log.clauses.iter().map(|entry| entry.clause.as_slice().iter().map(|&literal| {
            let variable = variables[literal >> 1] as i32;
            if literal & 1 == 0 { variable } else { -variable }
        }).collect()).collect()
    }
}
enum SearchOutcome { Sat(Vec<i32>), Unsat, Cancelled }
enum Propagation { Stable, Conflict(usize), Cancelled }
#[derive(Clone, Copy)]
struct Sharing<'a> { pool: &'a CloudPool, interval: usize }
struct SearchControl<'a> {
    cube: &'a [usize],
    cancel: Option<&'a AtomicBool>,
    sharing: Option<Sharing<'a>>,
}
#[inline]
fn poll_cancel(cancel: Option<&AtomicBool>, work: &mut usize) -> bool {
    let Some(cancel) = cancel else { return false; };
    *work += 1;
    if *work < 256 { return false; }
    *work = 0;
    cancel.load(Ordering::Acquire)
}
// Certificado estructural completo: no se infieren simetrías por frecuencias.
fn inject_pigeonhole_symmetry(formula: &mut Formula, variables: &[u32]) -> (usize, usize, usize) {
    use std::collections::{BTreeMap, BTreeSet};
    #[derive(Clone, Copy, Default)]
    struct Occurrence {
        positive_count: usize,
        negative_count: usize,
        positive_clause: usize,
        negative_clause: usize,
    }
    fn find(parent: &mut [usize], mut variable: usize) -> usize {
        while parent[variable] != variable {
            parent[variable] = parent[parent[variable]];
            variable = parent[variable];
        }
        variable
    }
    let variable_index = |literal: i32| {
        variables.binary_search(&literal.unsigned_abs()).expect("Indexed variable")
    };
    let mut occurrences = vec![Occurrence::default(); variables.len()];
    for (clause, literals) in formula.iter().enumerate() {
        for &literal in literals {
            let occurrence = &mut occurrences[variable_index(literal)];
            if literal > 0 {
                occurrence.positive_count += 1;
                occurrence.positive_clause = clause;
            } else {
                occurrence.negative_count += 1;
                occurrence.negative_clause = clause;
            }
        }
    }
    let auxiliary: Vec<bool> = occurrences.iter().map(|occurrence| {
        occurrence.positive_count == 1
            && occurrence.negative_count == 1
            && occurrence.positive_clause != occurrence.negative_clause
            && formula[occurrence.positive_clause].len() == 3
            && formula[occurrence.negative_clause].len() == 3
    }).collect();
    let mut consumed = vec![false; formula.len()];
    let mut rows = Vec::new();
    for source in 0..formula.len() {
        if !formula[source].iter().all(|&literal| literal > 0) {
            continue;
        }
        let mut row = Vec::new();
        let mut current = source;
        let mut incoming = None;
        loop {
            if consumed[current] {
                return (0, 0, 0);
            }
            consumed[current] = true;
            let mut next = None;
            let mut negative_count = 0;
            for &literal in &formula[current] {
                let variable = variable_index(literal);
                if literal < 0 {
                    negative_count += 1;
                    if Some(variable) != incoming {
                        return (0, 0, 0);
                    }
                } else if auxiliary[variable] {
                    if next.replace(variable).is_some() {
                        return (0, 0, 0);
                    }
                } else {
                    row.push(variable);
                }
            }
            if negative_count != usize::from(incoming.is_some()) {
                return (0, 0, 0);
            }
            let Some(variable) = next else { break };
            // (C|y)&(-y|D) equivale a C|D al proyectar el auxiliar privado y.
            current = occurrences[variable].negative_clause;
            incoming = Some(variable);
        }
        rows.push(row);
    }
    let pigeons = rows.len();
    let holes = rows.first().map_or(0, Vec::len);
    if pigeons < 2 || holes < 2 || rows.iter().any(|row| row.len() != holes) {
        return (0, 0, 0);
    }
    let mut owner = vec![usize::MAX; variables.len()];
    for (row_index, row) in rows.iter().enumerate() {
        for &variable in row {
            if owner[variable] != usize::MAX {
                return (0, 0, 0);
            }
            owner[variable] = row_index;
        }
    }
    let mut parent: Vec<usize> = (0..variables.len()).collect();
    let mut degree = vec![0_usize; variables.len()];
    let mut edges = BTreeSet::new();
    for (index, clause) in formula.iter().enumerate() {
        if consumed[index] {
            continue;
        }
        if clause.len() != 2 || clause.iter().any(|&literal| literal >= 0) {
            return (0, 0, 0);
        }
        let a = variable_index(clause[0]);
        let b = variable_index(clause[1]);
        if owner[a] == usize::MAX || owner[b] == usize::MAX || owner[a] == owner[b] {
            return (0, 0, 0);
        }
        if !edges.insert((a.min(b), a.max(b))) {
            return (0, 0, 0);
        }
        degree[a] += 1;
        degree[b] += 1;
        let first = find(&mut parent, a);
        let second = find(&mut parent, b);
        parent[first.max(second)] = first.min(second);
    }
    let mut columns: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for variable in 0..variables.len() {
        if owner[variable] == usize::MAX {
            continue;
        }
        if degree[variable] != pigeons - 1 {
            return (0, 0, 0);
        }
        columns.entry(find(&mut parent, variable)).or_default().push(variable);
    }
    if columns.len() != holes {
        return (0, 0, 0);
    }
    let mut matrix = vec![vec![0_i32; holes]; pigeons];
    for (column_index, column) in columns.values().enumerate() {
        if column.len() != pigeons {
            return (0, 0, 0);
        }
        let mut present = vec![false; pigeons];
        for &variable in column {
            let row = owner[variable];
            if present[row] {
                return (0, 0, 0);
            }
            present[row] = true;
            matrix[row][column_index] = variables[variable] as i32;
        }
        // p nodos distintos, grado p-1 y una celda por fila certifican K_p.
    }
    // L literales, V variables: reconocimiento O(L*V + L*log(L));
    // se añaden O(p*h^2) cláusulas y ninguna variable nueva.
    let before = formula.len();
    // PHP solo exige al menos una celda por fila: borrar otras celdas verdaderas
    // preserva sus cláusulas. Así se selecciona primero un modelo funcional.
    for row in &matrix {
        for first in 0..holes {
            for second in first + 1..holes {
                formula.push(vec![-row[first], -row[second]]);
            }
        }
    }
    // Las permutaciones de filas son simetrías del PHP proyectado certificado.
    // Orden estricto de las columnas elegidas: c[r] < c[r+1].
    for row in 0..pigeons - 1 {
        for first in 0..holes {
            for second in 0..=first {
                formula.push(vec![-matrix[row][first], -matrix[row + 1][second]]);
            }
        }
    }
    // Consecuencias del orden: c[r]>=r; si p<=h, c[r]<=h-p+r.
    // Para p=h+1, la última fila queda falsa y BCP cierra en nivel cero.
    for (row_index, row) in matrix.iter().enumerate() {
        for &literal in &row[..row_index.min(holes)] {
            formula.push(vec![-literal]);
        }
        if pigeons <= holes {
            for &literal in &row[holes - pigeons + row_index + 1..] {
                formula.push(vec![-literal]);
            }
        }
    }
    (pigeons, holes, formula.len() - before)
}
const DECAY: f64 = 0.95;
const CLEANUP_INTERVAL: usize = 64;
const MAX_ACTIVITY: f64 = 1e100;
#[derive(Clone, Copy)]
struct TrailEntry {
    literal: usize,
    level: usize,
    reason: Option<usize>,
}
#[derive(Clone)]
enum Undo {
    Assigned(usize),
}
#[derive(Clone)]
struct State {
    literals: Vec<usize>,
    offsets: Vec<usize>,
    watches: Vec<Vec<usize>>,
    watched_positions: Vec<[usize; 2]>,
    is_learned: Vec<bool>,
    clause_activity: Vec<f64>,
    learned_count: usize,
    values: Vec<i8>,
    levels: Vec<usize>,
    reasons: Vec<Option<usize>>,
    scores: Vec<f64>,
    activity_scale: f64,
    phases: Vec<usize>,
    phase_known: Vec<bool>,
    units: VecDeque<(usize, usize)>,
    trail: Vec<TrailEntry>,
    trail_limits: Vec<usize>,
    propagation_head: usize,
    undo: Vec<Undo>,
    seen: Vec<bool>,
    touched: Vec<usize>,
    next_cleanup: usize,
    root_unsat: bool,
    stats: SolveStats,
}
impl State {
    fn value(&self, literal: usize) -> i8 {
        let value = self.values[literal >> 1];
        if literal & 1 == 0 { value } else { -value }
    }
    fn add_clause(&mut self, clause: &[usize], learned: bool) -> usize {
        assert!(!clause.is_empty());
        let index = self.is_learned.len();
        let start = self.literals.len();
        self.literals.extend_from_slice(clause);
        self.offsets.push(self.literals.len());
        self.is_learned.push(learned);
        self.clause_activity.push(if learned { 1.0 } else { 0.0 });
        if clause.len() == 1 {
            self.watched_positions.push([start, start]);
            if !learned {
                self.units.push_back((clause[0], index));
            }
        } else {
            self.watched_positions.push([start, start + 1]);
            self.watches[clause[0]].push(index);
            self.watches[clause[1]].push(index);
        }
        if learned {
            self.learned_count += 1;
            self.stats.learned_clauses += 1;
        }
        index
    }
    fn assign(&mut self, literal: usize, reason: Option<usize>) -> bool {
        let variable = literal >> 1;
        let value = if literal & 1 == 0 { 1 } else { -1 };
        if self.values[variable] != 0 {
            return self.values[variable] == value;
        }
        let level = self.trail_limits.len();
        self.values[variable] = value;
        // Se guarda la fase de decisiones; las implicaciones nunca sustituyen el prior.
        if reason.is_none() {
            self.phases[variable] = literal & 1;
            self.phase_known[variable] = true;
        }
        self.levels[variable] = level;
        self.reasons[variable] = reason;
        self.trail.push(TrailEntry { literal, level, reason });
        self.undo.push(Undo::Assigned(variable));
        if reason.is_some() {
            self.stats.propagations += 1;
        }
        true
    }
    fn propagate(&mut self) -> Option<usize> {
        match self.propagate_with_cancel(None) {
            Propagation::Stable => None,
            Propagation::Conflict(clause) => Some(clause),
            Propagation::Cancelled => unreachable!("Serial propagation has no cancellation"),
        }
    }
    fn propagate_with_cancel(&mut self, cancel: Option<&AtomicBool>) -> Propagation {
        let mut work = 0;
        if cancel.is_some_and(|flag| flag.load(Ordering::Acquire)) {
            return Propagation::Cancelled;
        }
        loop {
            while !self.units.is_empty() {
                if poll_cancel(cancel, &mut work) { return Propagation::Cancelled; }
                let (literal, clause) = self.units.pop_front().expect("Queued unit");
                if !self.assign(literal, Some(clause)) {
                    return Propagation::Conflict(clause);
                }
            }
            if self.propagation_head == self.trail.len() {
                return Propagation::Stable;
            }
            if poll_cancel(cancel, &mut work) { return Propagation::Cancelled; }
            let false_literal = self.trail[self.propagation_head].literal ^ 1;
            self.propagation_head += 1;
            let mut index = 0;
            while index < self.watches[false_literal].len() {
                if poll_cancel(cancel, &mut work) { return Propagation::Cancelled; }
                let clause = self.watches[false_literal][index];
                let positions = self.watched_positions[clause];
                let slot = if self.literals[positions[0]] == false_literal {
                    0
                } else {
                    debug_assert_eq!(self.literals[positions[1]], false_literal);
                    1
                };
                let other = self.literals[positions[slot ^ 1]];
                let other_value = self.value(other);
                if other_value == 1 {
                    index += 1;
                    continue;
                }
                let mut replacement = None;
                for position in self.offsets[clause]..self.offsets[clause + 1] {
                    if poll_cancel(cancel, &mut work) { return Propagation::Cancelled; }
                    if position != positions[0]
                        && position != positions[1]
                        && self.value(self.literals[position]) != -1
                    {
                        replacement = Some(position);
                        break;
                    }
                }
                if let Some(position) = replacement {
                    self.watched_positions[clause][slot] = position;
                    self.watches[false_literal].swap_remove(index);
                    self.watches[self.literals[position]].push(clause);
                    continue;
                }
                if other_value == -1 {
                    return Propagation::Conflict(clause);
                }
                self.units.push_back((other, clause));
                index += 1;
            }
        }
    }
    fn import_level(&self, clause: SmallClause) -> Option<usize> {
        let mut nonfalse = None;
        let mut false_level = 0;
        for &literal in clause.as_slice() {
            if self.value(literal) == -1 {
                false_level = false_level.max(self.levels[literal >> 1]);
            } else if nonfalse.replace(literal).is_some() {
                return None;
            }
        }
        match nonfalse {
            None => Some(false_level),
            Some(literal) if self.value(literal) == 0
                || self.levels[literal >> 1] > false_level => Some(false_level),
            Some(_) => None,
        }
    }
    fn attach_import(&mut self, clause: SmallClause) -> Option<usize> {
        let mut literals = clause.literals;
        // Falsos de mayor nivel primero: sus watches se liberan tras un backjump.
        literals[..clause.len].sort_unstable_by_key(|&literal| {
            (self.value(literal) == -1, std::cmp::Reverse(self.levels[literal >> 1]))
        });
        let index = self.add_clause(&literals[..clause.len], true);
        self.stats.imported_clauses += 1;
        if literals[..clause.len].iter().any(|&literal| self.value(literal) == 1) {
            return None;
        }
        let mut unassigned = literals[..clause.len].iter()
            .copied().filter(|&literal| self.value(literal) == 0);
        let Some(literal) = unassigned.next() else { return Some(index); };
        if unassigned.next().is_none() {
            self.units.push_back((literal, index));
        }
        None
    }
    fn update_vsids(&mut self, conflict: usize) {
        let start = self.offsets[conflict];
        let end = self.offsets[conflict + 1];
        for index in start..end {
            let variable = self.literals[index] >> 1;
            if !self.literals[start..index].iter()
                .any(|&literal| literal >> 1 == variable)
            {
                self.scores[variable] += 1.0;
            }
        }
        for score in &mut self.scores {
            *score *= DECAY;
        }
    }
    fn analyze_conflict(&mut self, conflict: usize) -> (Vec<usize>, usize) {
        let current_level = self.trail_limits.len();
        debug_assert!(current_level > 0);
        let mut learned = vec![0];
        let mut pending = 0;
        let mut cursor = self.trail.len();
        let mut clause = conflict;
        let mut pivot_variable = None;
        loop {
            self.clause_activity[clause] += 1.0;
            for index in self.offsets[clause]..self.offsets[clause + 1] {
                let literal = self.literals[index];
                let variable = literal >> 1;
                if Some(variable) == pivot_variable
                    || self.seen[variable]
                    || self.levels[variable] == 0
                {
                    continue;
                }
                self.seen[variable] = true;
                self.touched.push(variable);
                if self.levels[variable] == current_level {
                    pending += 1;
                } else {
                    learned.push(literal);
                }
            }
            let pivot = loop {
                cursor = cursor.checked_sub(1).expect("Conflict has a current-level pivot");
                let entry = self.trail[cursor];
                if entry.level == current_level && self.seen[entry.literal >> 1] {
                    break entry;
                }
            };
            self.seen[pivot.literal >> 1] = false;
            pending -= 1;
            if pending == 0 {
                learned[0] = pivot.literal ^ 1;
                break;
            }
            pivot_variable = Some(pivot.literal >> 1);
            clause = pivot.reason.expect("A non-UIP pivot has an antecedent");
        }
        for variable in self.touched.drain(..) {
            self.seen[variable] = false;
        }
        let mut backjump_level = 0;
        if learned.len() > 1 {
            let mut highest = 1;
            for index in 2..learned.len() {
                if self.levels[learned[index] >> 1] > self.levels[learned[highest] >> 1] {
                    highest = index;
                }
            }
            learned.swap(1, highest);
            backjump_level = self.levels[learned[1] >> 1];
        }
        debug_assert!(backjump_level < current_level);
        (learned, backjump_level)
    }
    fn backjump(&mut self, level: usize) {
        let mark = self.trail_limits[level];
        while self.undo.len() > mark {
            match self.undo.pop().expect("Undo entry exists") {
                Undo::Assigned(variable) => {
                    self.values[variable] = 0;
                    self.levels[variable] = 0;
                    self.reasons[variable] = None;
                }
            }
        }
        self.trail.truncate(mark);
        self.trail_limits.truncate(level);
        self.propagation_head = mark;
        self.units.clear();
    }
    fn choose_activity_literal(&self) -> Option<usize> {
        let mut best_variable = None;
        let mut best_score = f64::NEG_INFINITY;
        for (variable, &score) in self.scores.iter().enumerate() {
            if self.values[variable] == 0 && score > best_score {
                best_variable = Some(variable);
                best_score = score;
            }
        }
        best_variable.map(|variable| (variable << 1) | self.phases[variable])
    }
    fn refresh_guidance(&mut self, tensor: &[f64], activity_boost: f64) {
        assert_eq!(tensor.len(), self.values.len());
        let scale = self.activity_scale;
        for (variable, &value) in tensor.iter().enumerate() {
            assert!(value.is_finite() && (0.0..=1.0).contains(&value));
            if self.values[variable] != 0 {
                continue;
            }
            self.phases[variable] = usize::from(value < 0.5);
            self.phase_known[variable] = true;
            let bonus = (activity_boost.min(MAX_ACTIVITY)
                * (value - 0.5).abs() * scale).min(MAX_ACTIVITY);
            self.scores[variable] = (self.scores[variable] + bonus).min(MAX_ACTIVITY);
        }
        // No se modifica ningún valor, antecedente, watch, nivel ni entrada del trail.
    }
    fn observed_pressure(&self, literal: usize) -> f64 {
        let mut pressure = 0.0;
        for &clause in &self.watches[literal] {
            let positions = self.watched_positions[clause];
            let other = if self.literals[positions[0]] == literal {
                self.literals[positions[1]]
            } else {
                debug_assert_eq!(self.literals[positions[1]], literal);
                self.literals[positions[0]]
            };
            if self.value(other) != 1 {
                pressure += 1.0;
            }
        }
        pressure
    }
    fn choose_literal(&self) -> Option<usize> {
        if self.stats.conflicts == 0 {
            return self.choose_activity_literal();
        }
        let mut best_literal = None;
        let mut best_score = f64::NEG_INFINITY;
        let dense = self.is_learned.len() >= self.values.len().saturating_mul(2);
        for (variable, &activity) in self.scores.iter().enumerate() {
            if self.values[variable] != 0 {
                continue;
            }
            let positive = variable << 1;
            let (p, n) = if dense {
                (self.watches[positive].len() as f64,
                    self.watches[positive ^ 1].len() as f64)
            } else {
                (self.observed_pressure(positive), self.observed_pressure(positive ^ 1))
            };
            let total = p + n;
            if total == 0.0 {
                continue;
            }
            let imbalance = (p - n) / (1.0 + total);
            let score = (1.0 + activity) * total * (1.0 + imbalance * imbalance);
            if score > best_score {
                best_score = score;
                let sign = if self.phase_known[variable] {
                    self.phases[variable]
                } else {
                    usize::from(p > n)
                };
                best_literal = Some(positive | sign);
            }
        }
        // Presión cero no prueba satisfacción: se conserva el fallback completo.
        best_literal.or_else(|| self.choose_activity_literal())
    }
    fn reduce_database(&mut self) {
        debug_assert!(self.units.is_empty());
        debug_assert_eq!(self.propagation_head, self.trail.len());
        let count = self.is_learned.len();
        let mut locked = vec![false; count];
        for &reason in self.reasons.iter().flatten() {
            locked[reason] = true;
        }
        let mut candidates: Vec<usize> = (0..count).filter(|&clause| {
            self.is_learned[clause]
                && !locked[clause]
                && self.offsets[clause + 1] - self.offsets[clause] > 2
        }).collect();
        candidates.sort_unstable_by(|&a, &b| {
            self.clause_activity[a].total_cmp(&self.clause_activity[b])
                .then_with(|| a.cmp(&b))
        });
        for activity in &mut self.clause_activity {
            *activity *= 0.5;
        }
        if candidates.is_empty() {
            return;
        }
        let remove_count = (candidates.len() / 2).max(1);
        let mut keep = vec![true; count];
        for &clause in &candidates[..remove_count] {
            keep[clause] = false;
        }
        let retained_literals: usize = (0..count).filter(|&c| keep[c])
            .map(|c| self.offsets[c + 1] - self.offsets[c]).sum();
        let retained_clauses = count - remove_count;
        let mut literals = Vec::with_capacity(retained_literals);
        let mut offsets = Vec::with_capacity(retained_clauses + 1);
        let mut watched_positions = Vec::with_capacity(retained_clauses);
        let mut is_learned = Vec::with_capacity(retained_clauses);
        let mut activity = Vec::with_capacity(retained_clauses);
        let mut remap = vec![usize::MAX; count];
        offsets.push(0);
        for old in 0..count {
            if !keep[old] {
                continue;
            }
            remap[old] = is_learned.len();
            let old_start = self.offsets[old];
            let start = literals.len();
            literals.extend_from_slice(&self.literals[old_start..self.offsets[old + 1]]);
            offsets.push(literals.len());
            let positions = self.watched_positions[old];
            watched_positions.push([
                start + positions[0] - old_start,
                start + positions[1] - old_start,
            ]);
            is_learned.push(self.is_learned[old]);
            activity.push(self.clause_activity[old]);
        }
        for reason in self.reasons.iter_mut().flatten() {
            *reason = remap[*reason];
            assert_ne!(*reason, usize::MAX, "Locked reason must survive cleanup");
        }
        for entry in &mut self.trail {
            if let Some(reason) = entry.reason {
                entry.reason = Some(remap[reason]);
            }
        }
        let mut watches = vec![Vec::new(); self.watches.len()];
        for clause in 0..retained_clauses {
            if offsets[clause + 1] - offsets[clause] >= 2 {
                let positions = watched_positions[clause];
                watches[literals[positions[0]]].push(clause);
                watches[literals[positions[1]]].push(clause);
            }
        }
        self.literals = literals;
        self.offsets = offsets;
        self.watched_positions = watched_positions;
        self.watches = watches;
        self.is_learned = is_learned;
        self.clause_activity = activity;
        self.learned_count -= remove_count;
        self.stats.deleted_clauses += remove_count;
        self.stats.database_reductions += 1;
    }
}
fn prepare_formula(
    mut formula: Formula,
    guidance: Option<&[(u32, f64)]>,
    activity_boost: f64,
) -> Option<(State, Vec<u32>)> {
    if formula.iter().any(Vec::is_empty) {
        return None;
    }
    let mut variables: Vec<u32> = formula.iter().flatten().map(|&literal| {
        assert!(literal != 0 && literal != i32::MIN, "Invalid SAT literal");
        literal.unsigned_abs()
    }).collect();
    variables.sort_unstable();
    variables.dedup();
    // La canonización precede a cualquier supuesto de un cubo.
    let (symmetry_rows, symmetry_columns, symmetry_clauses) =
        inject_pigeonhole_symmetry(&mut formula, &variables);
    let literal_count = variables.len().checked_mul(2).expect("Too many variables");
    let mut state = State {
        literals: Vec::new(),
        offsets: vec![0],
        watches: vec![Vec::new(); literal_count],
        watched_positions: Vec::with_capacity(formula.len()),
        is_learned: Vec::with_capacity(formula.len()),
        clause_activity: Vec::with_capacity(formula.len()),
        learned_count: 0,
        values: vec![0; variables.len()],
        levels: vec![0; variables.len()],
        reasons: vec![None; variables.len()],
        scores: vec![0.0; variables.len()],
        activity_scale: 1.0,
        phases: vec![0; variables.len()],
        phase_known: vec![false; variables.len()],
        units: VecDeque::new(),
        trail: Vec::with_capacity(variables.len()),
        trail_limits: Vec::with_capacity(variables.len()),
        propagation_head: 0,
        undo: Vec::with_capacity(variables.len()),
        seen: vec![false; variables.len()],
        touched: Vec::with_capacity(variables.len()),
        next_cleanup: CLEANUP_INTERVAL,
        root_unsat: false,
        stats: SolveStats {
            symmetry_rows, symmetry_columns, symmetry_clauses,
            ..SolveStats::default()
        },
    };
    for clause in formula {
        let packed: Vec<usize> = clause.into_iter().map(|literal| {
            let variable = variables.binary_search(&literal.unsigned_abs())
                .expect("Variable was indexed");
            state.scores[variable] += 1.0;
            (variable << 1) | usize::from(literal < 0)
        }).collect();
        state.add_clause(&packed, false);
    }
    // Referencia fija: el feedback no multiplica boosts anteriores.
    state.activity_scale = state.scores.iter().copied().fold(1.0_f64, f64::max);
    if let Some(prior) = guidance {
        assert!(prior.windows(2).all(|pair| pair[0].0 < pair[1].0));
        let scale = state.activity_scale;
        for &(id, value) in prior {
            assert!(value.is_finite() && (0.0..=1.0).contains(&value));
            if let Ok(variable) = variables.binary_search(&id) {
                state.phases[variable] = usize::from(value < 0.5);
                state.phase_known[variable] = true;
                // Saturar antes del producto evita inf*0 y mantiene el ordenamiento finito.
                let bonus = (activity_boost.min(MAX_ACTIVITY)
                    * (value - 0.5).abs() * scale).min(MAX_ACTIVITY);
                state.scores[variable] = (state.scores[variable] + bonus).min(MAX_ACTIVITY);
            }
        }
    }
    Some((state, variables))
}
pub fn solve_cdcl(
    formula: &[Vec<i32>],
    guidance: Option<&[(u32, f64)]>,
    activity_boost: f64,
) -> (Option<Vec<i32>>, SolveStats) {
    solve_cdcl_inner(formula, guidance, activity_boost, None)
}
pub fn solve_cdcl_with_reignition(
    formula: &[Vec<i32>],
    guidance: Option<&[(u32, f64)]>,
    activity_boost: f64,
    config: ReignitionConfig,
) -> (Option<Vec<i32>>, SolveStats) {
    solve_cdcl_inner(formula, guidance, activity_boost, Some(config))
}
fn solve_cdcl_inner(
    formula: &[Vec<i32>],
    guidance: Option<&[(u32, f64)]>,
    activity_boost: f64,
    config: Option<ReignitionConfig>,
) -> (Option<Vec<i32>>, SolveStats) {
    assert!(activity_boost.is_finite() && activity_boost >= 0.0);
    let Some((mut state, variables)) = prepare_formula(formula.to_vec(), guidance, activity_boost) else {
        return (None, SolveStats { conflicts: 1, ..SolveStats::default() });
    };
    let control = SearchControl { cube: &[], cancel: None, sharing: None };
    let (outcome, stats) = run_state(&mut state, &variables, formula, guidance, activity_boost, config, control);
    match outcome {
        SearchOutcome::Sat(model) => (Some(model), stats),
        SearchOutcome::Unsat => (None, stats),
        SearchOutcome::Cancelled => unreachable!("Serial execution cannot be cancelled"),
    }
}
fn run_state(
    state: &mut State,
    variables: &[u32],
    formula: &[Vec<i32>],
    guidance: Option<&[(u32, f64)]>,
    activity_boost: f64,
    config: Option<ReignitionConfig>,
    control: SearchControl<'_>,
) -> (SearchOutcome, SolveStats) {
    if control.cancel.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return (SearchOutcome::Cancelled, state.stats);
    }
    // El certificado raiz evita reprocesar una cola contradictoria y cubos posteriores.
    if state.root_unsat { return (SearchOutcome::Unsat, state.stats); }
    let learned_limit = variables.len().saturating_mul(4).max(64);
    let feedback = config.filter(|configuration| configuration.interval != 0);
    let mut next_feedback = feedback.map_or(usize::MAX, |configuration| configuration.interval);
    let mut workspace: Option<OracleWorkspace> = None;
    let mut cursor = 0;
    let mut next_share = 0;
    let mut pending_conflict = None;
    let mut known = HashSet::new();
    if control.sharing.is_some() {
        for span in state.offsets.windows(2) {
            if let Some(clause) = SmallClause::canonical(&state.literals[span[0]..span[1]]) {
                known.insert(clause);
            }
        }
    }
    loop {
        if control.cancel.is_some_and(|flag| flag.load(Ordering::Acquire)) {
            return (SearchOutcome::Cancelled, state.stats);
        }
        let propagation = if let Some(conflict) = pending_conflict.take() {
            Propagation::Conflict(conflict)
        } else {
            state.propagate_with_cancel(control.cancel)
        };
        match propagation {
            Propagation::Cancelled => return (SearchOutcome::Cancelled, state.stats),
            Propagation::Conflict(conflict) => {
                state.stats.conflicts += 1;
                state.update_vsids(conflict);
                let level = state.trail_limits.len();
                if level == 0 {
                    // No se publica la cláusula vacía ni se confunde cancelación con UNSAT.
                    state.root_unsat = true;
                    return (SearchOutcome::Unsat, state.stats);
                }
                let (learned, backjump_level) = state.analyze_conflict(conflict);
                if let Some(sharing) = control.sharing {
                    if let Some(clause) = SmallClause::canonical(&learned) {
                        // Clausulas globales de hasta tres literales; CDCL local conserva todas.
                        if known.insert(clause) {
                            if sharing.pool.publish(clause) {
                                state.stats.exported_clauses += 1;
                            } else {
                                state.stats.dropped_exports += 1;
                            }
                        }
                    }
                }
                state.stats.backjumps += 1;
                state.stats.nonchronological_backjumps += usize::from(level > backjump_level + 1);
                state.backjump(backjump_level);
                let clause = state.add_clause(&learned, true);
                assert!(state.assign(learned[0], Some(clause)), "Learned clause must assert");
                continue;
            }
            Propagation::Stable => {}
        }
        let mut reduced = false;
        if state.stats.conflicts >= state.next_cleanup {
            if state.learned_count > learned_limit {
                state.reduce_database();
                reduced = true;
            }
            state.next_cleanup = state.stats.conflicts.saturating_add(CLEANUP_INTERVAL);
        }
        let feedback_due = feedback.is_some() && state.stats.conflicts >= next_feedback;
        if let Some(sharing) = control.sharing {
            if state.stats.conflicts >= next_share || feedback_due || reduced {
                let imports: Vec<SmallClause> = sharing.pool.harvest(&mut cursor).into_iter()
                    .filter_map(|entry| known.insert(entry.clause).then_some(entry.clause)).collect();
                next_share = state.stats.conflicts.saturating_add(sharing.interval.max(1));
                if !imports.is_empty() {
                    loop {
                        let current = state.trail_limits.len();
                        let required = imports.iter().filter_map(|&clause| state.import_level(clause))
                            .min().unwrap_or(current);
                        if required >= current { break; }
                        state.backjump(required);
                        state.stats.import_backjumps += 1;
                    }
                    for clause in imports {
                        if let Some(conflict) = state.attach_import(clause) {
                            pending_conflict.get_or_insert(conflict);
                        }
                    }
                    // BCP/conflicto antes de Adam, replay del cubo o nuevas decisiones.
                    continue;
                }
            }
        }
        if let Some(configuration) = feedback {
            if feedback_due {
                let started = Instant::now();
                let oracle = workspace.get_or_insert_with(|| {
                    let mut initial: Vec<f64> = state.phases.iter()
                        .map(|&phase| if phase == 0 { 1.0 } else { 0.0 }).collect();
                    if let Some(prior) = guidance {
                        for &(id, value) in prior {
                            if let Ok(variable) = variables.binary_search(&id) {
                                initial[variable] = value;
                            }
                        }
                    }
                    OracleWorkspace::new(formula, &initial)
                });
                let (updates, cancelled) = if let Some(cancel) = control.cancel {
                    oracle.reignite_with_cancel(&state.values, configuration.oracle, cancel)
                } else {
                    (oracle.reignite(&state.values, configuration.oracle), false)
                };
                state.stats.reignitions += 1;
                state.stats.reignition_epochs += updates;
                state.stats.reignition_ms += started.elapsed().as_secs_f64() * 1000.0;
                if cancelled { return (SearchOutcome::Cancelled, state.stats); }
                state.refresh_guidance(oracle.values(), activity_boost);
                next_feedback = state.stats.conflicts.saturating_add(configuration.interval);
            }
        }
        if state.trail_limits.len() < control.cube.len() {
            let literal = control.cube[state.trail_limits.len()];
            let value = state.value(literal);
            if value == -1 { return (SearchOutcome::Unsat, state.stats); }
            // Un nivel por supuesto; niveles vacíos solo se crean en un fixpoint.
            state.trail_limits.push(state.trail.len());
            if value == 0 {
                state.stats.decisions += 1;
                state.stats.cube_decisions += 1;
                assert!(state.assign(literal, None));
            }
            // Cubos como DECISIONES: sus literales permanecen en las cláusulas 1UIP.
            continue;
        }
        let Some(literal) = state.choose_literal() else {
            let model = variables.iter().zip(&state.values).map(|(&id, &value)| {
                if value == 1 { id as i32 } else { -(id as i32) }
            }).collect();
            return (SearchOutcome::Sat(model), state.stats);
        };
        state.trail_limits.push(state.trail.len());
        state.stats.decisions += 1;
        assert!(state.assign(literal, None), "Decision variable must be unassigned");
    }
}
fn make_cubes(base: &State, config: SwarmConfig) -> Vec<Vec<usize>> {
    let target = config.workers.max(1).saturating_mul(config.cubes_per_worker.max(1)).clamp(1, 1024);
    let mut candidates: Vec<usize> = (0..base.values.len())
        .filter(|&variable| base.values[variable] == 0).collect();
    candidates.sort_unstable_by(|&a, &b| {
        base.scores[b].total_cmp(&base.scores[a]).then_with(|| a.cmp(&b))
    });
    let mut depth = 0;
    while depth < candidates.len().min(10) && (1 << depth) < target { depth += 1; }
    candidates.truncate(depth);
    // Todas las combinaciones: la guía ordena cubos y nunca descarta ninguno.
    (0..1usize << depth).map(|mask| candidates.iter().enumerate().map(|(index, &variable)| {
        (variable << 1) | (base.phases[variable] ^ ((mask >> index) & 1))
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
    formula: &[Vec<i32>],
    guidance: Option<&[(u32, f64)]>,
    cubes: &[Vec<usize>],
    next: &AtomicUsize,
    cancel: &AtomicBool,
    pool: &CloudPool,
    config: SwarmConfig,
) -> WorkerReport {
    let mut report = WorkerReport::default();
    let mut state = base.clone();
    loop {
        if cancel.load(Ordering::Acquire) { break; }
        let index = next.fetch_add(1, Ordering::Relaxed);
        let Some(cube) = cubes.get(index) else { break; };
        report.started += 1;
        if !state.root_unsat && !state.trail_limits.is_empty() {
            // Solo se llega al siguiente cubo tras CubeUNSAT en un fixpoint.
            debug_assert!(state.units.is_empty());
            debug_assert_eq!(state.propagation_head, state.trail.len());
            state.backjump(0);
        }
        // Clausulas, razones/unidades raiz, fases y actividades sobreviven al cubo.
        state.stats = SolveStats::default();
        state.next_cleanup = CLEANUP_INTERVAL;
        let sharing = (config.share_enabled && config.share_capacity != 0)
            .then_some(Sharing { pool, interval: config.share_interval.max(1) });
        let control = SearchControl { cube, cancel: Some(cancel), sharing };
        let (outcome, stats) = run_state(&mut state, variables, formula, guidance,
            config.activity_boost, Some(config.reignition), control);
        report.total.add_search(stats);
        match outcome {
            SearchOutcome::Sat(model) => {
                if cancel.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                    report.model = Some(model);
                } else { report.cancelled += 1; }
                break;
            }
            SearchOutcome::Unsat => report.unsat += 1,
            SearchOutcome::Cancelled => { report.cancelled += 1; break; }
        }
    }
    report
}
pub fn solve_swarm(
    formula: &[Vec<i32>],
    guidance: Option<&[(u32, f64)]>,
    config: SwarmConfig,
) -> (Option<Vec<i32>>, SwarmStats) {
    assert!(config.activity_boost.is_finite() && config.activity_boost >= 0.0);
    let mut stats = SwarmStats::default();
    let Some((mut base, variables)) = prepare_formula(formula.to_vec(), guidance, config.activity_boost) else {
        stats.total.conflicts = 1;
        return (None, stats);
    };
    if let Some(conflict) = base.propagate() {
        base.stats.conflicts += 1;
        base.update_vsids(conflict);
        stats.total = base.stats;
        return (None, stats);
    }
    stats.total = base.stats;
    if base.choose_literal().is_none() {
        let model = variables.iter().zip(&base.values).map(|(&id, &value)| {
            if value == 1 { id as i32 } else { -(id as i32) }
        }).collect();
        return (Some(model), stats);
    }
    let cubes = make_cubes(&base, config);
    stats.cubes_generated = cubes.len();
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
            let worker = move || swarm_worker(base, variables, formula, guidance, cubes,
                next, cancel, pool, config);
            match std::thread::Builder::new().name(format!("swarm-{owner}"))
                .spawn_scoped(scope, worker) {
                Ok(handle) => handles.push(handle),
                Err(_) => break,
            }
        }
        if handles.is_empty() {
            return (0, vec![swarm_worker(&base, &variables, formula, guidance, &cubes,
                &next, &cancel, &pool, config)]);
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
        assert_eq!(stats.cubes_unsat, stats.cubes_generated, "Incomplete cube coverage cannot prove UNSAT");
    }
    (winner, stats)
}

#[cfg(test)]
mod deterministic_import_tests {
    use super::*;

    #[test]
    fn published_consequence_is_imported_and_promoted_to_root() {
        // (x1 OR x2) AND (x1 OR !x2) entails x1 without a syntactic unit.
        let formula = vec![vec![1, 2], vec![1, -2]];
        let (mut state, variables) = prepare_formula(formula.clone(), None, 4.0).unwrap();
        let positive_x1 = 0;
        let negative_x2 = 3;
        state.trail_limits.push(state.trail.len());
        assert!(state.assign(negative_x2, None));
        assert!(state.propagate().is_none());
        assert_eq!(state.values[0], 1);
        assert_eq!(state.levels[0], 1);

        let pool = CloudPool::new(1);
        assert!(pool.publish(SmallClause::canonical(&[positive_x1]).unwrap()));
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

        let control = SearchControl {
            cube: &[], cancel: None,
            sharing: Some(Sharing { pool: &pool, interval: 1 }),
        };
        let (outcome, stats) = run_state(
            &mut state, &variables, &formula, None, 4.0, None, control);
        let model = match outcome {
            SearchOutcome::Sat(model) => model,
            _ => panic!("Importing a valid consequence must preserve SAT"),
        };
        assert!(model.contains(&1));
        assert_eq!(stats.imported_clauses, 1);
        assert_eq!(stats.import_backjumps, 1);
        assert_eq!(state.values[0], 1);
        assert_eq!(state.levels[0], 0);
        let reason = state.reasons[0].expect("Imported root unit has an antecedent");
        assert!(state.is_learned[reason]);
        assert_eq!(&state.literals[state.offsets[reason]..state.offsets[reason + 1]],
            &[positive_x1]);
    }
}
