// Checkpoint Fase 6 guardado antes de esta mutación: fe399e2.
// Cube-and-Conquer: canonización PHP y BCP raíz antes de clonar estados CDCL privados.
// Lookahead acotado selecciona variables; todos sus cubos se cubren, con un máximo de 1024.
// El primer modelo se publica bajo Mutex; AtomicBool cancela y todos los trabajadores se unen.
// SAT/UNSAT sigue siendo exacto; el modelo ganador y el trabajo total dependen del planificador.

#![forbid(unsafe_code)]
#![deny(warnings)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

pub type Clause = Vec<i32>;
pub type Formula = Vec<Clause>;

#[derive(Clone, Copy, Debug, Default)]
pub struct SolveStats {
    pub decisions: usize,
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
}

impl SolveStats {
    fn add_search(&mut self, other: Self) {
        self.decisions += other.decisions;
        self.propagations += other.propagations;
        self.conflicts += other.conflicts;
        self.learned_clauses += other.learned_clauses;
        self.backjumps += other.backjumps;
        self.nonchronological_backjumps += other.nonchronological_backjumps;
        self.deleted_clauses += other.deleted_clauses;
        self.database_reductions += other.database_reductions;
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ParallelStats {
    pub workers: usize,
    pub cubes_generated: usize,
    pub cubes_started: usize,
    pub cubes_unsat: usize,
    pub cubes_cancelled: usize,
    pub lookahead_propagations: usize,
    pub lookahead_conflicts: usize,
    pub total: SolveStats,
}

enum SearchOutcome {
    Sat(Vec<i32>),
    Unsat,
    Cancelled,
}

enum Propagation {
    Stable,
    Conflict(usize),
    Cancelled,
    Limit,
}

fn poll(cancel: &AtomicBool, budget: &mut usize) -> Option<Propagation> {
    if cancel.load(Ordering::Acquire) {
        return Some(Propagation::Cancelled);
    }
    if *budget == 0 {
        return Some(Propagation::Limit);
    }
    *budget -= 1;
    None
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

pub fn solve_dpll(formula: Formula, assignment: Vec<i32>) -> Option<Vec<i32>> {
    solve_cdcl(formula, assignment).0
}

const DECAY: f64 = 0.95;
const CLEANUP_INTERVAL: usize = 64;

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
    units: VecDeque<(usize, usize)>,
    trail: Vec<TrailEntry>,
    trail_limits: Vec<usize>,
    propagation_head: usize,
    undo: Vec<Undo>,
    seen: Vec<bool>,
    touched: Vec<usize>,
    next_cleanup: usize,
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
        self.levels[variable] = level;
        self.reasons[variable] = reason;
        self.trail.push(TrailEntry { literal, level, reason });
        self.undo.push(Undo::Assigned(variable));
        if reason.is_some() {
            self.stats.propagations += 1;
        }
        true
    }

    fn propagate(&mut self, cancel: &AtomicBool, mut budget: usize) -> Propagation {
        loop {
            while !self.units.is_empty() {
                if let Some(outcome) = poll(cancel, &mut budget) {
                    return outcome;
                }
                let (literal, clause) = self.units.pop_front().expect("Queued unit");
                if !self.assign(literal, Some(clause)) {
                    return Propagation::Conflict(clause);
                }
            }
            if self.propagation_head == self.trail.len() {
                return if cancel.load(Ordering::Acquire) {
                    Propagation::Cancelled
                } else {
                    Propagation::Stable
                };
            }
            if let Some(outcome) = poll(cancel, &mut budget) {
                return outcome;
            }
            let false_literal = self.trail[self.propagation_head].literal ^ 1;
            self.propagation_head += 1;
            let mut index = 0;
            while index < self.watches[false_literal].len() {
                if let Some(outcome) = poll(cancel, &mut budget) {
                    return outcome;
                }
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
                    if let Some(outcome) = poll(cancel, &mut budget) {
                        return outcome;
                    }
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

    fn analyze_conflict(&mut self, conflict: usize, cancel: &AtomicBool) -> Option<(Vec<usize>, usize)> {
        let current_level = self.trail_limits.len();
        debug_assert!(current_level > 0);
        let mut learned = vec![0];
        let mut pending = 0;
        let mut cursor = self.trail.len();
        let mut clause = conflict;
        let mut pivot_variable = None;

        loop {
            if cancel.load(Ordering::Acquire) {
                return None;
            }
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
                if cancel.load(Ordering::Acquire) {
                    return None;
                }
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
        Some((learned, backjump_level))
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
        best_variable.map(|variable| variable << 1)
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
                best_literal = Some(positive | usize::from(p > n));
            }
        }
        // Presión cero no prueba satisfacción: se conserva el fallback completo.
        best_literal.or_else(|| self.choose_activity_literal())
    }

    fn reduce_database(&mut self, cancel: &AtomicBool) {
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
        if cancel.load(Ordering::Acquire) {
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
            if cancel.load(Ordering::Acquire) {
                return;
            }
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

fn prepare_formula(mut formula: Formula) -> Option<(State, Vec<u32>)> {
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
        units: VecDeque::new(),
        trail: Vec::with_capacity(variables.len()),
        trail_limits: Vec::with_capacity(variables.len()),
        propagation_head: 0,
        undo: Vec::with_capacity(variables.len()),
        seen: vec![false; variables.len()],
        touched: Vec::with_capacity(variables.len()),
        next_cleanup: CLEANUP_INTERVAL,
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
    Some((state, variables))
}

fn model(state: &State, variables: &[u32], mut assignment: Vec<i32>) -> Vec<i32> {
    // assignment sigue siendo un prefijo de salida, como en la API original.
    assignment.extend(state.trail.iter().map(|entry| {
        let variable = variables[entry.literal >> 1] as i32;
        if entry.literal & 1 == 0 { variable } else { -variable }
    }));
    assignment
}

fn solve_state(
    mut state: State,
    variables: &[u32],
    assignment: Vec<i32>,
    cancel: &AtomicBool,
) -> (SearchOutcome, SolveStats) {
    let learned_limit = variables.len().saturating_mul(4).max(64);
    loop {
        if cancel.load(Ordering::Acquire) {
            return (SearchOutcome::Cancelled, state.stats);
        }
        match state.propagate(cancel, usize::MAX) {
            Propagation::Conflict(conflict) => {
                state.stats.conflicts += 1;
                state.update_vsids(conflict);
                let level = state.trail_limits.len();
                if level == 0 {
                    return (SearchOutcome::Unsat, state.stats);
                }
                let Some((learned, backjump_level)) = state.analyze_conflict(conflict, cancel) else {
                    return (SearchOutcome::Cancelled, state.stats);
                };
                state.stats.backjumps += 1;
                if level > backjump_level + 1 {
                    state.stats.nonchronological_backjumps += 1;
                }
                state.backjump(backjump_level);
                let clause = state.add_clause(&learned, true);
                assert!(state.assign(learned[0], Some(clause)), "Learned clause must assert");
                continue;
            }
            Propagation::Cancelled => return (SearchOutcome::Cancelled, state.stats),
            Propagation::Limit => unreachable!("Unlimited propagation exhausted its budget"),
            Propagation::Stable => {}
        }
        if state.stats.conflicts >= state.next_cleanup {
            if state.learned_count > learned_limit {
                state.reduce_database(cancel);
            }
            state.next_cleanup = state.stats.conflicts.saturating_add(CLEANUP_INTERVAL);
        }
        if cancel.load(Ordering::Acquire) {
            return (SearchOutcome::Cancelled, state.stats);
        }
        let Some(literal) = state.choose_literal() else {
            return (SearchOutcome::Sat(model(&state, variables, assignment)), state.stats);
        };
        state.trail_limits.push(state.trail.len());
        state.stats.decisions += 1;
        assert!(state.assign(literal, None), "Decision variable must be unassigned");
    }
}

pub fn solve_cdcl(formula: Formula, assignment: Vec<i32>) -> (Option<Vec<i32>>, SolveStats) {
    let Some((state, variables)) = prepare_formula(formula) else {
        return (None, SolveStats::default());
    };
    let cancel = AtomicBool::new(false);
    let (outcome, stats) = solve_state(state, &variables, assignment, &cancel);
    match outcome {
        SearchOutcome::Sat(model) => (Some(model), stats),
        SearchOutcome::Unsat => (None, stats),
        SearchOutcome::Cancelled => unreachable!("Serial execution cannot be cancelled"),
    }
}

fn make_cubes(base: &State, workers: usize, stats: &mut ParallelStats) -> Vec<Vec<usize>> {
    const MAX_DEPTH: usize = 10;
    const MAX_CANDIDATES: usize = 24;
    const PROBE_BUDGET: usize = 2048;
    let mut candidates: Vec<usize> = (0..base.values.len())
        .filter(|&variable| base.values[variable] == 0).collect();
    candidates.sort_unstable_by(|&a, &b| {
        base.scores[b].total_cmp(&base.scores[a]).then_with(|| a.cmp(&b))
    });
    let target = workers.saturating_mul(4).clamp(1, 1 << MAX_DEPTH);
    let mut depth = 0;
    while depth < MAX_DEPTH && depth < candidates.len() && (1 << depth) < target {
        depth += 1;
    }
    candidates.truncate(MAX_CANDIDATES.max(depth));
    let cancel = AtomicBool::new(false);
    let mut ranked = Vec::with_capacity(candidates.len());
    for variable in candidates {
        let mut impacts = [0_usize; 2];
        let mut conflicts = 0_usize;
        for (sign, impact) in impacts.iter_mut().enumerate() {
            let mut probe = base.clone();
            probe.stats = SolveStats::default();
            probe.add_clause(&[(variable << 1) | sign], false);
            let conflict = matches!(probe.propagate(&cancel, PROBE_BUDGET), Propagation::Conflict(_));
            conflicts += usize::from(conflict);
            *impact = probe.trail.len() - base.trail.len();
            stats.lookahead_propagations += probe.stats.propagations;
            stats.lookahead_conflicts += usize::from(conflict);
        }
        let product = impacts[0].saturating_add(1).saturating_mul(impacts[1].saturating_add(1));
        ranked.push((variable, conflicts, product));
    }
    ranked.sort_unstable_by(|a, b| {
        b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2))
            .then_with(|| base.scores[b.0].total_cmp(&base.scores[a.0]))
            .then_with(|| a.0.cmp(&b.0))
    });
    let selected: Vec<usize> = ranked.iter().take(depth).map(|&(variable, _, _)| variable).collect();
    // Se enumeran TODAS las combinaciones de las variables seleccionadas.
    // El lookahead únicamente ordena; alcanzar su presupuesto no poda ningún cubo.
    (0..1_usize << selected.len()).map(|mask| {
        selected.iter().enumerate().map(|(index, &variable)| {
            (variable << 1) | ((mask >> index) & 1)
        }).collect()
    }).collect()
}

#[derive(Default)]
struct WorkerReport {
    started: usize,
    unsat: usize,
    cancelled: usize,
    stats: SolveStats,
}

fn conquer_worker(
    base: &State,
    variables: &[u32],
    cubes: &[Vec<usize>],
    assignment: &[i32],
    next: &AtomicUsize,
    cancel: &AtomicBool,
    winner: &Mutex<Option<Vec<i32>>>,
) -> WorkerReport {
    let mut report = WorkerReport::default();
    loop {
        if cancel.load(Ordering::Acquire) {
            break;
        }
        let index = next.fetch_add(1, Ordering::Relaxed);
        let Some(cube) = cubes.get(index) else { break };
        report.started += 1;
        // Arena, watches, trail, diario y actividades son privados del trabajador.
        let mut state = base.clone();
        state.stats = SolveStats::default();
        for &literal in cube {
            // Unidades con antecedente: los supuestos sobreviven al backjump.
            state.add_clause(&[literal], false);
        }
        let (outcome, stats) = solve_state(state, variables, assignment.to_vec(), cancel);
        report.stats.add_search(stats);
        match outcome {
            SearchOutcome::Sat(model) => {
                let mut result = winner.lock().expect("Winner mutex poisoned");
                if result.is_none() {
                    *result = Some(model);
                    // El modelo se publica antes de anunciar la cancelación.
                    cancel.store(true, Ordering::Release);
                }
                break;
            }
            SearchOutcome::Unsat => report.unsat += 1,
            SearchOutcome::Cancelled => {
                report.cancelled += 1;
                break;
            }
        }
    }
    report
}

pub fn solve_parallel(formula: Formula, assignment: Vec<i32>) -> Option<Vec<i32>> {
    let workers = std::thread::available_parallelism().map_or(1, |count| count.get());
    solve_parallel_with_threads(formula, assignment, workers).0
}

pub fn solve_parallel_with_threads(
    formula: Formula,
    assignment: Vec<i32>,
    requested_workers: usize,
) -> (Option<Vec<i32>>, ParallelStats) {
    let mut stats = ParallelStats::default();
    let Some((mut base, variables)) = prepare_formula(formula) else {
        return (None, stats);
    };
    let cancel = AtomicBool::new(false);
    match base.propagate(&cancel, usize::MAX) {
        Propagation::Conflict(conflict) => {
            base.stats.conflicts += 1;
            base.update_vsids(conflict);
            stats.total = base.stats;
            return (None, stats);
        }
        Propagation::Stable => {}
        Propagation::Cancelled | Propagation::Limit => unreachable!("Root propagation is unlimited"),
    }
    stats.total = base.stats;
    if base.choose_literal().is_none() {
        return (Some(model(&base, &variables, assignment)), stats);
    }
    let requested_workers = requested_workers.max(1);
    let cubes = make_cubes(&base, requested_workers, &mut stats);
    stats.cubes_generated = cubes.len();
    let workers = requested_workers.min(cubes.len());
    let next = AtomicUsize::new(0);
    let winner = Mutex::new(None);
    let (actual_workers, reports) = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for index in 0..workers {
            if cancel.load(Ordering::Acquire) {
                break;
            }
            let (base_ref, variables_ref, cubes_ref, assignment_ref) =
                (&base, variables.as_slice(), cubes.as_slice(), assignment.as_slice());
            let (next_ref, cancel_ref, winner_ref) = (&next, &cancel, &winner);
            let worker = move || conquer_worker(base_ref, variables_ref, cubes_ref,
                assignment_ref, next_ref, cancel_ref, winner_ref);
            match std::thread::Builder::new().name(format!("cdcl-{index}"))
                .spawn_scoped(scope, worker)
            {
                Ok(handle) => handles.push(handle),
                Err(_) => break,
            }
        }
        // Si el SO no puede crear hilos, el mismo consumidor completa la cola.
        if handles.is_empty() {
            return (1, vec![conquer_worker(&base, &variables, &cubes, &assignment,
                &next, &cancel, &winner)]);
        }
        let actual_workers = handles.len();
        let reports = handles.into_iter().map(|handle| {
            handle.join().expect("CDCL worker panicked")
        }).collect::<Vec<_>>();
        (actual_workers, reports)
    });
    stats.workers = actual_workers;
    for report in reports {
        stats.cubes_started += report.started;
        stats.cubes_unsat += report.unsat;
        stats.cubes_cancelled += report.cancelled;
        stats.total.add_search(report.stats);
    }
    let result = winner.into_inner().expect("Winner mutex poisoned");
    if result.is_none() {
        // Cancelar nunca demuestra UNSAT: se exige terminar todos los cubos.
        assert_eq!(stats.cubes_unsat, stats.cubes_generated, "Incomplete cube coverage");
    }
    (result, stats)
}


fn run_case(
    name: &str,
    n: usize,
    formula: Formula,
    expected_sat: bool,
    require_skip: bool,
    parallel: bool,
) -> ParallelStats {
    use std::hint::black_box;
    use std::time::Instant;
    let clause_count = formula.len();
    let input = formula.clone();
    let start = Instant::now();
    let (result, stats) = if parallel {
        let workers = std::thread::available_parallelism().map_or(1, |n| n.get());
        black_box(solve_parallel_with_threads(black_box(input), Vec::new(), workers))
    } else {
        let (result, total) = black_box(solve_cdcl(black_box(input), Vec::new()));
        (result, ParallelStats { total, ..ParallelStats::default() })
    };
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(result.is_some(), expected_sat);
    if let Some(model) = &result {
        let mut values = vec![0_i8; n + 1];
        for &literal in model {
            let variable = literal.unsigned_abs() as usize;
            let value = if literal > 0 { 1 } else { -1 };
            assert!(values[variable] == 0 || values[variable] == value);
            values[variable] = value;
        }
        assert!(formula.iter().all(|clause| clause.iter().any(|&literal| {
            values[literal.unsigned_abs() as usize] == if literal > 0 { 1 } else { -1 }
        })));
    }
    if require_skip && !parallel {
        assert!(stats.total.nonchronological_backjumps > 0);
    }
    let status = if result.is_some() { "SAT" } else { "UNSAT" };
    let mode = if parallel { "parallel" } else { "serial" };
    let total = stats.total;
    println!("{name},{mode},{n},{clause_count},{status},{elapsed_ms:.6},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        total.decisions, total.propagations, total.conflicts, total.learned_clauses,
        total.backjumps, total.nonchronological_backjumps,
        total.deleted_clauses, total.database_reductions,
        total.symmetry_rows, total.symmetry_columns, total.symmetry_clauses,
        stats.workers, stats.cubes_generated, stats.cubes_started,
        stats.cubes_unsat, stats.cubes_cancelled,
        stats.lookahead_propagations, stats.lookahead_conflicts);
    stats
}

fn pigeonhole_3sat(holes: usize) -> (usize, Formula) {
    let pigeons = holes + 1;
    let mut variables = pigeons * holes;
    let mut formula = Formula::new();
    for pigeon in 0..pigeons {
        let positive: Vec<i32> = (0..holes)
            .map(|hole| (pigeon * holes + hole + 1) as i32).collect();
        if positive.len() <= 3 {
            formula.push(positive);
        } else {
            variables += 1;
            let mut auxiliary = variables as i32;
            formula.push(vec![positive[0], positive[1], auxiliary]);
            for &literal in &positive[2..positive.len() - 2] {
                variables += 1;
                let next = variables as i32;
                formula.push(vec![-auxiliary, literal, next]);
                auxiliary = next;
            }
            formula.push(vec![-auxiliary, positive[positive.len() - 2], positive[positive.len() - 1]]);
        }
    }
    for hole in 0..holes {
        for first in 0..pigeons {
            for second in first + 1..pigeons {
                formula.push(vec![-((first * holes + hole + 1) as i32),
                    -((second * holes + hole + 1) as i32)]);
            }
        }
    }
    (variables, formula)
}

fn main() {
    println!("case,mode,n,clauses,status,elapsed_ms,decisions,propagations,conflicts,learned,backjumps,nonchronological,deleted,reductions,symmetry_rows,symmetry_columns,symmetry_clauses,workers,cubes_generated,cubes_started,cubes_unsat,cubes_cancelled,lookahead_propagations,lookahead_conflicts");
    for n in [10_i32, 100, 1000] {
        let mut chain: Formula = (1..n).rev().map(|v| vec![-v, v + 1]).collect();
        chain.push(vec![1]);
        let independent: Formula = (1..n - 1).step_by(3)
            .map(|v| vec![v, v + 1, v + 2]).collect();
        let rollback: Formula = (1..n - 2).step_by(3).flat_map(|v| {
            [vec![v, v + 1], vec![-v, v + 2], vec![-v, -(v + 2)]]
        }).collect();
        for (name, formula) in [
            ("reverse_chain", chain), ("independent", independent), ("rollback", rollback),
        ] {
            for parallel in [false, true] {
                run_case(name, n as usize, formula.clone(), true, false, parallel);
            }
        }
    }
    let mut nonchronological = vec![
        vec![-1, -3, 4], vec![-3, 5], vec![-4, -5, 6], vec![-6, 7], vec![-4, -7],
    ];
    nonchronological.extend((8..=18).map(|v| vec![1, v]));
    nonchronological.extend((19..=28).map(|v| vec![2, v]));
    nonchronological.extend((29..=34).map(|v| vec![3, v]));
    for parallel in [false, true] {
        run_case("nonchronological", 34, nonchronological.clone(), true, true, parallel);
    }
    for holes in [3, 4, 5, 6, 14, 19] {
        let (n, formula) = pigeonhole_3sat(holes);
        for parallel in [false, true] {
            let stats = run_case(&format!("pigeonhole_{}_{holes}", holes + 1),
                n, formula.clone(), false, false, parallel);
            assert_eq!(stats.total.symmetry_rows, holes + 1);
            assert_eq!(stats.total.symmetry_columns, holes);
            assert_eq!(stats.total.decisions, 0);
            assert_eq!(stats.total.conflicts, 1);
        }
    }
    // La tautología conserva SAT/UNSAT y ejercita la ruta CDCL general.
    let (n, mut stress) = pigeonhole_3sat(7);
    stress.push(vec![1, -1]);
    for parallel in [false, true] {
        run_case("cdcl_pigeonhole_8_7", n, stress.clone(), false, false, parallel);
    }
}
