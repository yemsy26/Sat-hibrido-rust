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

pub fn solve_cdcl(
    mut formula: Formula,
    mut assignment: Vec<i32>,
) -> (Option<Vec<i32>>, SolveStats) {
    use std::collections::VecDeque;

    const DECAY: f64 = 0.95;
    const CLEANUP_INTERVAL: usize = 64;

    #[derive(Clone, Copy)]
    struct TrailEntry {
        literal: usize,
        level: usize,
        reason: Option<usize>,
    }

    enum Undo {
        Assigned(usize),
    }

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

        fn propagate(&mut self) -> Option<usize> {
            loop {
                while let Some((literal, clause)) = self.units.pop_front() {
                    if !self.assign(literal, Some(clause)) {
                        return Some(clause);
                    }
                }
                if self.propagation_head == self.trail.len() {
                    return None;
                }
                let false_literal = self.trail[self.propagation_head].literal ^ 1;
                self.propagation_head += 1;
                let mut index = 0;
                while index < self.watches[false_literal].len() {
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
                    let replacement = (self.offsets[clause]..self.offsets[clause + 1])
                        .find(|&position| {
                            position != positions[0]
                                && position != positions[1]
                                && self.value(self.literals[position]) != -1
                        });
                    if let Some(position) = replacement {
                        self.watched_positions[clause][slot] = position;
                        self.watches[false_literal].swap_remove(index);
                        self.watches[self.literals[position]].push(clause);
                        continue;
                    }
                    if other_value == -1 {
                        return Some(clause);
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

    if formula.iter().any(Vec::is_empty) {
        return (None, SolveStats::default());
    }
    if formula.is_empty() {
        return (Some(assignment), SolveStats::default());
    }
    let mut variables: Vec<u32> = formula.iter().flatten().map(|&literal| {
        assert!(literal != 0 && literal != i32::MIN, "Invalid SAT literal");
        literal.unsigned_abs()
    }).collect();
    variables.sort_unstable();
    variables.dedup();
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
            symmetry_rows,
            symmetry_columns,
            symmetry_clauses,
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
    let learned_limit = variables.len().saturating_mul(4).max(64);

    loop {
        if let Some(conflict) = state.propagate() {
            state.stats.conflicts += 1;
            state.update_vsids(conflict);
            let level = state.trail_limits.len();
            if level == 0 {
                return (None, state.stats);
            }
            let (learned, backjump_level) = state.analyze_conflict(conflict);
            state.stats.backjumps += 1;
            if level > backjump_level + 1 {
                state.stats.nonchronological_backjumps += 1;
            }
            state.backjump(backjump_level);
            let clause = state.add_clause(&learned, true);
            assert!(state.assign(learned[0], Some(clause)), "Learned clause must assert");
            continue;
        }

        if state.stats.conflicts >= state.next_cleanup {
            if state.learned_count > learned_limit {
                state.reduce_database();
            }
            state.next_cleanup = state.stats.conflicts.saturating_add(CLEANUP_INTERVAL);
        }
        let Some(literal) = state.choose_literal() else {
            // El prefijo recibido conserva la semántica del solver original.
            assignment.extend(state.trail.iter().map(|entry| {
                let variable = variables[entry.literal >> 1] as i32;
                if entry.literal & 1 == 0 { variable } else { -variable }
            }));
            return (Some(assignment), state.stats);
        };
        state.trail_limits.push(state.trail.len());
        state.stats.decisions += 1;
        assert!(state.assign(literal, None), "Decision variable must be unassigned");
    }
}
