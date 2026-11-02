// Fase 10: feedback Adam acotado en puntos estables de propagación CDCL.
// El trail fija coordenadas durante cada refinamiento; solo se guían variables libres.
// Fórmula original y buffers continuos se reutilizan; no se copian cláusulas aprendidas.
// Re-ignición modifica prioridades, conserva razones, watches y exactitud SAT/UNSAT.
#![forbid(unsafe_code)]
#![deny(warnings)]
pub mod gradient {
pub type Clause = Vec<i32>;

#[derive(Clone, Copy, Debug)]
pub struct GradientConfig {
    pub max_epochs: usize,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub loss_threshold: f64,
    pub restarts: usize,
    pub check_every: usize,
    pub seed: u64,
}

impl Default for GradientConfig {
    fn default() -> Self {
        Self {
            max_epochs: 1000, learning_rate: 0.03, beta1: 0.9, beta2: 0.999,
            epsilon: 1e-8, loss_threshold: 1e-4, restarts: 1, check_every: 25,
            seed: 0x243f_6a88_85a3_08d3,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GradientReport {
    pub model: Option<Vec<i32>>,
    pub variable_ids: Vec<u32>,
    pub continuous_values: Vec<f64>,
    pub epochs: usize,
    pub restarts: usize,
    pub loss: f64,
    pub gradient_l2: f64,
    pub violated_clauses: usize,
    pub learning_rate: f64,
}

pub struct CompiledFormula {
    variables: Vec<u32>,
    literals: Vec<usize>,
    offsets: Vec<usize>,
    max_width: usize,
}

impl CompiledFormula {
    pub fn new(formula: &[Clause]) -> Self {
        let mut variables: Vec<u32> = formula.iter().flatten().map(|&literal| {
            assert!(literal != 0 && literal != i32::MIN, "Invalid SAT literal");
            literal.unsigned_abs()
        }).collect();
        variables.sort_unstable();
        variables.dedup();
        assert!(variables.len() <= usize::MAX / 2);
        let mut literals = Vec::with_capacity(formula.iter().map(Vec::len).sum());
        let mut offsets = Vec::with_capacity(formula.len() + 1);
        let mut max_width = 0;
        offsets.push(0);
        for clause in formula {
            max_width = max_width.max(clause.len());
            for &literal in clause {
                let variable = variables.binary_search(&literal.unsigned_abs()).expect("Indexed variable");
                literals.push((variable << 1) | usize::from(literal < 0));
            }
            offsets.push(literals.len());
        }
        Self { variables, literals, offsets, max_width }
    }

    pub fn variable_ids(&self) -> &[u32] { &self.variables }

    pub fn violations(&self, model: &[bool]) -> usize {
        assert_eq!(model.len(), self.variables.len());
        self.offsets.windows(2).filter(|span| {
            !self.literals[span[0]..span[1]].iter().any(|&literal| {
                model[literal >> 1] == (literal & 1 == 0)
            })
        }).count()
    }

    pub fn verify(&self, model: &[bool]) -> bool { self.violations(model) == 0 }

    pub fn loss_and_gradient(&self, values: &[f64], gradient: &mut [f64]) -> f64 {
        assert_eq!(values.len(), self.variables.len());
        assert_eq!(gradient.len(), values.len());
        assert!(values.iter().all(|&value| value.is_finite() && (0.0..=1.0).contains(&value)));
        let mut prefix = vec![1.0; self.max_width + 1];
        self.evaluate(values, gradient, &mut prefix)
    }

    fn evaluate(&self, values: &[f64], gradient: &mut [f64], prefix: &mut [f64]) -> f64 {
        gradient.fill(0.0);
        let mut loss = 0.0;
        for span in self.offsets.windows(2) {
            let clause = &self.literals[span[0]..span[1]];
            prefix[0] = 1.0;
            for (index, &literal) in clause.iter().enumerate() {
                let value = values[literal >> 1];
                let factor = if literal & 1 == 0 { 1.0 - value } else { value };
                prefix[index + 1] = prefix[index] * factor;
            }
            loss += prefix[clause.len()];
            let mut suffix = 1.0;
            for (index, &literal) in clause.iter().enumerate().rev() {
                let variable = literal >> 1;
                let positive = literal & 1 == 0;
                gradient[variable] += (if positive { -1.0 } else { 1.0 }) * prefix[index] * suffix;
                suffix *= if positive { 1.0 - values[variable] } else { values[variable] };
            }
        }
        // Prefijos/sufijos: O(L+C), incluso con factores cero o variables repetidas.
        loss
    }

    fn signed_model(&self, model: &[bool]) -> Vec<i32> {
        self.variables.iter().zip(model).map(|(&variable, &value)| {
            if value { variable as i32 } else { -(variable as i32) }
        }).collect()
    }
}

/// Reuses the original clause arena and every optimizer buffer across CDCL callbacks.
/// Tensor and assignment positions follow `CompiledFormula::variable_ids()`.
pub struct OracleWorkspace {
    compiled: CompiledFormula,
    values: Vec<f64>,
    gradient: Vec<f64>,
    first_moment: Vec<f64>,
    second_moment: Vec<f64>,
    prefix: Vec<f64>,
}

impl OracleWorkspace {
    pub fn new(formula: &[Vec<i32>], initial: &[f64]) -> Self {
        let compiled = CompiledFormula::new(formula);
        let variables = compiled.variable_ids().len();
        assert_eq!(initial.len(), variables, "Initial tensor must match packed variables");
        assert!(initial.iter().all(|&value| value.is_finite() && (0.0..=1.0).contains(&value)));
        let prefix = vec![1.0; compiled.max_width + 1];
        Self {
            compiled,
            values: initial.to_vec(),
            gradient: vec![0.0; variables],
            first_moment: vec![0.0; variables],
            second_moment: vec![0.0; variables],
            prefix,
        }
    }

    /// Performs exactly `max_epochs` optimizer updates from the previous tensor.
    /// This warm-start trajectory requires `restarts == 1`; no random restart occurs.
    /// Trail assignments stay clamped; this tensor only supplies search guidance.
    pub fn reignite(&mut self, assignments: &[i8], config: GradientConfig) -> usize {
        assert_eq!(assignments.len(), self.values.len(), "Trail must match packed variables");
        assert!(assignments.iter().all(|&value| (-1..=1).contains(&value)));
        assert!(config.learning_rate.is_finite() && config.learning_rate > 0.0);
        assert!(config.beta1.is_finite() && (0.0..1.0).contains(&config.beta1));
        assert!(config.beta2.is_finite() && (0.0..1.0).contains(&config.beta2));
        assert!(config.epsilon.is_finite() && config.epsilon > 0.0);
        assert!(config.loss_threshold.is_finite() && config.loss_threshold >= 0.0);
        assert_eq!(config.restarts, 1, "Reignition is one warm-start trajectory");
        assert!(config.check_every > 0);

        self.values.iter_mut().zip(assignments).for_each(|(value, &assignment)| {
            if assignment != 0 {
                *value = if assignment > 0 { 1.0 } else { 0.0 };
            }
        });
        self.first_moment.fill(0.0);
        self.second_moment.fill(0.0);
        let mut beta1_power = 1.0;
        let mut beta2_power = 1.0;

        for _ in 0..config.max_epochs {
            // The shared gradient is complete before any free variable changes.
            // Original literals, prefix products and optimizer arrays are reused.
            self.compiled.evaluate(&self.values, &mut self.gradient, &mut self.prefix);
            beta1_power *= config.beta1;
            beta2_power *= config.beta2;
            let correction1 = 1.0 - beta1_power;
            let correction2 = 1.0 - beta2_power;
            self.values.iter_mut().zip(&self.gradient)
                .zip(self.first_moment.iter_mut()).zip(self.second_moment.iter_mut())
                .zip(assignments).for_each(|((((value, &g), first), second), &assignment)| {
                    if assignment != 0 {
                        *value = if assignment > 0 { 1.0 } else { 0.0 };
                    } else {
                        *first = config.beta1 * *first + (1.0 - config.beta1) * g;
                        *second = config.beta2 * *second + (1.0 - config.beta2) * g * g;
                        *value = (*value - config.learning_rate * (*first / correction1)
                            / ((*second / correction2).sqrt() + config.epsilon)).clamp(0.0, 1.0);
                    }
                });
        }
        config.max_epochs
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub fn solve_gradient(formula: &[Clause], config: GradientConfig) -> GradientReport {
    assert!(config.learning_rate.is_finite() && config.learning_rate > 0.0);
    assert!(config.beta1.is_finite() && (0.0..1.0).contains(&config.beta1));
    assert!(config.beta2.is_finite() && (0.0..1.0).contains(&config.beta2));
    assert!(config.epsilon.is_finite() && config.epsilon > 0.0);
    assert!(config.loss_threshold.is_finite() && config.loss_threshold >= 0.0);
    assert!(config.restarts > 0 && config.check_every > 0);
    let compiled = CompiledFormula::new(formula);
    let variables = compiled.variables.len();
    let trajectories = config.restarts.min(config.max_epochs.max(1));
    let mut values = vec![0.0; variables];
    let mut gradient = vec![0.0; variables];
    let mut first_moment = vec![0.0; variables];
    let mut second_moment = vec![0.0; variables];
    let mut prefix = vec![1.0; compiled.max_width + 1];
    let mut rounded = vec![false; variables];
    let mut epochs = 0;
    let mut final_loss = 0.0;
    let mut final_norm = 0.0;
    for attempt in 0..trajectories {
        let attempt_seed = mix64(config.seed ^ attempt as u64);
        for (value, &variable) in values.iter_mut().zip(&compiled.variables) {
            let bits = mix64(attempt_seed ^ u64::from(variable));
            *value = 0.1 + 0.8 * ((bits >> 11) as f64 / 9_007_199_254_740_992.0);
        }
        first_moment.fill(0.0);
        second_moment.fill(0.0);
        let mut beta1_power = 1.0;
        let mut beta2_power = 1.0;
        let budget = config.max_epochs / trajectories
            + usize::from(attempt < config.max_epochs % trajectories);
        for local_epoch in 0..=budget {
            final_loss = compiled.evaluate(&values, &mut gradient, &mut prefix);
            final_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt().abs();
            if final_loss <= config.loss_threshold || local_epoch % config.check_every == 0
                || local_epoch == budget
            {
                rounded.iter_mut().zip(&values).for_each(|(bit, &value)| *bit = value >= 0.5);
                // Loss cero por sí solo no certifica SAT en aritmética flotante.
                if compiled.verify(&rounded) {
                    return GradientReport {
                        model: Some(compiled.signed_model(&rounded)),
                        variable_ids: compiled.variables.clone(), continuous_values: values,
                        epochs, restarts: attempt, loss: final_loss, gradient_l2: final_norm,
                        violated_clauses: 0, learning_rate: config.learning_rate,
                    };
                }
            }
            if local_epoch == budget { break; }
            beta1_power *= config.beta1;
            beta2_power *= config.beta2;
            let correction1 = 1.0 - beta1_power;
            let correction2 = 1.0 - beta2_power;
            // Un gradiente común antes de actualizar; sin asignaciones por época.
            values.iter_mut().zip(&gradient).zip(first_moment.iter_mut())
                .zip(second_moment.iter_mut()).for_each(|(((value, &g), first), second)| {
                    *first = config.beta1 * *first + (1.0 - config.beta1) * g;
                    *second = config.beta2 * *second + (1.0 - config.beta2) * g * g;
                    *value = (*value - config.learning_rate * (*first / correction1)
                        / ((*second / correction2).sqrt() + config.epsilon)).clamp(0.0, 1.0);
                });
            epochs += 1;
        }
    }
    rounded.iter_mut().zip(&values).for_each(|(bit, &value)| *bit = value >= 0.5);
    let violated_clauses = compiled.violations(&rounded);
    GradientReport {
        model: if violated_clauses == 0 { Some(compiled.signed_model(&rounded)) } else { None },
        variable_ids: compiled.variables.clone(), continuous_values: values,
        epochs, restarts: trajectories - 1, loss: final_loss, gradient_l2: final_norm,
        violated_clauses, learning_rate: config.learning_rate,
    }
}

}
pub mod cdcl {
use std::collections::VecDeque;
use super::gradient::{GradientConfig, OracleWorkspace};
use std::time::Instant;
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
    pub reignitions: usize,
    pub reignition_epochs: usize,
    pub reignition_ms: f64,
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
        loop {
            while !self.units.is_empty() {
                let (literal, clause) = self.units.pop_front().expect("Queued unit");
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
                let mut replacement = None;
                for position in self.offsets[clause]..self.offsets[clause + 1] {
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
    let learned_limit = variables.len().saturating_mul(4).max(64);
    let feedback = config.filter(|configuration| configuration.interval != 0);
    let mut next_feedback = feedback.map_or(usize::MAX, |configuration| configuration.interval);
    let mut workspace: Option<OracleWorkspace> = None;
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
            state.stats.nonchronological_backjumps += usize::from(level > backjump_level + 1);
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
        if let Some(configuration) = feedback {
            if state.stats.conflicts >= next_feedback {
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
                // Estado estable: asignaciones lógicas fijas durante el refinamiento.
                let updates = oracle.reignite(&state.values, configuration.oracle);
                state.refresh_guidance(oracle.values(), activity_boost);
                state.stats.reignitions += 1;
                state.stats.reignition_epochs += updates;
                state.stats.reignition_ms += started.elapsed().as_secs_f64() * 1000.0;
                next_feedback = state.stats.conflicts.saturating_add(configuration.interval);
            }
        }
        let Some(literal) = state.choose_literal() else {
            let model = variables.iter().zip(&state.values).map(|(&id, &value)| {
                if value == 1 { id as i32 } else { -(id as i32) }
            }).collect();
            return (Some(model), state.stats);
        };
        state.trail_limits.push(state.trail.len());
        state.stats.decisions += 1;
        assert!(state.assign(literal, None), "Decision variable must be unassigned");
    }
}

}

pub use cdcl::SolveStats;
pub use gradient::{GradientConfig, GradientReport};
pub type Clause = Vec<i32>;
pub type Formula = Vec<Clause>;

#[derive(Clone, Copy, Debug)]
pub struct HybridConfig {
    pub oracle: GradientConfig,
    pub activity_boost: f64,
    // Cero desactiva el feedback sin cambiar el motor exacto.
    pub reignition_interval: usize,
    pub reignition_epochs: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            oracle: GradientConfig::default(), activity_boost: 4.0,
            reignition_interval: 500, reignition_epochs: 100,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HybridReport {
    pub model: Option<Vec<i32>>,
    pub oracle: GradientReport,
    pub cdcl: SolveStats,
    pub oracle_ms: f64,
    pub cdcl_ms: f64,
    pub used_cdcl: bool,
}

pub fn verify_model(formula: &[Clause], model: &[i32]) -> bool {
    use std::collections::BTreeMap;
    let mut values = BTreeMap::new();
    for &literal in model {
        if literal == 0 || literal == i32::MIN { return false; }
        if let Some(previous) = values.insert(literal.unsigned_abs(), literal > 0) {
            if previous != (literal > 0) { return false; }
        }
    }
    formula.iter().all(|clause| clause.iter().any(|&literal| {
        values.get(&literal.unsigned_abs()) == Some(&(literal > 0))
    }))
}

pub fn solve_hybrid(formula: &[Clause], config: HybridConfig) -> HybridReport {
    use std::time::Instant;
    assert!(config.activity_boost.is_finite() && config.activity_boost >= 0.0);
    let start = Instant::now();
    let oracle = gradient::solve_gradient(formula, config.oracle);
    let oracle_ms = start.elapsed().as_secs_f64() * 1000.0;
    if let Some(model) = &oracle.model {
        assert!(verify_model(formula, model), "Oracle model must satisfy original clauses");
        return HybridReport {
            model: Some(model.clone()), oracle, cdcl: SolveStats::default(),
            oracle_ms, cdcl_ms: 0.0, used_cdcl: false,
        };
    }
    let guidance: Vec<(u32, f64)> = oracle.variable_ids.iter().copied()
        .zip(oracle.continuous_values.iter().copied()).collect();
    let start = Instant::now();
    // La guía modifica prioridades; nunca se añade como unidades ni suposiciones.
    let feedback = cdcl::ReignitionConfig {
        interval: config.reignition_interval,
        oracle: GradientConfig {
            max_epochs: config.reignition_epochs, restarts: 1, ..config.oracle
        },
    };
    let (model, cdcl) = cdcl::solve_cdcl_with_reignition(
        formula, Some(&guidance), config.activity_boost, feedback,
    );
    let cdcl_ms = start.elapsed().as_secs_f64() * 1000.0;
    if let Some(model) = &model {
        assert!(verify_model(formula, model), "CDCL model must satisfy original clauses");
    }
    HybridReport { model, oracle, cdcl, oracle_ms, cdcl_ms, used_cdcl: true }
}

fn run_case(name: &str, n: usize, formula: Formula, expected_sat: bool, config: HybridConfig) {
    use std::hint::black_box;
    use std::time::Instant;
    let start = Instant::now();
    let (baseline_model, baseline) = black_box(cdcl::solve_cdcl(black_box(&formula), None, config.activity_boost));
    let baseline_ms = start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(baseline_model.is_some(), expected_sat);
    if let Some(model) = &baseline_model { assert!(verify_model(&formula, model)); }
    let start = Instant::now();
    let report = black_box(solve_hybrid(black_box(&formula), config));
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(report.model.is_some(), expected_sat);
    // Reutiliza exactamente el mismo tensor inicial para aislar el efecto del feedback.
    let (static_ms, static_stats) = if report.used_cdcl && config.reignition_interval != 0 {
        let prior: Vec<(u32, f64)> = report.oracle.variable_ids.iter().copied()
            .zip(report.oracle.continuous_values.iter().copied()).collect();
        let start = Instant::now();
        let (model, stats) = black_box(cdcl::solve_cdcl(black_box(&formula), Some(&prior), config.activity_boost));
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(model.is_some(), expected_sat);
        if let Some(model) = &model { assert!(verify_model(&formula, model)); }
        (elapsed, stats)
    } else { (report.cdcl_ms, report.cdcl) };
    let feedback_saved = static_stats.decisions as i128 - report.cdcl.decisions as i128;
    let feedback_reduction = if static_stats.decisions == 0 { "NA".to_owned() } else {
        format!("{:.4}", 100.0 * feedback_saved as f64 / static_stats.decisions as f64)
    };
    let saved = baseline.decisions as i128 - report.cdcl.decisions as i128;
    let reduction = if baseline.decisions == 0 { "NA".to_owned() } else {
        format!("{:.4}", 100.0 * saved as f64 / baseline.decisions as f64)
    };
    let status = if report.model.is_some() { "SAT" } else { "UNSAT" };
    let route = if report.used_cdcl { "oracle_cdcl" } else { "oracle_model" };
    // Diferencia real contra el mismo CDCL sin guía; valores negativos indican regresión.
    // reignition_ms forma parte de guided_cdcl_ms; no se suma de nuevo al total.
    println!("{name},{n},{},{status},{route},{baseline_ms:.6},{:.6},{:.6},{total_ms:.6},{},{:.12e},{},{},{saved},{reduction},{},{},{},{},{},{},{:.6},{static_ms:.6},{},{},{feedback_saved},{feedback_reduction}",
        formula.len(), report.oracle_ms, report.cdcl_ms,
        report.oracle.epochs, report.oracle.loss, baseline.decisions, report.cdcl.decisions,
        baseline.conflicts, report.cdcl.conflicts, report.cdcl.nonchronological_backjumps,
        report.cdcl.symmetry_clauses, report.cdcl.reignitions,
        report.cdcl.reignition_epochs, report.cdcl.reignition_ms,
        static_stats.decisions, static_stats.conflicts);
}

fn pigeonhole_3sat(pigeons: usize, holes: usize) -> (usize, Formula) {
    let mut variables = pigeons * holes;
    let mut formula = Formula::new();
    for pigeon in 0..pigeons {
        let positive: Vec<i32> = (0..holes)
            .map(|hole| (pigeon * holes + hole + 1) as i32).collect();
        if positive.len() <= 3 { formula.push(positive); } else {
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
    println!("case,n,clauses,status,route,baseline_cdcl_ms,oracle_ms,guided_cdcl_ms,total_ms,epochs,oracle_loss,baseline_decisions,guided_decisions,decisions_saved,decision_reduction_pct,baseline_conflicts,guided_conflicts,nonchronological,symmetry_clauses,reignitions,reignition_epochs,reignition_ms,static_guided_cdcl_ms,static_guided_decisions,static_guided_conflicts,feedback_decisions_saved,feedback_decision_reduction_pct");
    let config = HybridConfig::default();
    run_case("empty_formula", 0, vec![], true, config);
    run_case("contradictory_units", 1, vec![vec![1], vec![-1]], false, config);
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
        ] { run_case(name, n as usize, formula, true, config); }
    }
    let mut nonchronological = vec![
        vec![-1, -3, 4], vec![-3, 5], vec![-4, -5, 6], vec![-6, 7], vec![-4, -7],
    ];
    nonchronological.extend((8..=18).map(|v| vec![1, v]));
    nonchronological.extend((19..=28).map(|v| vec![2, v]));
    nonchronological.extend((29..=34).map(|v| vec![3, v]));
    run_case("nonchronological", 34, nonchronological, true, config);
    for holes in [3, 4, 5, 6, 14, 19] {
        let (n, formula) = pigeonhole_3sat(holes + 1, holes);
        run_case(&format!("pigeonhole_{}_{holes}", holes + 1), n, formula, false, config);
    }
    for (pigeons, holes) in [(3, 3), (4, 4), (6, 8)] {
        let (n, formula) = pigeonhole_3sat(pigeons, holes);
        run_case(&format!("pigeonhole_{pigeons}_{holes}"), n, formula, true, config);
    }
    // La tautología conserva SAT/UNSAT y fuerza la ruta general sin reconocimiento PHP.
    let (n, mut stress) = pigeonhole_3sat(8, 7);
    stress.push(vec![1, -1]);
    run_case("cdcl_pigeonhole_8_7", n, stress, false, config);
}
