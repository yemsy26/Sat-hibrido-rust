// Fase 12: chasis CDCL con arena lineal, watchers de 8 bytes, VMTF/EVSIDS y restarts.
// Los cubos son decisiones; cada worker conserva su aprendizaje entre cubos.
// El oraculo solo guia; SAT verifica la formula original. Compartir no garantiza mejoras.
// Medicion local 10/9: serial 65 ms (Fase 11: 408 ms); el enjambre de 16 workers no la mejora.
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
        self.reignite_internal(assignments, config, None).0
    }

    /// Polls before every Adam epoch and reports only completed optimizer updates.
    /// A cancelled call still clamps the current trail; cancellation proves no SAT status.
    pub fn reignite_with_cancel(
        &mut self,
        assignments: &[i8],
        config: GradientConfig,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> (usize, bool) {
        self.reignite_internal(assignments, config, Some(cancel))
    }

    fn reignite_internal(
        &mut self,
        assignments: &[i8],
        config: GradientConfig,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> (usize, bool) {
        use std::sync::atomic::Ordering;

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

        for completed in 0..config.max_epochs {
            if cancel.map_or(false, |signal| signal.load(Ordering::Acquire)) {
                return (completed, true);
            }
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
        (
            config.max_epochs,
            cancel.map_or(false, |signal| signal.load(Ordering::Acquire)),
        )
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
use std::collections::HashSet;
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
    pub restarts: usize,
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
        self.restarts += other.restarts;
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
    literals: [u32; 3],
    len: usize,
}
impl SmallClause {
    fn canonical(clause: &[u32]) -> Option<Self> {
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
    fn as_slice(&self) -> &[u32] { &self.literals[..self.len] }
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
            let variable = variables[(literal >> 1) as usize] as i32;
            if literal & 1 == 0 { variable } else { -variable }
        }).collect()).collect()
    }
}
enum SearchOutcome { Sat(Vec<i32>), Unsat, Cancelled }
#[derive(Clone, Copy)]
struct Sharing<'a> { pool: &'a CloudPool, interval: usize }
struct SearchControl<'a> {
    cube: &'a [u32],
    cancel: Option<&'a AtomicBool>,
    sharing: Option<Sharing<'a>>,
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
    activity_scale: f64,
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
            activity_scale: 1.0,
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
    fn import_level(&self, clause: SmallClause) -> Option<usize> {
        if let [literal] = *clause.as_slice() {
            // Una unidad solo queda resuelta si ya es verdadera en la raíz.
            let settled = self.value(literal) == 1 && self.levels[variable_of(literal)] == 0;
            return (!settled).then_some(0);
        }
        let mut nonfalse = None;
        let mut false_level = 0;
        for &literal in clause.as_slice() {
            if self.value(literal) == -1 {
                false_level = false_level.max(self.levels[variable_of(literal)] as usize);
            } else if nonfalse.replace(literal).is_some() {
                return None;
            }
        }
        match nonfalse {
            None => Some(false_level),
            Some(literal) if self.value(literal) == 0
                || self.levels[variable_of(literal)] as usize > false_level => Some(false_level),
            Some(_) => None,
        }
    }
    fn attach_import(&mut self, clause: SmallClause) -> Option<Conflict> {
        let mut literals = clause.literals;
        // No falsos primero; falsos de mayor nivel antes: sus watches se liberan tras un backjump.
        literals[..clause.len].sort_unstable_by_key(|&literal| {
            if self.value(literal) == -1 {
                (true, std::cmp::Reverse(self.levels[variable_of(literal)]))
            } else {
                (false, std::cmp::Reverse(0))
            }
        });
        self.stats.imported_clauses += 1;
        self.stats.learned_clauses += 1;
        let (first, second) = (literals[0], literals[1]);
        let (reason, conflict) = match clause.len {
            1 => {
                // `import_level` ya forzó el nivel 0 para toda unidad no resuelta.
                match self.value(first) {
                    0 => self.assign_root(first),
                    1 => {}
                    _ => self.root_unsat = true,
                }
                return None;
            }
            2 => {
                self.watch_binary(first, second);
                (BINARY | second, Conflict::Binary(first, second))
            }
            _ => {
                let reference = self.add_long(&literals, true, 3);
                (reference, Conflict::Long(reference))
            }
        };
        // Con el segundo literal no falso quedan dos vigilados válidos.
        if self.value(second) != -1 {
            return None;
        }
        match self.value(first) {
            0 => self.assign(first, reason),
            1 => {}
            _ => return Some(conflict),
        }
        None
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
    fn refresh_guidance(&mut self, tensor: &[f64], activity_boost: f64) {
        assert_eq!(tensor.len(), self.levels.len());
        for (variable, &value) in tensor.iter().enumerate() {
            assert!(value.is_finite() && (0.0..=1.0).contains(&value));
            if self.values[variable << 1] != 0 {
                continue;
            }
            self.phases[variable] = u8::from(value < 0.5);
            // El bonus se mide en incrementos EVSIDS vigentes; el orden VMTF no se toca.
            let bonus = (activity_boost.min(MAX_ACTIVITY) * (value - 0.5).abs()
                * self.activity_scale).min(MAX_ACTIVITY) * self.increment;
            self.scores[variable] = (self.scores[variable] + bonus).min(MAX_ACTIVITY);
            if self.heap.contains(variable as u32) {
                self.heap.increase(variable as u32, &self.scores);
            }
        }
        // No se modifica ningún valor, antecedente, watch, nivel ni entrada del trail.
    }
    fn assignment(&self) -> Vec<i8> {
        self.values.iter().step_by(2).copied().collect()
    }
    // Unidades raíz, binarias inline y ternarias de la arena.
    fn small_clauses(&self) -> HashSet<SmallClause> {
        let mut known = HashSet::new();
        let root = self.trail_limits.first().copied().unwrap_or(self.trail.len());
        known.extend(self.trail[..root].iter()
            .filter_map(|&literal| SmallClause::canonical(&[literal])));
        for (literal, watchers) in self.watches.iter().enumerate() {
            for watcher in watchers {
                if watcher.reference & BINARY != 0 && (literal as u32) < watcher.blocker {
                    known.extend(SmallClause::canonical(&[literal as u32, watcher.blocker]));
                }
            }
        }
        let mut reference = 0;
        while reference < self.arena.len() {
            let start = reference + HEADER;
            let size = self.arena[reference] as usize;
            if size == 3 {
                known.extend(SmallClause::canonical(&self.arena[start..start + size]));
            }
            reference = start + size;
        }
        known
    }
    // Candidatas: aprendidas con LBD > 2 que no son razón ni se usaron desde la última reducción.
    fn reduce_database(&mut self) {
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
        for &reference in &candidates[..remove_count] {
            self.arena[reference + 1] |= GARBAGE;
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
    // Literales y referencias comparten 31 bits; NO_REASON queda fuera de ambos rangos.
    assert!(variables.len() < 1 << 30, "Too many variables");
    // La canonización precede a cualquier supuesto de un cubo.
    let (symmetry_rows, symmetry_columns, symmetry_clauses) =
        inject_pigeonhole_symmetry(&mut formula, &variables);
    let mut state = State::new(variables.len(), SolveStats {
        symmetry_rows, symmetry_columns, symmetry_clauses,
        ..SolveStats::default()
    });
    let mut packed: Vec<u32> = Vec::new();
    for clause in &formula {
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
    }
    if state.root_unsat {
        state.stats.conflicts = 1;
    }
    // Referencia fija: el feedback no multiplica boosts anteriores.
    state.activity_scale = state.scores.iter().copied().fold(1.0_f64, f64::max);
    if let Some(prior) = guidance {
        assert!(prior.windows(2).all(|pair| pair[0].0 < pair[1].0));
        let scale = state.activity_scale;
        for &(id, value) in prior {
            assert!(value.is_finite() && (0.0..=1.0).contains(&value));
            if let Ok(variable) = variables.binary_search(&id) {
                state.phases[variable] = u8::from(value < 0.5);
                // Saturar antes del producto evita inf*0 y mantiene el ordenamiento finito.
                let bonus = (activity_boost.min(MAX_ACTIVITY)
                    * (value - 0.5).abs() * scale).min(MAX_ACTIVITY);
                state.scores[variable] = (state.scores[variable] + bonus).min(MAX_ACTIVITY);
            }
        }
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
fn extract_model(state: &State, variables: &[u32]) -> Vec<i32> {
    variables.iter().zip(state.values.iter().step_by(2)).map(|(&id, &value)| {
        if value == 1 { id as i32 } else { -(id as i32) }
    }).collect()
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
    // La cancelación se consulta una vez por conflicto o decisión, nunca dentro del BCP.
    let cancelled = || control.cancel.is_some_and(|flag| flag.load(Ordering::Acquire));
    if cancelled() {
        return (SearchOutcome::Cancelled, state.stats);
    }
    // El certificado raiz evita reprocesar una cola contradictoria y cubos posteriores.
    if state.root_unsat { return (SearchOutcome::Unsat, state.stats); }
    let feedback = config.filter(|configuration| configuration.interval != 0);
    let mut next_feedback = feedback.map_or(usize::MAX, |configuration| configuration.interval);
    let mut workspace: Option<OracleWorkspace> = None;
    let mut cursor = 0;
    let mut next_share = 0;
    let mut pending_conflict = None;
    let mut known = if control.sharing.is_some() { state.small_clauses() } else { HashSet::new() };
    loop {
        if cancelled() {
            return (SearchOutcome::Cancelled, state.stats);
        }
        if let Some(conflict) = pending_conflict.take().or_else(|| state.propagate()) {
            state.stats.conflicts += 1;
            let level = state.level();
            if level == 0 {
                // No se publica la cláusula vacía ni se confunde cancelación con UNSAT.
                state.root_unsat = true;
                return (SearchOutcome::Unsat, state.stats);
            }
            let (backjump_level, lbd) = state.analyze(conflict);
            if let Some(sharing) = control.sharing {
                if let Some(clause) = SmallClause::canonical(&state.learned) {
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
            state.learn(lbd);
            continue;
        }
        let reduced = state.conflicts >= state.next_cleanup;
        if reduced {
            state.reduce_database();
        }
        let feedback_due = feedback.is_some() && state.stats.conflicts >= next_feedback;
        if let Some(sharing) = control.sharing {
            if state.stats.conflicts >= next_share || feedback_due || reduced {
                let imports: Vec<SmallClause> = sharing.pool.harvest(&mut cursor).into_iter()
                    .filter_map(|entry| known.insert(entry.clause).then_some(entry.clause)).collect();
                next_share = state.stats.conflicts.saturating_add(sharing.interval.max(1));
                if !imports.is_empty() {
                    loop {
                        let current = state.level();
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
                    if state.root_unsat {
                        // Una unidad importada contradice la raíz: consecuencia global, UNSAT.
                        state.stats.conflicts += 1;
                        return (SearchOutcome::Unsat, state.stats);
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
                let assignment = state.assignment();
                let (updates, interrupted) = if let Some(cancel) = control.cancel {
                    oracle.reignite_with_cancel(&assignment, configuration.oracle, cancel)
                } else {
                    (oracle.reignite(&assignment, configuration.oracle), false)
                };
                state.stats.reignitions += 1;
                state.stats.reignition_epochs += updates;
                state.stats.reignition_ms += started.elapsed().as_secs_f64() * 1000.0;
                if interrupted { return (SearchOutcome::Cancelled, state.stats); }
                state.refresh_guidance(oracle.values(), activity_boost);
                next_feedback = state.stats.conflicts.saturating_add(configuration.interval);
            }
        }
        if state.level() < control.cube.len() {
            let literal = control.cube[state.level()];
            let value = state.value(literal);
            if value == -1 { return (SearchOutcome::Unsat, state.stats); }
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
            // El restart conserva los supuestos del cubo; en serie limpia hasta el nivel 0.
            state.restart(control.cube.len());
            continue;
        }
        let Some(literal) = state.decide() else {
            return (SearchOutcome::Sat(extract_model(state, variables)), state.stats);
        };
        state.trail_limits.push(state.trail.len());
        state.stats.decisions += 1;
        state.assign(literal, NO_REASON);
    }
}
fn make_cubes(base: &State, config: SwarmConfig) -> Vec<Vec<u32>> {
    let target = config.workers.max(1).saturating_mul(config.cubes_per_worker.max(1)).clamp(1, 1024);
    let mut candidates: Vec<usize> = (0..base.levels.len())
        .filter(|&variable| base.values[variable << 1] == 0).collect();
    candidates.sort_unstable_by(|&a, &b| {
        base.scores[b].total_cmp(&base.scores[a]).then_with(|| a.cmp(&b))
    });
    let mut depth = 0;
    while depth < candidates.len().min(10) && (1 << depth) < target { depth += 1; }
    candidates.truncate(depth);
    // Todas las combinaciones: la guía ordena cubos y nunca descarta ninguno.
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
    formula: &[Vec<i32>],
    guidance: Option<&[(u32, f64)]>,
    cubes: &[Vec<u32>],
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
        if !state.root_unsat {
            // Solo se llega al siguiente cubo tras CubeUNSAT en un fixpoint.
            debug_assert_eq!(state.head, state.trail.len());
            state.backjump(0);
        }
        // Clausulas, razones/unidades raiz, fases y actividades sobreviven al cubo.
        state.stats = SolveStats::default();
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
    if !base.root_unsat && base.propagate().is_some() {
        base.root_unsat = true;
        base.stats.conflicts += 1;
    }
    stats.total = base.stats;
    if base.root_unsat {
        return (None, stats);
    }
    if base.trail.len() == variables.len() {
        return (Some(extract_model(&base, &variables)), stats);
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
        state.assign(negative_x2, NO_REASON);
        assert!(state.propagate().is_none());
        assert_eq!(state.value(positive_x1), 1);
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
        assert_eq!(state.value(positive_x1), 1);
        assert_eq!(state.levels[0], 0);
        // La unidad importada vive en el trail raíz: no necesita cláusula antecedente.
        assert_eq!(state.reasons[0], NO_REASON);
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

#[derive(Clone, Debug)]
pub struct ParallelHybridReport {
    pub model: Option<Vec<i32>>,
    pub oracle: GradientReport,
    pub swarm: cdcl::SwarmStats,
    pub oracle_ms: f64,
    pub conquer_ms: f64,
}

pub fn solve_hybrid_parallel(
    formula: &[Clause], config: HybridConfig, mut swarm: cdcl::SwarmConfig,
) -> ParallelHybridReport {
    use std::time::Instant;
    assert!(config.activity_boost.is_finite() && config.activity_boost >= 0.0);
    let start = Instant::now();
    let oracle = gradient::solve_gradient(formula, config.oracle);
    let oracle_ms = start.elapsed().as_secs_f64() * 1000.0;
    if let Some(model) = &oracle.model {
        assert!(verify_model(formula, model));
        return ParallelHybridReport {
            model: Some(model.clone()), oracle, swarm: cdcl::SwarmStats::default(),
            oracle_ms, conquer_ms: 0.0,
        };
    }
    let guidance: Vec<(u32, f64)> = oracle.variable_ids.iter().copied()
        .zip(oracle.continuous_values.iter().copied()).collect();
    swarm.activity_boost = config.activity_boost;
    swarm.reignition = cdcl::ReignitionConfig {
        interval: config.reignition_interval,
        oracle: GradientConfig { max_epochs: config.reignition_epochs, restarts: 1, ..config.oracle },
    };
    let start = Instant::now();
    let (model, swarm) = cdcl::solve_swarm(formula, Some(&guidance), swarm);
    let conquer_ms = start.elapsed().as_secs_f64() * 1000.0;
    if let Some(model) = &model { assert!(verify_model(formula, model)); }
    ParallelHybridReport { model, oracle, swarm, oracle_ms, conquer_ms }
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

fn run_case(name: &str, n: usize, formula: Formula, expected_sat: bool, compare_sharing: bool) {
    use std::hint::black_box;
    use std::time::Instant;
    let config = HybridConfig::default();
    let start = Instant::now();
    let serial = black_box(solve_hybrid(black_box(&formula), config));
    let serial_ms = start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(serial.model.is_some(), expected_sat);
    if let Some(model) = &serial.model { assert!(verify_model(&formula, model)); }
    let serial_status = if serial.model.is_some() { "SAT" } else { "UNSAT" };
    println!("{name},serial,{n},{},{serial_status},{serial_ms:.6},{:.6},{:.6},{},{},{},{},{:.6},1,0,0,0,0,0,0,0,0,0,1.000000,0,0.0000",
        formula.len(), serial.oracle_ms, serial.cdcl_ms, serial.cdcl.decisions,
        serial.cdcl.conflicts, serial.cdcl.reignitions, serial.cdcl.reignition_epochs,
        serial.cdcl.reignition_ms);
    let mut no_share_decisions = None;
    for share_enabled in [false, true] {
        if !share_enabled && !compare_sharing { continue; }
        let swarm_config = cdcl::SwarmConfig { share_enabled, ..cdcl::SwarmConfig::default() };
        let start = Instant::now();
        let report = black_box(solve_hybrid_parallel(black_box(&formula), config, swarm_config));
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(report.model.is_some(), expected_sat);
        let stats = &report.swarm;
        let total = stats.total;
        if !share_enabled { no_share_decisions = Some(total.decisions); }
        let reduction = no_share_decisions.map_or_else(|| "NA".to_owned(), |before| {
            if before == 0 { "NA".to_owned() } else {
                format!("{:.4}", 100.0 * (before as f64 - total.decisions as f64) / before as f64)
            }
        });
        let status = if report.model.is_some() { "SAT" } else { "UNSAT" };
        let mode = if share_enabled { "parallel_shared" } else { "parallel_private" };
        let speedup = serial_ms / elapsed_ms.max(f64::MIN_POSITIVE);
        // Trabajo total real: no se presupone que compartir reduzca decisiones o tiempo.
        println!("{name},{mode},{n},{},{status},{elapsed_ms:.6},{:.6},{:.6},{},{},{},{},{:.6},{},{},{},{},{},{},{},{},{},{},{speedup:.6},{},{reduction}",
            formula.len(), report.oracle_ms, report.conquer_ms,
            total.decisions, total.conflicts, total.reignitions, total.reignition_epochs,
            total.reignition_ms, stats.workers, stats.cubes_generated, stats.cubes_started,
            stats.cubes_unsat, stats.cubes_cancelled,
            total.exported_clauses, total.imported_clauses, total.dropped_exports,
            total.import_backjumps, stats.shared_clauses.len(),
            no_share_decisions.map_or_else(|| "NA".to_owned(), |value| value.to_string()));
    }
}

fn pigeonhole_3sat(pigeons: usize, holes: usize) -> (usize, Formula) {
    let mut variables = pigeons * holes;
    let mut formula = Formula::new();
    for pigeon in 0..pigeons {
        let row: Vec<i32> = (0..holes).map(|hole| (pigeon * holes + hole + 1) as i32).collect();
        if row.len() <= 3 { formula.push(row); } else {
            variables += 1;
            let mut auxiliary = variables as i32;
            formula.push(vec![row[0], row[1], auxiliary]);
            for &literal in &row[2..row.len() - 2] {
                variables += 1;
                let next = variables as i32;
                formula.push(vec![-auxiliary, literal, next]);
                auxiliary = next;
            }
            formula.push(vec![-auxiliary, row[row.len() - 2], row[row.len() - 1]]);
        }
    }
    for hole in 0..holes {
        for first in 0..pigeons {
            for second in first + 1..pigeons {
                formula.push(vec![-((first * holes + hole + 1) as i32), -((second * holes + hole + 1) as i32)]);
            }
        }
    }
    (variables, formula)
}

fn main() {
    println!("case,mode,n,clauses,status,elapsed_ms,oracle_ms,conquer_ms,decisions,conflicts,reignitions,reignition_epochs,reignition_ms,workers,cubes_generated,cubes_started,cubes_unsat,cubes_cancelled,exports,imports,sharing_drops,import_backjumps,pool_clauses,speedup_vs_serial,private_decisions,decision_reduction_vs_private_pct");
    run_case("empty_formula", 0, vec![], true, false);
    run_case("contradictory_units", 1, vec![vec![1], vec![-1]], false, false);
    for n in [10_i32, 100, 1000] {
        let mut chain: Formula = (1..n).rev().map(|v| vec![-v, v + 1]).collect();
        chain.push(vec![1]);
        let independent: Formula = (1..n - 1).step_by(3).map(|v| vec![v, v + 1, v + 2]).collect();
        let rollback: Formula = (1..n - 2).step_by(3).flat_map(|v| {
            [vec![v, v + 1], vec![-v, v + 2], vec![-v, -(v + 2)]]
        }).collect();
        for (name, formula) in [("reverse_chain", chain), ("independent", independent), ("rollback", rollback)] {
            run_case(name, n as usize, formula, true, false);
        }
    }
    for holes in [3, 4, 5, 6, 14, 19] {
        let (n, formula) = pigeonhole_3sat(holes + 1, holes);
        run_case(&format!("pigeonhole_{}_{holes}", holes + 1), n, formula, false, false);
    }
    for (pigeons, holes) in [(3, 3), (4, 4), (6, 8)] {
        let (n, formula) = pigeonhole_3sat(pigeons, holes);
        run_case(&format!("pigeonhole_{pigeons}_{holes}"), n, formula, true, false);
    }
    // cargo run --release -- 10 ejecuta 11/10; el valor por defecto ejecuta 10/9.
    let holes = std::env::args().nth(1).map_or(9, |value| value.parse::<usize>().expect("Expected hole count"));
    assert!((7..=10).contains(&holes), "Stress hole count must be 7..=10");
    let (n, mut stress) = pigeonhole_3sat(holes + 1, holes);
    stress.push(vec![1, -1]);
    run_case(&format!("cdcl_pigeonhole_{}_{holes}", holes + 1), n, stress, false, true);
}
