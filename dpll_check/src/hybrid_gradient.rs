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
