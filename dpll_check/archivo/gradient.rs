pub type Clause = Vec<i32>;
pub type Formula = Vec<Clause>;

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Sat(Vec<i32>),
    Unknown,
}

#[derive(Clone, Copy, Debug)]
pub struct GradientConfig {
    // Presupuesto TOTAL de actualizaciones entre todos los reinicios.
    pub max_epochs: usize,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub loss_threshold: f64,
    // Número máximo de trayectorias; cada una reinicia los momentos de Adam.
    pub restarts: usize,
    pub check_every: usize,
    pub seed: u64,
}

impl Default for GradientConfig {
    fn default() -> Self {
        Self {
            max_epochs: 5000,
            learning_rate: 0.03,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            loss_threshold: 1e-4,
            restarts: 4,
            check_every: 25,
            seed: 0x243f_6a88_85a3_08d3,
        }
    }
}

impl GradientConfig {
    fn validate(self) {
        assert!(self.learning_rate.is_finite() && self.learning_rate > 0.0);
        assert!(self.beta1.is_finite() && (0.0..1.0).contains(&self.beta1));
        assert!(self.beta2.is_finite() && (0.0..1.0).contains(&self.beta2));
        assert!(self.epsilon.is_finite() && self.epsilon > 0.0);
        assert!(self.loss_threshold.is_finite() && self.loss_threshold >= 0.0);
        assert!(self.restarts > 0 && self.check_every > 0);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GradientReport {
    pub outcome: Outcome,
    pub epochs: usize,
    // Reinicios efectivamente realizados después de la primera trayectoria.
    pub restarts: usize,
    pub loss: f64,
    pub gradient_l2: f64,
    pub violated_clauses: usize,
    pub learning_rate: f64,
    // Orden correspondiente a CompiledFormula::variable_ids().
    pub continuous_values: Vec<f64>,
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
                let variable = variables.binary_search(&literal.unsigned_abs())
                    .expect("Indexed variable");
                literals.push((variable << 1) | usize::from(literal < 0));
            }
            offsets.push(literals.len());
        }
        Self { variables, literals, offsets, max_width }
    }

    pub fn variable_ids(&self) -> &[u32] {
        &self.variables
    }

    pub fn verify(&self, model: &[bool]) -> bool {
        self.violations(model) == 0
    }

    pub fn violations(&self, model: &[bool]) -> usize {
        assert_eq!(model.len(), self.variables.len());
        self.offsets.windows(2).filter(|span| {
            !self.literals[span[0]..span[1]].iter().any(|&literal| {
                model[literal >> 1] == (literal & 1 == 0)
            })
        }).count()
    }

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
            // Una cláusula vacía tiene penalización 1 y gradiente cero.
            loss += prefix[clause.len()];
            let mut suffix = 1.0;
            for (index, &literal) in clause.iter().enumerate().rev() {
                let variable = literal >> 1;
                let positive = literal & 1 == 0;
                let derivative = if positive { -1.0 } else { 1.0 };
                gradient[variable] += derivative * prefix[index] * suffix;
                let factor = if positive { 1.0 - values[variable] } else { values[variable] };
                suffix *= factor;
            }
        }
        // Prefijos/sufijos: O(literales), sin divisiones por factores que pueden ser cero.
        loss
    }

    fn signed_model(&self, model: &[bool]) -> Vec<i32> {
        self.variables.iter().zip(model).map(|(&variable, &value)| {
            if value { variable as i32 } else { -(variable as i32) }
        }).collect()
    }
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub fn solve_gradient(formula: &[Clause], config: GradientConfig) -> GradientReport {
    config.validate();
    let compiled = CompiledFormula::new(formula);
    let variables = compiled.variable_ids().len();
    let trajectories = config.restarts.min(config.max_epochs.max(1));
    let mut values = vec![0.0; variables];
    let mut gradient = vec![0.0; variables];
    let mut first_moment = vec![0.0; variables];
    let mut second_moment = vec![0.0; variables];
    let mut prefix = vec![1.0; compiled.max_width + 1];
    let mut rounded = vec![false; variables];
    let mut best_values = vec![0.0; variables];
    let mut best_loss = f64::INFINITY;
    let mut best_gradient_l2 = 0.0;
    let mut epochs = 0;

    for attempt in 0..trajectories {
        // Inicialización determinista asimétrica: evita imponer v=0.5 a todas las variables.
        let attempt_seed = mix64(config.seed ^ attempt as u64);
        for (value, &variable) in values.iter_mut().zip(compiled.variable_ids()) {
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
            let loss = compiled.evaluate(&values, &mut gradient, &mut prefix);
            let gradient_l2 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt().abs();
            if loss < best_loss {
                best_loss = loss;
                best_values.copy_from_slice(&values);
                best_gradient_l2 = gradient_l2;
            }
            if loss <= config.loss_threshold
                || local_epoch % config.check_every == 0
                || local_epoch == budget
            {
                rounded.iter_mut().zip(&values).for_each(|(bit, &value)| *bit = value >= 0.5);
                // La pérdida numérica nunca sustituye esta comprobación Booleana exacta.
                if compiled.verify(&rounded) {
                    return GradientReport {
                        outcome: Outcome::Sat(compiled.signed_model(&rounded)),
                        epochs, restarts: attempt, loss, gradient_l2,
                        violated_clauses: 0, learning_rate: config.learning_rate,
                        continuous_values: values,
                    };
                }
            }
            if local_epoch == budget {
                break;
            }
            beta1_power *= config.beta1;
            beta2_power *= config.beta2;
            let correction1 = 1.0 - beta1_power;
            let correction2 = 1.0 - beta2_power;
            // Actualización simultánea sobre buffers contiguos; sin asignaciones por época.
            values.iter_mut().zip(&gradient).zip(first_moment.iter_mut())
                .zip(second_moment.iter_mut()).for_each(|(((value, &g), first), second)| {
                    *first = config.beta1 * *first + (1.0 - config.beta1) * g;
                    *second = config.beta2 * *second + (1.0 - config.beta2) * g * g;
                    let corrected_first = *first / correction1;
                    let corrected_second = *second / correction2;
                    *value = (*value - config.learning_rate * corrected_first
                        / (corrected_second.sqrt() + config.epsilon)).clamp(0.0, 1.0);
                });
            epochs += 1;
        }
    }
    rounded.iter_mut().zip(&best_values).for_each(|(bit, &value)| *bit = value >= 0.5);
    let violated_clauses = compiled.violations(&rounded);
    let outcome = if violated_clauses == 0 {
        Outcome::Sat(compiled.signed_model(&rounded))
    } else {
        Outcome::Unknown
    };
    GradientReport {
        outcome,
        epochs,
        restarts: trajectories - 1,
        loss: best_loss,
        gradient_l2: best_gradient_l2,
        violated_clauses,
        learning_rate: config.learning_rate,
        continuous_values: best_values,
    }
}
