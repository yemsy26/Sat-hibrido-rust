#[path = "hybrid_gradient.rs"]
pub mod gradient;
#[path = "hybrid_cdcl.rs"]
pub mod cdcl;

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
