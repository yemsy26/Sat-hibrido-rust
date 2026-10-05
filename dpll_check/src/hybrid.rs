#[path = "hybrid_cdcl.rs"]
pub mod cdcl;

pub use cdcl::{OracleConfig, SolveStats};
pub type Clause = Vec<i32>;
pub type Formula = Vec<Clause>;

#[derive(Clone, Copy, Debug)]
pub struct HybridConfig {
    pub oracle: OracleConfig,
    // Cero desactiva el oráculo sin cambiar el motor exacto.
    pub reignition_interval: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        let reignition = cdcl::ReignitionConfig::default();
        Self { oracle: reignition.oracle, reignition_interval: reignition.interval }
    }
}

#[derive(Clone, Debug)]
pub struct HybridReport {
    pub model: Option<Vec<i32>>,
    pub cdcl: SolveStats,
    pub cdcl_ms: f64,
}

#[derive(Clone, Debug)]
pub struct ParallelHybridReport {
    pub model: Option<Vec<i32>>,
    pub swarm: cdcl::SwarmStats,
    pub conquer_ms: f64,
}

pub fn solve_hybrid_parallel(
    formula: &[Clause], config: HybridConfig, mut swarm: cdcl::SwarmConfig,
) -> ParallelHybridReport {
    use std::time::Instant;
    swarm.reignition = cdcl::ReignitionConfig {
        interval: config.reignition_interval, oracle: config.oracle,
    };
    let start = Instant::now();
    let (model, swarm) = cdcl::solve_swarm(formula, swarm);
    let conquer_ms = start.elapsed().as_secs_f64() * 1000.0;
    if let Some(model) = &model { assert!(verify_model(formula, model)); }
    ParallelHybridReport { model, swarm, conquer_ms }
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
    let start = Instant::now();
    // El oráculo vive dentro del CDCL y solo escribe fases; su modelo se verifica aquí.
    let feedback = cdcl::ReignitionConfig {
        interval: config.reignition_interval, oracle: config.oracle,
    };
    let (model, cdcl) = cdcl::solve_cdcl_with_reignition(formula, feedback);
    let cdcl_ms = start.elapsed().as_secs_f64() * 1000.0;
    if let Some(model) = &model {
        assert!(verify_model(formula, model), "Model must satisfy original clauses");
    }
    HybridReport { model, cdcl, cdcl_ms }
}
