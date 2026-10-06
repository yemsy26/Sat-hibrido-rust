#[path = "hybrid_proof.rs"]
pub mod proof;
#[path = "hybrid_parity.rs"]
pub mod parity;
#[path = "hybrid_symmetry.rs"]
pub mod symmetry;
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
    // Ruptura estática de simetrías antes del oráculo y del CDCL.
    pub symmetry: bool,
    // Extracción de paridad y eliminación gaussiana antes del oráculo y del CDCL.
    pub parity: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        let reignition = cdcl::ReignitionConfig::default();
        Self {
            oracle: reignition.oracle, reignition_interval: reignition.interval,
            symmetry: true, parity: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SymmetryStats {
    pub generators: usize,
    pub clauses: usize,
    pub ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ParityStats {
    pub equations: usize,
    pub rank: usize,
    pub units: usize,
    pub equivalences: usize,
    pub clauses: usize,
    pub auxiliaries: usize,
    pub refuted: bool,
    pub ms: f64,
}

#[derive(Clone, Debug)]
pub struct HybridReport {
    pub model: Option<Vec<i32>>,
    pub symmetry: SymmetryStats,
    pub parity: ParityStats,
    pub cdcl: SolveStats,
    pub cdcl_ms: f64,
}

#[derive(Clone, Debug)]
pub struct ParallelHybridReport {
    pub model: Option<Vec<i32>>,
    pub symmetry: SymmetryStats,
    pub parity: ParityStats,
    pub swarm: cdcl::SwarmStats,
    pub conquer_ms: f64,
}

// Capas de inprocessing. La fórmula que ven el oráculo y el CDCL lleva las consecuencias
// lineales y los predicados de ruptura; el modelo se sigue verificando contra la original.
// Ambas capas leen la fórmula original: la paridad solo añade consecuencias suyas, así que
// no altera las simetrías ni la validez de sus predicados.
fn preprocess<'f>(
    formula: &'f [Clause], config: HybridConfig, mut proof: Option<&mut (dyn proof::ProofLog + '_)>,
) -> (std::borrow::Cow<'f, [Clause]>, SymmetryStats, ParityStats) {
    use std::borrow::Cow;
    use std::time::Instant;
    let mut added: Vec<Clause> = Vec::new();
    let mut symmetry_stats = SymmetryStats::default();
    let mut break_symmetry = |added: &mut Vec<Clause>, proof: Option<&mut (dyn proof::ProofLog + '_)>| {
        let start = Instant::now();
        let breaking = symmetry::break_symmetries_with_proof(formula, proof);
        symmetry_stats = SymmetryStats {
            generators: breaking.generators.len(),
            clauses: breaking.clauses.len(),
            ms: start.elapsed().as_secs_f64() * 1000.0,
        };
        added.extend(breaking.clauses);
    };
    // Con una traza que admite dominancia la simetría se certifica primero, sobre la
    // fórmula intacta: lo que derive después la paridad no tiene por qué ser simétrico.
    // Con una traza que no la admite, no se rompe ninguna simetría.
    let certified = proof.is_some();
    if config.symmetry && certified {
        break_symmetry(&mut added, proof.as_deref_mut());
    }
    let mut parity_stats = ParityStats::default();
    if config.parity {
        let start = Instant::now();
        let reduction = parity::reduce_parity_with_proof(formula, proof.as_deref_mut());
        parity_stats = ParityStats {
            equations: reduction.equations,
            rank: reduction.rank,
            units: reduction.units,
            equivalences: reduction.equivalences,
            clauses: reduction.clauses.len(),
            auxiliaries: reduction.auxiliaries,
            refuted: reduction.refuted,
            ms: start.elapsed().as_secs_f64() * 1000.0,
        };
        added.extend(reduction.clauses);
    }
    // Sin traza, con 0 = 1 ya demostrado no queda nada que romper.
    if config.symmetry && !certified && !parity_stats.refuted {
        break_symmetry(&mut added, None);
    }
    if added.is_empty() {
        return (Cow::Borrowed(formula), symmetry_stats, parity_stats);
    }
    let mut extended = formula.to_vec();
    extended.extend(added);
    (Cow::Owned(extended), symmetry_stats, parity_stats)
}

// Las variables auxiliares de la paridad quedan por encima de todo identificador original.
fn restrict(model: Option<Vec<i32>>, formula: &[Clause], parity: &ParityStats) -> Option<Vec<i32>> {
    if parity.auxiliaries == 0 {
        return model;
    }
    let last = formula.iter().flatten().map(|literal| literal.unsigned_abs()).max().unwrap_or(0);
    model.map(|mut model| {
        model.retain(|literal| literal.unsigned_abs() <= last);
        model
    })
}

pub fn solve_hybrid_parallel(
    formula: &[Clause], config: HybridConfig, swarm: cdcl::SwarmConfig,
) -> ParallelHybridReport {
    solve_hybrid_parallel_with_progress(formula, config, swarm, None)
}

// El mismo recorrido, con un observador que puede leer el avance del enjambre mientras busca.
pub fn solve_hybrid_parallel_with_progress(
    formula: &[Clause], config: HybridConfig, swarm: cdcl::SwarmConfig,
    progress: Option<&cdcl::SwarmProgress>,
) -> ParallelHybridReport {
    solve_hybrid_parallel_with_filter(formula, config, swarm, progress, None)
}

// El mismo recorrido, con un prefiltro que ve los cubos del enjambre antes que los workers.
// El filtro contesta sobre `formula`, la original: lo que refuta ahí queda refutado también
// con las cláusulas que añaden las capas previas, y un modelo suyo se comprueba abajo como
// cualquier otro.
pub fn solve_hybrid_parallel_with_filter(
    formula: &[Clause], config: HybridConfig, mut swarm: cdcl::SwarmConfig,
    progress: Option<&cdcl::SwarmProgress>, filter: Option<cdcl::CubeFilter<'_>>,
) -> ParallelHybridReport {
    use std::time::Instant;
    swarm.reignition = cdcl::ReignitionConfig {
        interval: config.reignition_interval, oracle: config.oracle,
    };
    let (extended, symmetry, parity) = preprocess(formula, config, None);
    let start = Instant::now();
    let (model, swarm) = cdcl::solve_swarm_with_filter(&extended, swarm, progress, filter);
    let conquer_ms = start.elapsed().as_secs_f64() * 1000.0;
    let model = restrict(model, formula, &parity);
    if let Some(model) = &model { assert!(verify_model(formula, model)); }
    ParallelHybridReport { model, symmetry, parity, swarm, conquer_ms }
}

// GPU y cubos duros se coordinan en el llamador; la ruta estática sigue intacta.
pub fn solve_hybrid_parallel_with_recubing(
    formula: &[Clause], config: HybridConfig, mut swarm: cdcl::SwarmConfig,
    mut recube: cdcl::RecubeConfig, progress: Option<&cdcl::SwarmProgress>,
    filter: Option<cdcl::CubeFilter<'_>>,
) -> ParallelHybridReport {
    use std::time::Instant;
    swarm.reignition = cdcl::ReignitionConfig {
        interval: config.reignition_interval, oracle: config.oracle,
    };
    recube.max_split_variable = recube.max_split_variable.min(
        formula.iter().flatten().map(|literal| literal.unsigned_abs()).max().unwrap_or(0),
    );
    let (extended, symmetry, parity) = preprocess(formula, config, None);
    let start = Instant::now();
    let (model, swarm) = cdcl::solve_swarm_with_recubing(
        &extended, swarm, recube, progress, filter,
    );
    let conquer_ms = start.elapsed().as_secs_f64() * 1000.0;
    let model = restrict(model, formula, &parity);
    if let Some(model) = &model { assert!(verify_model(formula, model)); }
    ParallelHybridReport { model, symmetry, parity, swarm, conquer_ms }
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
    let (extended, symmetry, parity) = preprocess(formula, config, None);
    let start = Instant::now();
    // El oráculo vive dentro del CDCL y solo escribe fases; su modelo se verifica aquí.
    let feedback = cdcl::ReignitionConfig {
        interval: config.reignition_interval, oracle: config.oracle,
    };
    let (model, cdcl) = cdcl::solve_cdcl_with_reignition(&extended, feedback);
    let cdcl_ms = start.elapsed().as_secs_f64() * 1000.0;
    let model = restrict(model, formula, &parity);
    if let Some(model) = &model {
        assert!(verify_model(formula, model), "Model must satisfy original clauses");
    }
    HybridReport { model, symmetry, parity, cdcl, cdcl_ms }
}

// Como `solve_hybrid`, dejando en `proof` una traza de todo lo que se añade a la fórmula:
// la dominancia de cada simetría, las derivaciones de la paridad y cada cláusula aprendida
// o borrada por el CDCL. La capa de simetría solo participa si el formato tiene regla de
// dominancia (`PbProof`); con DRAT no rompe nada. Un resultado UNSAT termina con la
// cláusula vacía; un modelo no necesita traza.
pub fn solve_hybrid_with_proof(
    formula: &[Clause], config: HybridConfig, proof: &mut dyn proof::ProofLog,
) -> HybridReport {
    use std::time::Instant;
    let (extended, symmetry, parity) = preprocess(formula, config, Some(&mut *proof));
    let start = Instant::now();
    let feedback = cdcl::ReignitionConfig {
        interval: config.reignition_interval, oracle: config.oracle,
    };
    let (model, cdcl) = if parity.refuted {
        // La derivación de 0 = 1 ya cerró la traza con la cláusula vacía.
        (None, SolveStats { conflicts: 1, ..SolveStats::default() })
    } else {
        cdcl::solve_cdcl_with_proof(&extended, feedback, proof)
    };
    if formula.iter().any(Vec::is_empty) {
        proof.input_refuted();
    }
    let cdcl_ms = start.elapsed().as_secs_f64() * 1000.0;
    let model = restrict(model, formula, &parity);
    if let Some(model) = &model {
        assert!(verify_model(formula, model), "Model must satisfy original clauses");
    }
    HybridReport { model, symmetry, parity, cdcl, cdcl_ms }
}
