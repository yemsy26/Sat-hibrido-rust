#[path = "hybrid_proof.rs"]
pub mod proof;
#[path = "hybrid_parity.rs"]
pub mod parity;
#[path = "hybrid_congruence.rs"]
pub mod congruence;
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
    // Equivalencias y cierre por congruencia de puertas; sustituye cada clase por un literal.
    pub congruence: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        let reignition = cdcl::ReignitionConfig::default();
        Self {
            oracle: reignition.oracle, reignition_interval: reignition.interval,
            symmetry: true, parity: true, congruence: true,
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
    pub refuted: bool,
    pub ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CongruenceStats {
    pub ands: usize,
    pub xors: usize,
    pub ites: usize,
    pub equivalences: usize,
    pub units: usize,
    pub clauses_removed: usize,
    pub refuted: bool,
    pub ms: f64,
}

#[derive(Clone, Debug)]
pub struct HybridReport {
    pub model: Option<Vec<i32>>,
    // Se paró a petición de fuera: sin modelo, esto no dice nada de la fórmula.
    pub stopped: bool,
    pub symmetry: SymmetryStats,
    pub parity: ParityStats,
    pub congruence: CongruenceStats,
    pub cdcl: SolveStats,
    pub cdcl_ms: f64,
}

#[derive(Clone, Debug)]
pub struct ParallelHybridReport {
    pub model: Option<Vec<i32>>,
    pub symmetry: SymmetryStats,
    pub parity: ParityStats,
    pub congruence: CongruenceStats,
    pub swarm: cdcl::SwarmStats,
    pub conquer_ms: f64,
}

// Lo que dejan las capas previas: la fórmula para el CDCL y cómo volver a la original.
struct Prepared<'f> {
    formula: std::borrow::Cow<'f, [Clause]>,
    symmetry: SymmetryStats,
    parity: ParityStats,
    congruence: CongruenceStats,
    closure: congruence::Congruence,
}

impl Prepared<'_> {
    // Una capa ya cerró la traza con la cláusula vacía.
    fn refuted(&self) -> bool {
        self.parity.refuted || self.congruence.refuted
    }
}

// Capas de inprocessing. La fórmula que ven el oráculo y el CDCL lleva las consecuencias
// lineales y los predicados de ruptura; el modelo se sigue verificando contra la original.
// Ambas capas leen la fórmula original: la paridad solo añade consecuencias suyas, así que
// no altera las simetrías ni la validez de sus predicados. La congruencia lee la fórmula y lo
// que añadió la paridad; su sustitución va al final, sobre todo lo demás, porque toda
// equivalencia que encuentra se cumple en cada modelo de la fórmula.
fn preprocess<'f>(
    formula: &'f [Clause], config: HybridConfig, mut proof: Option<&mut (dyn proof::ProofLog + '_)>,
    stop: Option<&std::sync::atomic::AtomicBool>,
) -> Prepared<'f> {
    use std::borrow::Cow;
    use std::time::Instant;
    let mut added: Vec<Clause> = Vec::new();
    let mut symmetry_stats = SymmetryStats::default();
    // Las dos capas crean variables: cada una numera por encima de lo que dejó la anterior.
    let last = formula.iter().flatten().map(|literal| literal.unsigned_abs()).max().unwrap_or(0);
    let mut next_free = last + 1;
    let mut break_symmetry = |added: &mut Vec<Clause>, next_free: &mut u32, proof: Option<&mut (dyn proof::ProofLog + '_)>| {
        let start = Instant::now();
        let breaking = symmetry::break_symmetries_from(formula, *next_free, proof, stop);
        *next_free += breaking.auxiliaries as u32;
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
        break_symmetry(&mut added, &mut next_free, proof.as_deref_mut());
    }
    let mut parity_stats = ParityStats::default();
    let mut parity_clauses: Vec<Clause> = Vec::new();
    if config.parity {
        let start = Instant::now();
        let reduction = parity::reduce_parity_above(formula, next_free, proof.as_deref_mut());
        parity_stats = ParityStats {
            equations: reduction.equations,
            rank: reduction.rank,
            units: reduction.units,
            equivalences: reduction.equivalences,
            clauses: reduction.clauses.len(),
            refuted: reduction.refuted,
            ms: start.elapsed().as_secs_f64() * 1000.0,
        };
        parity_clauses = reduction.clauses;
    }
    let mut congruence_stats = CongruenceStats::default();
    let mut closure = congruence::Congruence::default();
    if config.congruence && !parity_stats.refuted {
        let start = Instant::now();
        closure = congruence::close(&[formula, &parity_clauses], proof.as_deref_mut(), stop);
        congruence_stats = CongruenceStats {
            ands: closure.ands, xors: closure.xors, ites: closure.ites,
            equivalences: closure.equivalences, units: closure.units,
            clauses_removed: 0, refuted: closure.refuted,
            ms: start.elapsed().as_secs_f64() * 1000.0,
        };
    }
    added.extend(parity_clauses);
    if closure.refuted {
        added.push(Vec::new());
    }
    // Sin traza, con 0 = 1 ya demostrado no queda nada que romper.
    if config.symmetry && !certified && !parity_stats.refuted && !closure.refuted {
        break_symmetry(&mut added, &mut next_free, None);
    }
    if added.is_empty() && closure.is_identity() {
        return Prepared {
            formula: Cow::Borrowed(formula), symmetry: symmetry_stats, parity: parity_stats,
            congruence: congruence_stats, closure,
        };
    }
    let mut extended = formula.to_vec();
    extended.extend(added);
    if !closure.is_identity() {
        let start = Instant::now();
        let before = extended.len();
        extended = closure.rewrite(&extended, proof.as_deref_mut());
        congruence_stats.clauses_removed = before - extended.len();
        congruence_stats.ms += start.elapsed().as_secs_f64() * 1000.0;
        if extended.iter().any(Vec::is_empty) {
            // Todos sus literales están fijados a falso: la vacía es RUP.
            congruence_stats.refuted = true;
        }
    }
    Prepared {
        formula: Cow::Owned(extended), symmetry: symmetry_stats, parity: parity_stats,
        congruence: congruence_stats, closure,
    }
}

// Las auxiliares de las capas quedan por encima de todo identificador original; las variables
// sustituidas vuelven con el valor de su representante.
fn restrict(model: Option<Vec<i32>>, formula: &[Clause], closure: &congruence::Congruence) -> Option<Vec<i32>> {
    let last = formula.iter().flatten().map(|literal| literal.unsigned_abs()).max().unwrap_or(0);
    model.map(|mut model| {
        model.retain(|literal| literal.unsigned_abs() <= last);
        closure.extend(model, formula)
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
    let prepared = preprocess(formula, config, None, progress.map(|observer| &observer.stop));
    let start = Instant::now();
    let (model, swarm) = cdcl::solve_swarm_with_filter(&prepared.formula, swarm, progress, filter);
    let conquer_ms = start.elapsed().as_secs_f64() * 1000.0;
    let model = restrict(model, formula, &prepared.closure);
    if let Some(model) = &model { assert!(verify_model(formula, model)); }
    let Prepared { symmetry, parity, congruence, .. } = prepared;
    ParallelHybridReport { model, symmetry, parity, congruence, swarm, conquer_ms }
}

// Como `solve_hybrid_parallel_with_filter`, dejando en `proof` la traza de las capas previas y de
// todos los workers. Un resultado UNSAT termina con la cláusula vacía.
pub fn solve_hybrid_parallel_with_proof(
    formula: &[Clause], config: HybridConfig, mut swarm: cdcl::SwarmConfig,
    progress: Option<&cdcl::SwarmProgress>, filter: Option<cdcl::CubeFilter<'_>>,
    proof: &mut (dyn proof::ProofLog + Send),
) -> ParallelHybridReport {
    use std::time::Instant;
    swarm.reignition = cdcl::ReignitionConfig {
        interval: config.reignition_interval, oracle: config.oracle,
    };
    let prepared = preprocess(
        formula, config, Some(&mut *proof as &mut dyn proof::ProofLog), progress.map(|observer| &observer.stop),
    );
    let start = Instant::now();
    let (model, swarm) = if prepared.refuted() {
        (None, cdcl::SwarmStats::default())
    } else {
        cdcl::solve_swarm_with_proof(&prepared.formula, swarm, progress, filter, proof)
    };
    if formula.iter().any(Vec::is_empty) {
        proof.input_refuted();
    }
    let conquer_ms = start.elapsed().as_secs_f64() * 1000.0;
    let model = restrict(model, formula, &prepared.closure);
    if let Some(model) = &model { assert!(verify_model(formula, model)); }
    let Prepared { symmetry, parity, congruence, .. } = prepared;
    ParallelHybridReport { model, symmetry, parity, congruence, swarm, conquer_ms }
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
    let prepared = preprocess(formula, config, None, progress.map(|observer| &observer.stop));
    let start = Instant::now();
    let (model, swarm) = cdcl::solve_swarm_with_recubing(
        &prepared.formula, swarm, recube, progress, filter,
    );
    let conquer_ms = start.elapsed().as_secs_f64() * 1000.0;
    let model = restrict(model, formula, &prepared.closure);
    if let Some(model) = &model { assert!(verify_model(formula, model)); }
    let Prepared { symmetry, parity, congruence, .. } = prepared;
    ParallelHybridReport { model, symmetry, parity, congruence, swarm, conquer_ms }
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
    solve_hybrid_observed(formula, config, None, None)
}

// Como `solve_hybrid`, dejando en `proof` una traza de todo lo que se añade a la fórmula:
// la dominancia de cada simetría, las derivaciones de la paridad y cada cláusula aprendida
// o borrada por el CDCL. La capa de simetría solo participa si el formato tiene regla de
// dominancia (`PbProof`); con DRAT no rompe nada. Un resultado UNSAT termina con la
// cláusula vacía; un modelo no necesita traza.
pub fn solve_hybrid_with_proof(
    formula: &[Clause], config: HybridConfig, proof: &mut dyn proof::ProofLog,
) -> HybridReport {
    solve_hybrid_observed(formula, config, Some(proof), None)
}

// El mismo recorrido, con traza o sin ella, y un observador que lee los contadores y puede
// pedir la parada por `progress.stop`.
pub fn solve_hybrid_observed(
    formula: &[Clause], config: HybridConfig, mut proof: Option<&mut (dyn proof::ProofLog + '_)>,
    progress: Option<&cdcl::SwarmProgress>,
) -> HybridReport {
    use std::time::Instant;
    let prepared = preprocess(formula, config, proof.as_deref_mut(), progress.map(|observer| &observer.stop));
    let start = Instant::now();
    // El oráculo vive dentro del CDCL y solo escribe fases; su modelo se verifica aquí.
    let feedback = cdcl::ReignitionConfig {
        interval: config.reignition_interval, oracle: config.oracle,
    };
    let (model, cdcl, stopped) = if prepared.refuted() && proof.is_some() {
        // La derivación de la contradicción ya cerró la traza con la cláusula vacía.
        (None, SolveStats { conflicts: 1, ..SolveStats::default() }, false)
    } else {
        cdcl::solve_cdcl_observed(&prepared.formula, feedback, proof.as_deref_mut(), progress)
    };
    if let Some(proof) = proof.filter(|_| formula.iter().any(Vec::is_empty)) {
        proof.input_refuted();
    }
    let cdcl_ms = start.elapsed().as_secs_f64() * 1000.0;
    let model = restrict(model, formula, &prepared.closure);
    if let Some(model) = &model {
        assert!(verify_model(formula, model), "Model must satisfy original clauses");
    }
    let Prepared { symmetry, parity, congruence, .. } = prepared;
    HybridReport { model, stopped, symmetry, parity, congruence, cdcl, cdcl_ms }
}
