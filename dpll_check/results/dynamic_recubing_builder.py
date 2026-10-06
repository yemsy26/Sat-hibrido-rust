from pathlib import Path
import difflib

root = Path(__file__).resolve().parent.parent
review = Path((root / 'results/recube-review-path.txt').read_text().strip())

def replace_once(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:100])
    return text.replace(old, new, 1)

cdcl = (root / 'src/hybrid_cdcl.rs').read_text(encoding='utf-8')
cdcl = replace_once(cdcl, '#[derive(Clone, Debug, Default)]\npub struct SwarmStats', '''#[derive(Clone, Copy, Debug)]
pub struct RecubeConfig {
    pub max_conflicts_per_cube: usize,
    pub split_variables: usize,
    pub max_live_cubes: usize,
    // El prefiltro no conoce las auxiliares que añade preprocess.
    pub max_split_variable: u32,
}
impl Default for RecubeConfig {
    fn default() -> Self {
        Self {
            max_conflicts_per_cube: 50_000,
            split_variables: 10,
            max_live_cubes: 65_536,
            max_split_variable: u32::MAX,
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct SwarmStats''')
cdcl = replace_once(cdcl, '    pub cubes_filtered: usize,\n    pub total:', '    pub cubes_filtered: usize,\n    pub cubes_yielded: usize,\n    pub cubes_split: usize,\n    pub total:')
cdcl = replace_once(cdcl, '    pub gpu_filtered: AtomicU64,\n}', '    pub gpu_filtered: AtomicU64,\n    pub cubes_yielded: AtomicU64,\n    pub cubes_split: AtomicU64,\n}')
cdcl = replace_once(cdcl, 'enum SearchOutcome { Sat(Vec<i32>), Unsat, Cancelled }', 'enum SearchOutcome { Sat(Vec<i32>), Unsat, Cancelled, Yielded }')
cdcl = replace_once(cdcl, '        SearchOutcome::Cancelled => unreachable!("Serial execution cannot be cancelled"),', '''        SearchOutcome::Cancelled => unreachable!("Serial execution cannot be cancelled"),
        SearchOutcome::Yielded => unreachable!("Serial execution has no conflict budget"),''')
cdcl = replace_once(cdcl, '''    mut control: SearchControl<'_, '_>,
) -> (SearchOutcome, SolveStats) {
    let outcome = search(state, variables, feedback, &mut control);''', '''    control: SearchControl<'_, '_>,
) -> (SearchOutcome, SolveStats) {
    run_state_budget::<false>(state, variables, feedback, control, 0)
}
fn run_state_budget<const BUDGETED: bool>(
    state: &mut State,
    variables: &[u32],
    feedback: Option<ReignitionConfig>,
    mut control: SearchControl<'_, '_>,
    max_conflicts: usize,
) -> (SearchOutcome, SolveStats) {
    debug_assert!(!BUDGETED || control.proof.is_none());
    let outcome = search::<BUDGETED>(state, variables, feedback, &mut control, max_conflicts);''')
cdcl = replace_once(cdcl, 'fn search(\n    state: &mut State,', 'fn search<const BUDGETED: bool>(\n    state: &mut State,')
cdcl = replace_once(cdcl, '''    control: &mut SearchControl<'_, '_>,
) -> SearchOutcome {
    // La cancelación''', '''    control: &mut SearchControl<'_, '_>,
    max_conflicts: usize,
) -> SearchOutcome {
    // La especialización <false> elimina el presupuesto de la ruta estática.
    let mut yield_after_root = false;
    // La cancelación''')
cdcl = replace_once(cdcl, '''                beacon.publish(state.stats.conflicts, false);
            }
            continue;
        }
        if state.conflicts >= state.next_cleanup''', '''                beacon.publish(state.stats.conflicts, false);
            }
            if BUDGETED && max_conflicts != 0 && state.stats.conflicts >= max_conflicts {
                // El conflicto ya se aprendió. Restaurar raíz antes de ceder el cubo.
                state.backjump(0);
                yield_after_root = true;
            }
            continue;
        }
        if BUDGETED && yield_after_root {
            debug_assert_eq!(state.level(), 0);
            debug_assert_eq!(state.head, state.trail.len());
            if control.cube.iter().any(|&literal| state.value(literal) == -1) {
                return SearchOutcome::Unsat;
            }
            return SearchOutcome::Yielded;
        }
        if state.conflicts >= state.next_cleanup''')
cdcl = replace_once(cdcl, '            SearchOutcome::Cancelled => { report.cancelled += 1; break; }', '''            SearchOutcome::Cancelled => { report.cancelled += 1; break; }
            SearchOutcome::Yielded => unreachable!("Static worker has no conflict budget"),''')
fragment = (root / 'results/recube_orchestrator.fragment.rs').read_text(encoding='utf-8')
cdcl = replace_once(cdcl, '#[cfg(test)]\nmod deterministic_import_tests', fragment + '\n#[cfg(test)]\nmod deterministic_import_tests')
(review / 'src/hybrid_cdcl.rs').write_text(cdcl, encoding='utf-8')

facade = (root / 'src/hybrid.rs').read_text(encoding='utf-8')
facade = replace_once(facade, 'pub fn verify_model(formula:', '''// GPU y cubos duros se coordinan en el llamador; la ruta estática sigue intacta.
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

pub fn verify_model(formula:''')
(review / 'src/hybrid.rs').write_text(facade, encoding='utf-8')

hunt = (root / 'src/bin/hunt.rs').read_text(encoding='utf-8')
hunt = replace_once(hunt, '//     --no-share', '//     --recube-conflicts C cede cubos tras C conflictos y los reenvía a la GPU (opcional)\n//     --no-share')
hunt = replace_once(hunt, '    use_gpu: bool,\n}', '    use_gpu: bool,\n    recube: Option<hybrid::cdcl::RecubeConfig>,\n}')
hunt = replace_once(hunt, 'csv: None, use_gpu: false,', 'csv: None, use_gpu: false, recube: None,')
hunt = replace_once(hunt, '            "--no-share" =>', '''            "--recube-conflicts" => {
                let budget = usize::try_from(number(value(flag)?)?.max(1))
                    .map_err(|_| "Presupuesto de conflictos demasiado grande")?;
                options.recube = Some(hybrid::cdcl::RecubeConfig {
                    max_conflicts_per_cube: budget,
                    ..hybrid::cdcl::RecubeConfig::default()
                });
            }
            "--no-share" =>''')
hunt = replace_once(hunt, '    Ok(options)\n}', '''    if options.recube.is_some() && (options.workers.is_none() || options.preprocess_only) {
        return Err("--recube-conflicts exige --workers y no admite --preprocess-only".into());
    }
    Ok(options)
}''')
hunt = replace_once(hunt, '''    gpu: Option<&mut gpu_dispatcher::GpuConqueror>,
) -> hybrid::ParallelHybridReport {''', '''    gpu: Option<&mut gpu_dispatcher::GpuConqueror>,
    recube: Option<hybrid::cdcl::RecubeConfig>,
) -> hybrid::ParallelHybridReport {''')
hunt = replace_once(hunt, '''    let Some(gpu) = gpu else {
        return solve_hybrid_parallel_with_progress(formula, config, swarm, progress);
    };''', '''    let Some(gpu) = gpu else {
        return match recube {
            Some(recube) => hybrid::solve_hybrid_parallel_with_recubing(
                formula, config, swarm, recube, progress, None,
            ),
            None => solve_hybrid_parallel_with_progress(formula, config, swarm, progress),
        };
    };''')
hunt = replace_once(hunt, '''    hybrid::solve_hybrid_parallel_with_filter(
        formula, config, swarm, progress, Some(&mut filter),
    )''', '''    match recube {
        Some(recube) => hybrid::solve_hybrid_parallel_with_recubing(
            formula, config, swarm, recube, progress, Some(&mut filter),
        ),
        None => hybrid::solve_hybrid_parallel_with_filter(
            formula, config, swarm, progress, Some(&mut filter),
        ),
    }''')
hunt = replace_once(hunt, '    let gpu_active = conqueror.is_some();', '    let gpu_active = conqueror.is_some();\n    let recube_active = options.recube.is_some();')
hunt = replace_once(hunt, '''                    let total = progress.cubes_total.load(Ordering::Relaxed);

                    if gpu_active''', '''                    let total = progress.cubes_total.load(Ordering::Relaxed);
                    let recube_suffix = if recube_active {
                        format!("; recubings={}, yielded={}",
                            progress.cubes_split.load(Ordering::Relaxed),
                            progress.cubes_yielded.load(Ordering::Relaxed))
                    } else {
                        String::new()
                    };

                    if gpu_active''')
hunt = replace_once(hunt, '"{}; gpu_filtered={gpu_filtered}",', '"{}{recube_suffix}; gpu_filtered={gpu_filtered}",')
hunt = replace_once(hunt, '''                        eprintln!(
                            "{}",
                            status_line(''', '''                        eprintln!(
                            "{}{recube_suffix}",
                            status_line(''')
hunt = replace_once(hunt, '''            &formula, config, swarm, progress.as_deref(), conqueror.as_mut(),''', '''            &formula, config, swarm, progress.as_deref(), conqueror.as_mut(), options.recube,''')
hunt = replace_once(hunt, '''        let stats = &report.swarm;
        Outcome {''', '''        let stats = &report.swarm;
        if recube_active {
            eprintln!(
                "c recubing: yielded={}, splits={}, leaves={}, gpu_filtered={}",
                stats.cubes_yielded, stats.cubes_split,
                stats.cubes_generated - stats.cubes_split,
                progress.as_ref().map_or(0, |p| p.gpu_filtered.load(Ordering::Relaxed)),
            );
        }
        Outcome {''')
(review / 'src/bin/hunt.rs').write_text(hunt, encoding='utf-8')

parts = []
for relative in ('src/hybrid_cdcl.rs', 'src/hybrid.rs', 'src/bin/hunt.rs', 'tests/recubing.rs'):
    before = (root / relative).read_text(encoding='utf-8') if (root / relative).exists() else ''
    after = (review / relative).read_text(encoding='utf-8')
    parts.extend(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f'a/dpll_check/{relative}' if before else '/dev/null', tofile=f'b/dpll_check/{relative}'))
(root / 'results/dynamic-recubing.patch').write_text(''.join(parts), encoding='utf-8')
print(review)
