//! Entrada experimental independiente: ninguna dependencia del motor CDCL.
#![deny(unsafe_code)]
#![deny(warnings)]

#[path = "../ramsey43_gpu.rs"]
pub mod ramsey43_gpu;
use ramsey43_gpu::{
    canonical_star, parse_partial_dimacs, self_test, Graph43, NativeRamsey43, Outcome,
    SearchConfig, SearchReport,
};
use std::path::{Path, PathBuf};

const HELP: &str = "gpu_ramsey43 [--self-test] [--base parcial.sat | --canonical-stars] [--lookahead] [--explicit-trail] [--neuro-model modelo.onnx|engine.plan] [--neuro-mock] [--policy-interval R] [--prefix-bits P] [--steps S] [--epochs E] [--batch-warps K] [--seed S] [--model modelo.sat]\n\
Exclusivo K43: 903 aristas, dos colores y ausencia de K5 en AMBOS colores.\n\
La base debe contener líneas v DIMACS sin variables auxiliares. Sin --base comienza vacío.\n\
--canonical-stars ejecuta secuencialmente grados 18,19,20,21; épocas por clase.\n\
--lookahead fuerza colores por K5 incompleto y límites de grado necesarios.\n\
--explicit-trail usa journal u16 y sufijo léxico; conserva los prefijos de --seed.\n\
--neuro-model archivo.onnx|archivo.plan: TensorRT real, requiere feature neuro-trt; engine cargado una vez.\n\
--neuro-mock prueba el puente GPU determinista; implica --explicit-trail y --lookahead.\n\
--policy-interval R: decisiones libres entre consultas, 1..65536; defecto 32.\n\
--dataset archivo.nrds: muestrea padres reales; requiere feature neuro-dataset e implica journal/lookahead.\n\
--sample-every N: probabilidad hash 1/N por decisión; defecto 10000.\n\
--probe-horizon H: presupuesto de cada etiqueta, 32..4096; defecto 128.\n\
Defecto: 16 universos, 4 pasos por época, 16 épocas; agotamiento = UNKNOWN.\n\
Pasos por época hasta 65536, ejecutados en ráfagas adaptativas de hasta 512.\n\
--epochs 0 mantiene el mismo DFS residente hasta SAT/UNSAT o Ctrl+C; omitirlo conserva 16 épocas.\n\
No demuestra UNSAT global si --base restringe el grafo. No cambia hunt ni su ejecución.\n\
Ejemplo: cargo run --release --features gpu --bin gpu_ramsey43 -- --self-test";

#[derive(Default)]
struct Options {
    config: SearchConfig,
    base: Option<PathBuf>,
    model: Option<PathBuf>,
    canonical_stars: bool,
    self_test: bool,
    help: bool,
    dataset: DatasetOptions,
    neuro_model: Option<PathBuf>,
}

struct DatasetOptions {
    path: Option<PathBuf>,
    every: u32,
    horizon: u32,
}
impl Default for DatasetOptions {
    fn default() -> Self {
        Self {
            path: None,
            every: 10_000,
            horizon: 128,
        }
    }
}

fn parse_options(args: impl IntoIterator<Item = String>) -> Result<Options, String> {
    let mut options = Options::default();
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--help" | "-h" => options.help = true,
            "--self-test" => options.self_test = true,
            "--canonical-stars" => options.canonical_stars = true,
            "--lookahead" => options.config.lookahead = true,
            "--explicit-trail" => options.config.explicit_trail = true,
            "--neuro-mock" => {
                options.config.neuro_mock = true;
                options.config.explicit_trail = true;
                options.config.lookahead = true;
            }
            "--neuro-model" => {
                options.neuro_model =
                    Some(args.next().ok_or("falta ruta de --neuro-model")?.into());
                options.config.neuro_trt = true;
                options.config.explicit_trail = true;
                options.config.lookahead = true;
            }
            "--dataset" => {
                options.dataset.path = Some(args.next().ok_or("falta ruta de --dataset")?.into());
                options.config.explicit_trail = true;
                options.config.lookahead = true;
            }
            "--sample-every" | "--probe-horizon" => {
                let text = args
                    .next()
                    .ok_or_else(|| format!("falta valor de {flag}"))?;
                let value = text
                    .parse::<u32>()
                    .map_err(|_| format!("valor inválido de {flag}"))?;
                if flag == "--sample-every" {
                    if value == 0 {
                        return Err("--sample-every debe ser positivo".into());
                    }
                    options.dataset.every = value;
                } else {
                    if !(32..=4096).contains(&value) {
                        return Err("--probe-horizon requiere 32..4096".into());
                    }
                    options.dataset.horizon = value;
                }
            }
            "--base" => options.base = Some(args.next().ok_or("falta ruta de --base")?.into()),
            "--model" => options.model = Some(args.next().ok_or("falta ruta de --model")?.into()),
            "--prefix-bits" | "--steps" | "--epochs" | "--batch-warps" | "--seed"
            | "--policy-interval" => {
                let text = args
                    .next()
                    .ok_or_else(|| format!("falta valor de {flag}"))?;
                let value = text
                    .parse::<u32>()
                    .map_err(|_| format!("valor inválido de {flag}: {text}"))?;
                match flag.as_str() {
                    "--prefix-bits" => options.config.prefix_bits = value,
                    "--steps" => options.config.steps = value,
                    "--epochs" => options.config.epochs = value,
                    "--batch-warps" => options.config.batch_warps = value,
                    "--policy-interval" => {
                        if !(1..=65_536).contains(&value) {
                            return Err("--policy-interval requiere 1..65536".into());
                        }
                        options.config.policy_interval = value;
                    }
                    _ => options.config.order_seed = value,
                }
            }
            _ => return Err(format!("opción desconocida: {flag}")),
        }
    }
    if options.canonical_stars && options.base.is_some() {
        return Err("--canonical-stars y --base son mutuamente excluyentes".into());
    }
    if options.dataset.path.is_some() && (options.config.neuro_mock || options.config.neuro_trt) {
        return Err(
            "--dataset usa el DFS de referencia; la política neuronal selecciona otro modo".into(),
        );
    }
    if options.config.neuro_mock && options.config.neuro_trt {
        return Err("--neuro-mock y --neuro-model son mutuamente excluyentes".into());
    }
    Ok(options)
}

fn print_report(report: &SearchReport) {
    let seconds = report.elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
    let launch_count = report.launches.max(1) as f64;
    println!("elapsed_ms,universes,epochs,launches,transitions,prunes,completed,transitions_per_s,prunes_per_s,launch_mean_ms,launch_max_ms,min_batch_warps,max_batch_warps,max_launch_steps,decisions,forced_assignments,lookahead_checks,lookahead_conflicts");
    println!(
        "{:.3},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{},{},{},{},{},{},{}",
        report.elapsed.as_secs_f64() * 1000.0,
        report.universes,
        report.epochs,
        report.launches,
        report.transitions,
        report.prunes,
        report.completed,
        report.transitions as f64 / seconds,
        report.prunes as f64 / seconds,
        report.launch_time.as_secs_f64() * 1000.0 / launch_count,
        report.max_launch.as_secs_f64() * 1000.0,
        report.min_batch_warps,
        report.max_batch_warps,
        report.max_launch_steps,
        report.decisions,
        report.forced_assignments,
        report.lookahead_checks,
        report.lookahead_conflicts,
    );
    if report.policy_queries != 0 || report.policy_responses != 0 || report.policy_fallbacks != 0 {
        println!(
            "c neuro-policy: queries={}; responses={}; fallbacks={}",
            report.policy_queries, report.policy_responses, report.policy_fallbacks
        );
    }
}

fn print_outcome(
    outcome: Outcome,
    model_path: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    match outcome {
        Outcome::Sat(model) => {
            let literals = model.model()?;
            let text = format!(
                "s SATISFIABLE\nv {} 0\n",
                literals
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            if let Some(path) = model_path {
                std::fs::write(path, &text)?;
            }
            print!("{text}");
        }
        Outcome::Unsat => {
            println!("s UNSATISFIABLE");
        }
        Outcome::Unknown => println!("s UNKNOWN"),
    }
    Ok(())
}

fn solve_class(
    device: &NativeRamsey43,
    graph: &Graph43,
    config: SearchConfig,
    #[cfg(feature = "neuro-dataset")] exporter: Option<&mut ramsey43_gpu::dataset::DatasetExporter>,
    #[cfg(feature = "neuro-dataset")] sample_every: u32,
    #[cfg(feature = "neuro-dataset")] probe_horizon: u32,
) -> Result<SearchReport, ramsey43_gpu::Error> {
    let universes = config.validate(graph)?;
    let epoch_limit = if config.epochs == 0 {
        "unlimited".to_owned()
    } else {
        config.epochs.to_string()
    };
    println!("c base: free_edges={}; universes={universes}; prefix_bits={}; steps={}; epochs={epoch_limit}; batch_warps={}; seed={}; lookahead={}; explicit_trail={}", graph.free_edges(), config.prefix_bits, config.steps, config.batch_warps, config.order_seed, config.lookahead, config.explicit_trail);
    if config.neuro_mock {
        println!(
            "c neuro-mock: batch_capacity={}; policy_interval={}; one_resident_slot=true",
            config.batch_warps, config.policy_interval
        );
    }
    if config.neuro_trt {
        println!("c neuro-trt: batch_capacity={}; policy_interval={}; tensors=VRAM; resident_engine=true", config.batch_warps, config.policy_interval);
    }
    #[cfg(feature = "neuro-dataset")]
    let report = match exporter {
        Some(exporter) => {
            device.solve_with_dataset(graph, config, exporter, sample_every, probe_horizon)?
        }
        None => device.solve(graph, config)?,
    };
    #[cfg(not(feature = "neuro-dataset"))]
    let report = device.solve(graph, config)?;
    print_report(&report);
    Ok(report)
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options(std::env::args().skip(1)).map_err(std::io::Error::other)?;
    if options.help {
        println!("{HELP}");
        return Ok(());
    }
    #[cfg(not(feature = "neuro-dataset"))]
    if options.dataset.path.is_some() {
        return Err(std::io::Error::other("--dataset requiere --features neuro-dataset").into());
    }
    // Validar antes de abrir CUDA conserva errores de entrada independientes del driver.
    let graph = if options.canonical_stars {
        canonical_star(18)?
    } else if let Some(path) = options.base.as_ref() {
        parse_partial_dimacs(&std::fs::read_to_string(path)?)?
    } else {
        Graph43::default()
    };
    options.config.validate(&graph)?;
    let device = NativeRamsey43::new(0)?;
    println!("c device: {}", device.name());
    if options.self_test {
        let checks = self_test(&device)?;
        println!("c self-test: {checks} comprobaciones CUDA aprobadas");
        return Ok(());
    }
    #[cfg(feature = "neuro-trt")]
    if let Some(path) = options.neuro_model.as_ref() {
        let started = std::time::Instant::now();
        device.load_tensorrt(path, options.config.batch_warps)?;
        println!(
            "c neuro-trt-loaded: model={}; load_ms={:.3}; reuse_across_classes=true",
            path.display(),
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
    #[cfg(feature = "neuro-dataset")]
    let mut exporter = match options.dataset.path.as_ref() {
        Some(path) => Some(ramsey43_gpu::dataset::DatasetExporter::start(
            ramsey43_gpu::dataset::DatasetConfig {
                path: path.clone(),
                sample_every: options.dataset.every,
                horizon: options.dataset.horizon,
                seed: u64::from(options.config.order_seed),
                queue_batches: 4,
                max_batch_records: 32,
            },
        )?),
        None => None,
    };
    if !options.canonical_stars {
        let report = solve_class(
            &device,
            &graph,
            options.config,
            #[cfg(feature = "neuro-dataset")]
            exporter.as_mut(),
            #[cfg(feature = "neuro-dataset")]
            options.dataset.every,
            #[cfg(feature = "neuro-dataset")]
            options.dataset.horizon,
        )?;
        if matches!(report.outcome, Outcome::Unsat) && options.base.is_some() {
            println!(
                "c UNSAT se refiere a todas las extensiones de esta base, no a bases distintas."
            );
        }
        print_outcome(report.outcome, options.model.as_deref())?;
        #[cfg(feature = "neuro-dataset")]
        finish_dataset(exporter)?;
        return Ok(());
    }

    let mut classes_unsat = 0_u32;
    println!(
        "c canonical-run: pid={}; classes=4; order=18,19,20,21; mode=sequential",
        std::process::id()
    );
    let mut classes_unknown = 0_u32;
    let mut total_elapsed = std::time::Duration::ZERO;
    let mut total_transitions = 0_u64;
    let mut total_prunes = 0_u64;
    let mut total_completed = 0_u64;
    let mut total_decisions = 0_u64;
    let mut total_forced = 0_u64;
    for degree in 18..=21 {
        let epoch_limit = if options.config.epochs == 0 {
            "unlimited".to_owned()
        } else {
            options.config.epochs.to_string()
        };
        println!(
            "c class: root_degree={degree}; classes=18,19,20,21; epochs_per_class={epoch_limit}"
        );
        let graph = canonical_star(degree)?;
        // solve libera su arena local antes de devolver el informe; el contexto se reutiliza.
        let report = solve_class(
            &device,
            &graph,
            options.config,
            #[cfg(feature = "neuro-dataset")]
            exporter.as_mut(),
            #[cfg(feature = "neuro-dataset")]
            options.dataset.every,
            #[cfg(feature = "neuro-dataset")]
            options.dataset.horizon,
        )?;
        total_elapsed += report.elapsed;
        total_transitions += report.transitions;
        total_prunes += report.prunes;
        total_completed += report.completed;
        total_decisions += report.decisions;
        total_forced += report.forced_assignments;
        match report.outcome {
            Outcome::Sat(model) => {
                println!("c class-result: root_degree={degree}; verdict=SATISFIABLE");
                print_outcome(Outcome::Sat(model), options.model.as_deref())?;
                #[cfg(feature = "neuro-dataset")]
                finish_dataset(exporter)?;
                return Ok(());
            }
            Outcome::Unsat => {
                classes_unsat += 1;
                println!("c class-result: root_degree={degree}; verdict=UNSATISFIABLE");
            }
            Outcome::Unknown => {
                classes_unknown += 1;
                println!("c class-result: root_degree={degree}; verdict=UNKNOWN; la clase no queda refutada");
            }
        }
    }
    println!("c canonical-summary: classes_run=4; classes_unsat={classes_unsat}; classes_unknown={classes_unknown}; elapsed_ms={:.3}; transitions={total_transitions}; prunes={total_prunes}; completed={total_completed}; decisions={total_decisions}; forced_assignments={total_forced}", total_elapsed.as_secs_f64() * 1000.0);
    if classes_unsat == 4 {
        println!(
            "c Las cuatro estrellas cubren K43 por renombrado y complemento global de colores."
        );
        print_outcome(Outcome::Unsat, options.model.as_deref())?;
    } else {
        print_outcome(Outcome::Unknown, options.model.as_deref())?;
    }
    #[cfg(feature = "neuro-dataset")]
    finish_dataset(exporter)?;
    Ok(())
}

#[cfg(feature = "neuro-dataset")]
fn finish_dataset(
    exporter: Option<ramsey43_gpu::dataset::DatasetExporter>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(exporter) = exporter {
        let stats = exporter.finish()?;
        println!("c dataset-writer: written_records={}; written_batches={}; dropped_records={}; dropped_batches={}",
            stats.written_records, stats.written_batches, stats.dropped_records, stats.dropped_batches);
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("c error: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trained_policy_is_explicit_and_excludes_other_backends() {
        let defaults = Options::default();
        assert!(!defaults.config.neuro_trt);
        assert!(defaults.neuro_model.is_none());
        let options =
            parse_options(["--neuro-model", "neuro_ramsey_policy.onnx"].map(str::to_owned))
                .unwrap();
        assert!(options.config.neuro_trt);
        assert!(options.config.explicit_trail && options.config.lookahead);
        assert!(!options.config.neuro_mock);
        assert_eq!(
            options.neuro_model.as_deref(),
            Some(Path::new("neuro_ramsey_policy.onnx"))
        );
        for args in [
            vec!["--neuro-model"],
            vec!["--neuro-model", "policy.onnx", "--neuro-mock"],
            vec!["--neuro-mock", "--neuro-model", "policy.onnx"],
            vec!["--dataset", "samples.nrds", "--neuro-model", "policy.onnx"],
        ] {
            assert!(parse_options(args.into_iter().map(str::to_owned)).is_err());
        }
    }
    #[test]
    fn dataset_is_opt_in_and_enables_exact_parent_capture() {
        let defaults = Options::default();
        assert!(defaults.dataset.path.is_none());
        assert_eq!(
            (defaults.dataset.every, defaults.dataset.horizon),
            (10_000, 128)
        );
        let parsed = parse_options(
            [
                "--dataset",
                "sample.nrds",
                "--sample-every",
                "10000",
                "--probe-horizon",
                "256",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        assert!(parsed.config.explicit_trail && parsed.config.lookahead);
        assert!(!parsed.config.neuro_mock);
        assert_eq!(parsed.dataset.horizon, 256);
        for arguments in [
            vec!["--sample-every", "0"],
            vec!["--probe-horizon", "31"],
            vec!["--dataset"],
            vec!["--dataset", "data.nrds", "--neuro-mock"],
        ] {
            assert!(parse_options(arguments.into_iter().map(str::to_owned)).is_err());
        }
    }
    #[test]
    fn cli_options_do_not_enable_the_cdcl_or_any_existing_feature() {
        let options = parse_options(
            [
                "--prefix-bits",
                "8",
                "--steps",
                "1",
                "--epochs",
                "2",
                "--batch-warps",
                "64",
                "--base",
                "base.sat",
                "--model",
                "out.sat",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(options.config.validate(&Graph43::default()).unwrap(), 256);
        assert_eq!(options.config.steps, 1);
        assert_eq!(options.base, Some("base.sat".into()));
        assert_eq!(options.model, Some("out.sat".into()));
        assert!(parse_options(["--workers".to_owned(), "16".to_owned()]).is_err());
        assert!(parse_options(["--epochs".to_owned(), "-1".to_owned()]).is_err());
        assert!(parse_options(["--base".to_owned()]).is_err());
    }

    #[test]
    fn continuous_epochs_are_explicit_and_default_budgets_remain_finite() {
        let defaults = parse_options(std::iter::empty()).unwrap();
        assert_eq!(defaults.config.epochs, 16);
        assert_eq!(defaults.config.steps, 4);
        assert!(!defaults.config.has_next_epoch(16));
        let continuous = parse_options(["--epochs", "0"].map(str::to_owned)).unwrap();
        assert_eq!(continuous.config.epochs, 0);
        assert!(continuous.config.validate(&Graph43::default()).is_ok());
        assert!(continuous.config.has_next_epoch(u64::MAX));
    }

    #[test]
    fn canonical_stars_do_not_mix_with_a_restrictive_base() {
        assert!(
            parse_options(["--canonical-stars", "--base", "custom.sat"].map(str::to_owned))
                .is_err()
        );
        assert!(
            parse_options(["--base", "custom.sat", "--canonical-stars"].map(str::to_owned))
                .is_err()
        );
        let options =
            parse_options(["--canonical-stars", "--lookahead"].map(str::to_owned)).unwrap();
        assert!(options.canonical_stars);
        assert!(options.config.lookahead);
        assert!(options.base.is_none());
        assert!(!Options::default().config.lookahead);
        assert!(!Options::default().config.explicit_trail);
    }

    #[test]
    fn explicit_trail_is_optional_and_composes_with_lookahead() {
        let options = parse_options(["--explicit-trail"].map(str::to_owned)).unwrap();
        assert!(options.config.explicit_trail);
        assert!(!options.config.lookahead);
        let options =
            parse_options(["--lookahead", "--explicit-trail"].map(str::to_owned)).unwrap();
        assert!(options.config.explicit_trail);
        assert!(options.config.lookahead);
    }

    #[test]
    fn mock_implies_exact_journal_and_validates_interval() {
        let options =
            parse_options(["--neuro-mock", "--policy-interval", "1"].map(str::to_owned)).unwrap();
        assert!(
            options.config.neuro_mock && options.config.lookahead && options.config.explicit_trail
        );
        assert_eq!(options.config.policy_interval, 1);
        assert!(!Options::default().config.neuro_mock);
        assert_eq!(Options::default().config.policy_interval, 32);
        for value in ["0", "65537", "-1", "nan"] {
            assert!(parse_options(["--policy-interval", value].map(str::to_owned)).is_err());
        }
        assert!(parse_options(["--policy-interval"].map(str::to_owned)).is_err());
        #[cfg(not(feature = "neuro-mock"))]
        assert!(options
            .config
            .validate(&Graph43::default())
            .unwrap_err()
            .to_string()
            .contains("--features neuro-mock"));
        #[cfg(feature = "neuro-mock")]
        assert!(options.config.validate(&Graph43::default()).is_ok());
    }
}
