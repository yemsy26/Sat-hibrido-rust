//! Paridad CPU/GPU: la referencia de `ramsey_ref.rs` recorre el mismo árbol que los kernels
//! `ramsey43_search_journal[_smart]` en K43 (un universo), contador a contador.
//! `cargo test --release --features gpu --test ramsey_gpu_parity -- --ignored`
#![cfg(feature = "gpu")]
use motor_core::ramsey43_gpu::{canonical_star, NativeRamsey43, Outcome, SearchConfig};
use motor_core::ramsey_ref::{self, solve, Config, Heuristic, Problem};

#[test]
#[ignore = "requiere GPU CUDA y NVRTC 12.6"]
fn journal_kernels_match_cpu_reference_on_k43_stars() {
    let device = NativeRamsey43::new(0).unwrap();
    let problem = Problem::new(5, 5, 43).unwrap();
    let (steps, epochs) = (65_536, 8);
    for (lookahead, seed) in [(false, 0), (true, 0), (false, 0x5eed), (true, 0x5eed)] {
        for degree in problem.star_degrees() {
            let config = SearchConfig {
                prefix_bits: 0,
                order_seed: seed,
                steps,
                epochs,
                batch_warps: 1,
                lookahead,
                explicit_trail: true,
                ..SearchConfig::default()
            };
            let gpu = device.solve(&canonical_star(degree).unwrap(), config).unwrap();
            let budget = u64::from(steps) * u64::from(epochs);
            let mut reference = Config::new(Heuristic::Lex, lookahead, budget);
            reference.seed = seed;
            let cpu = solve(problem, &problem.canonical_star(degree), reference, None).unwrap();
            let label = format!("estrella {degree} lookahead={lookahead} semilla={seed}");
            assert!(matches!(gpu.outcome, Outcome::Unknown), "{label}");
            assert_eq!(cpu.outcome, ramsey_ref::Outcome::Unknown, "{label}");
            let c = cpu.counters;
            assert_eq!(
                (gpu.transitions, gpu.prunes, gpu.decisions, gpu.forced_assignments),
                (c.transitions, c.prunes, c.decisions, c.forced),
                "{label}"
            );
            assert_eq!(
                (gpu.lookahead_checks, gpu.lookahead_conflicts),
                (c.checks, c.conflicts),
                "{label}"
            );
        }
    }
}
