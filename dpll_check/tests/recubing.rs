#![forbid(unsafe_code)]
#![deny(warnings)]
#[allow(dead_code)]
#[path = "../src/hybrid.rs"]
mod hybrid;

use hybrid::cdcl::{solve_swarm_with_recubing, CubeVerdict, RecubeConfig, ReignitionConfig,
    SwarmConfig, SwarmProgress, SwarmStats};
use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::Ordering;

fn parameters(workers: usize) -> SwarmConfig {
    SwarmConfig {
        workers,
        cubes_per_worker: 1,
        reignition: ReignitionConfig { interval: 0, ..ReignitionConfig::default() },
        share_interval: 1,
        ..SwarmConfig::default()
    }
}
fn recube() -> RecubeConfig {
    RecubeConfig { max_conflicts_per_cube: 1, split_variables: 2, max_live_cubes: 64,
        ..RecubeConfig::default() }
}
fn pigeonhole(pigeons: i32, holes: i32) -> Vec<Vec<i32>> {
    let variable = |pigeon: i32, hole: i32| 1 + pigeon * holes + hole;
    let mut formula = Vec::new();
    for pigeon in 0..pigeons {
        formula.push((0..holes).map(|hole| variable(pigeon, hole)).collect());
    }
    for hole in 0..holes {
        for a in 0..pigeons {
            for b in a + 1..pigeons {
                formula.push(vec![-variable(a, hole), -variable(b, hole)]);
            }
        }
    }
    formula
}
fn check_unsat(stats: &SwarmStats, progress: &SwarmProgress) {
    assert_eq!(stats.cubes_unsat + stats.cubes_filtered + stats.cubes_split, stats.cubes_generated);
    assert!(stats.cubes_started <= stats.cubes_generated);
    assert_eq!(progress.cubes_total.load(Ordering::Relaxed),
        (stats.cubes_generated - stats.cubes_split) as u64);
    assert_eq!(progress.cubes_closed.load(Ordering::Relaxed), stats.cubes_unsat as u64);
    assert_eq!(progress.cubes_yielded.load(Ordering::Relaxed), stats.cubes_yielded as u64);
    assert_eq!(progress.cubes_split.load(Ordering::Relaxed), stats.cubes_split as u64);
    assert_eq!(progress.restarts.load(Ordering::Relaxed), stats.total.restarts as u64);
    assert_eq!(progress.conflicts.load(Ordering::Relaxed), stats.total.conflicts as u64);
}
#[test]
fn recube_round_trip_calls_non_send_filter_again_and_preserves_coverage() {
    let formula = pigeonhole(4, 3);
    let calls = Rc::new(Cell::new(0));
    let observed = Rc::clone(&calls);
    // Rc deliberately prevents Send: filtering must remain on the caller.
    let mut filter = move |cubes: &[Vec<i32>]| {
        let call = observed.get();
        observed.set(call + 1);
        vec![if call == 0 { CubeVerdict::Unknown } else { CubeVerdict::Unsat }; cubes.len()]
    };
    let progress = SwarmProgress::default();
    let (model, stats) = solve_swarm_with_recubing(
        &formula, parameters(2), recube(), Some(&progress), Some(&mut filter),
    );
    assert!(model.is_none());
    assert!(calls.get() > 1);
    assert!(stats.cubes_yielded > 0 && stats.cubes_split > 0);
    assert!(stats.cubes_filtered > 0);
    check_unsat(&stats, &progress);
}
#[test]
fn recube_malformed_filter_never_discards_an_unknown_cube() {
    let formula = pigeonhole(4, 3);
    let mut filter = |_: &[Vec<i32>]| Vec::new();
    let progress = SwarmProgress::default();
    let (model, stats) = solve_swarm_with_recubing(
        &formula, parameters(2), recube(), Some(&progress), Some(&mut filter),
    );
    assert!(model.is_none());
    assert_eq!(stats.cubes_filtered, 0);
    assert!(stats.cubes_yielded > 0);
    check_unsat(&stats, &progress);
}
#[test]
fn recube_filter_panic_unwinds_the_scoped_workers_instead_of_proving_unsat() {
    let formula = pigeonhole(4, 3);
    let mut calls = 0;
    let mut filter = |cubes: &[Vec<i32>]| {
        calls += 1;
        assert_eq!(calls, 1, "injected GPU callback failure");
        vec![CubeVerdict::Unknown; cubes.len()]
    };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        solve_swarm_with_recubing(
            &formula, parameters(2), recube(), None, Some(&mut filter),
        )
    }));
    assert!(outcome.is_err());
    assert!(calls > 1);
}
#[test]
fn recube_frontier_limit_resumes_the_parent_without_duplicate_start_count() {
    let formula = pigeonhole(4, 3);
    let progress = SwarmProgress::default();
    let (model, stats) = solve_swarm_with_recubing(
        &formula, parameters(1), RecubeConfig { max_live_cubes: 1, ..recube() },
        Some(&progress), None,
    );
    assert!(model.is_none());
    assert!(stats.cubes_yielded > 0);
    assert_eq!(stats.cubes_split, 0);
    assert_eq!(stats.cubes_generated, 1);
    assert_eq!(stats.cubes_started, 1);
    check_unsat(&stats, &progress);
}
#[test]
fn recube_no_representable_split_variables_resumes_unbounded() {
    let formula = pigeonhole(4, 3);
    let progress = SwarmProgress::default();
    let (model, stats) = solve_swarm_with_recubing(
        &formula, parameters(2), RecubeConfig { max_split_variable: 0, ..recube() },
        Some(&progress), None,
    );
    assert!(model.is_none());
    assert!(stats.cubes_yielded > 0);
    assert_eq!(stats.cubes_split, 0);
    check_unsat(&stats, &progress);
}
#[test]
fn recube_preserves_sat_models_on_small_exhaustive_formulas() {
    let variables = 4;
    for allowed in [0_u16, 1, 2, 0x8000, 0x0042, 0x5555, 0xffff] {
        let formula: Vec<Vec<i32>> = (0..1 << variables)
            .filter(|&bits| allowed & (1 << bits) == 0)
            .map(|bits| (0..variables).map(|variable| {
                let literal = variable + 1;
                if bits & (1 << variable) == 0 { literal } else { -literal }
            }).collect()).collect();
        for workers in [1, 2, 4] {
            let (model, _) = solve_swarm_with_recubing(
                &formula, parameters(workers), recube(), None, None,
            );
            assert_eq!(model.is_some(), allowed != 0);
            if let Some(model) = model {
                assert!(hybrid::verify_model(&formula, &model));
            }
        }
    }
}
