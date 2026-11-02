// La API de solver.rs tal como la usa otro programa.
use motor_core::dimacs::Formula;
use motor_core::gen_ramsey;
use motor_core::hybrid::cdcl::{CubeVerdict, RecubeConfig};
use motor_core::hybrid::verify_model;
use motor_core::solver::{solve, solve_with, Event, Extras, Halt, Options, Problem, SolveError, Status, StopToken};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn unchecked() -> Options {
    Options { certify: false, ..Options::default() }
}

// Sin simetria ni paridad, el palomar no termina a tiempo.
fn pigeonhole(holes: i32) -> Problem {
    let variable = |pigeon: i32, hole: i32| pigeon * holes + hole + 1;
    let mut formula: Formula = (0..=holes).map(|pigeon| (0..holes).map(|hole| variable(pigeon, hole)).collect()).collect();
    for hole in 0..holes {
        for first in 0..=holes {
            for second in first + 1..=holes {
                formula.push(vec![-variable(first, hole), -variable(second, hole)]);
            }
        }
    }
    Problem::cnf(formula).unwrap()
}

fn hard_options() -> Options {
    Options { symmetry: false, parity: false, ..unchecked() }
}

fn generator(seed: u64) -> impl FnMut(u64) -> u64 {
    let mut state = seed;
    move |bound| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    }
}

fn random_formula(next: &mut impl FnMut(u64) -> u64, variables: u64, clauses: u64) -> Formula {
    (0..clauses).map(|_| (0..1 + next(3)).map(|_| {
        let id = 1 + next(variables) as i32;
        if next(2) == 0 { id } else { -id }
    }).collect()).collect()
}

fn satisfiable_by_brute_force(formula: &Formula, variables: u64) -> bool {
    (0..1_u32 << variables).any(|bits| formula.iter().all(|clause| clause.iter().any(|&literal| {
        (bits >> (literal.unsigned_abs() - 1) & 1 == 1) == (literal > 0)
    })))
}

#[test]
fn the_answer_matches_brute_force_in_series_and_with_the_swarm() {
    let mut next = generator(0x9E37_79B9_7F4A_7C15);
    for round in 0..300 {
        let clauses = 4 + next(24);
        let formula = random_formula(&mut next, 6, clauses);
        let expected = satisfiable_by_brute_force(&formula, 6);
        let problem = Problem::cnf(formula.clone()).unwrap();
        for workers in [None, Some(3)] {
            let solution = solve(&problem, &Options { workers, ..unchecked() }).unwrap();
            assert_eq!(solution.status, if expected { Status::Sat } else { Status::Unsat }, "{round}: {formula:?}");
            assert_eq!(solution.halt, None);
            assert!(!solution.proof_checked);
            assert_eq!(solution.model_checked, expected);
            if let Some(model) = &solution.model {
                assert!(verify_model(&formula, model), "{round}: {formula:?}");
            }
        }
    }
}

#[test]
fn bad_input_is_an_error_not_a_verdict() {
    assert!(matches!(Problem::cnf(vec![vec![1, 0]]), Err(SolveError::Input(_))));
    assert!(matches!(Problem::cnf(vec![vec![i32::MIN]]), Err(SolveError::Input(_))));
    assert!(matches!(Problem::dimacs("1 x 0"), Err(SolveError::Input(_))));
    assert!(matches!(Problem::opb("+1 x1 x2 >= 1 ;"), Err(SolveError::Input(_))));
    assert!(matches!(Problem::from_file("no-existe.cnf"), Err(SolveError::Io(_))));
    let bad = motor_core::opb::Problem {
        variables: 1, objective: None,
        constraints: vec![motor_core::opb::Constraint { terms: vec![(1, 2)], degree: 1 }],
    };
    assert!(matches!(Problem::pb(bad), Err(SolveError::Input(_))));
}

#[test]
fn a_time_limit_is_never_an_unsat_in_series_or_with_the_swarm() {
    let problem = pigeonhole(15);
    for workers in [None, Some(2)] {
        let start = Instant::now();
        let options = Options { workers, timeout: Some(Duration::from_millis(300)), ..hard_options() };
        let solution = solve(&problem, &options).unwrap();
        assert_eq!(solution.status, Status::Unknown);
        assert_eq!(solution.halt, Some(Halt::Timeout));
        assert_eq!(solution.exit_code(), 124);
        assert!(start.elapsed() < Duration::from_secs(10), "{:?}", start.elapsed());
    }
}

#[test]
fn stop_works_from_another_thread_and_the_token_can_be_reused() {
    let problem = pigeonhole(15);
    for workers in [None, Some(2)] {
        let token = StopToken::default();
        let stopper = {
            let token = token.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                token.stop();
            })
        };
        let options = Options { workers, stop: Some(token.clone()), ..hard_options() };
        let solution = solve(&problem, &options).unwrap();
        stopper.join().unwrap();
        assert_eq!((solution.status, solution.halt), (Status::Unknown, Some(Halt::Cancelled)));
        token.clear();
        let easy = Problem::dimacs("p cnf 1 1\n1 0\n").unwrap();
        assert_eq!(solve(&easy, &options).unwrap().status, Status::Sat);
    }
}

#[test]
fn stopping_the_dynamic_swarm_is_not_a_failed_coverage_check() {
    let problem = pigeonhole(15);
    let options = Options { workers: Some(2), timeout: Some(Duration::from_millis(300)), ..hard_options() };
    let extras = Extras { recube: Some(RecubeConfig { max_conflicts_per_cube: 200, ..RecubeConfig::default() }), ..Extras::default() };
    let solution = solve_with(&problem, &options, extras).unwrap();
    assert_eq!((solution.status, solution.halt), (Status::Unknown, Some(Halt::Timeout)));
}

#[test]
fn progress_is_reported_while_searching() {
    let beats = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&beats);
    let options = Options {
        timeout: Some(Duration::from_millis(400)), progress_interval: Some(Duration::from_millis(50)),
        on_event: Some(Arc::new(move |event: &Event| {
            if let Event::Progress(progress) = event {
                seen.lock().unwrap().push((progress.conflicts, progress.stopping));
            }
        })),
        ..hard_options()
    };
    solve(&pigeonhole(15), &options).unwrap();
    let beats = beats.lock().unwrap();
    assert!(beats.len() >= 2, "{beats:?}");
    assert!(beats.iter().any(|&(conflicts, _)| conflicts > 0), "{beats:?}");
    assert!(beats.iter().any(|&(_, stopping)| stopping), "{beats:?}");
}

#[test]
fn an_objective_reports_each_improvement_and_the_optimum() {
    let problem = Problem::opb("min: +3 x1 +2 x2 +4 x3 ;\n+1 x1 +1 x2 +1 x3 >= 2 ;\n+2 x1 -1 ~x3 <= 1 ;\n").unwrap();
    for workers in [None, Some(2)] {
        let costs = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&costs);
        let options = Options {
            workers,
            on_event: Some(Arc::new(move |event: &Event| {
                if let Event::Improved { cost } = event {
                    seen.lock().unwrap().push(*cost);
                }
            })),
            ..unchecked()
        };
        let solution = solve(&problem, &options).unwrap();
        assert_eq!(solution.status, Status::Optimum);
        assert_eq!(solution.cost, Some(5));
        assert_eq!(solution.model, Some(vec![1, 2, -3]));
        assert!(solution.model_checked && !solution.proof_checked);
        assert_eq!(costs.lock().unwrap().last(), Some(&5));
    }
}

#[test]
fn a_panic_inside_the_engine_does_not_reach_the_caller() {
    let problem = pigeonhole(6);
    let mut filter = |_: &[Vec<i32>]| -> Vec<CubeVerdict> { panic!("prefiltro roto") };
    let options = Options { workers: Some(2), ..hard_options() };
    let extras = Extras { filter: Some(&mut filter), ..Extras::default() };
    let error = solve_with(&problem, &options, extras).unwrap_err();
    assert!(matches!(&error, SolveError::Internal(message) if message.contains("prefiltro roto")), "{error}");
}

#[test]
fn the_json_is_one_line_with_the_result() {
    let solution = solve(&Problem::dimacs("p cnf 2 2\n1 2 0\n-1 0\n").unwrap(), &unchecked()).unwrap();
    let json = solution.to_json();
    assert!(!json.contains('\n'));
    assert!(json.starts_with("{\"status\":\"SAT\",\"model\":[-1,2],\"cost\":null,\"model_checked\":true,\"proof_checked\":false,\"reason\":null,"), "{json}");
    assert!(json.ends_with("}}"), "{json}");
}

#[test]
#[ignore = "requiere veripb"]
fn a_formula_in_memory_comes_out_certified() {
    let problem = Problem::dimacs("p cnf 2 4\n1 2 0\n-1 2 0\n1 -2 0\n-1 -2 0\n").unwrap();
    for workers in [None, Some(2)] {
        let solution = solve(&problem, &Options { workers, ..Options::default() }).unwrap();
        assert_eq!((solution.status, solution.proof_checked), (Status::Unsat, true));
    }
    let optimum = Problem::opb("min: +1 x1 +1 x2 ;\n+1 x1 +1 x2 >= 1 ;\n").unwrap();
    let solution = solve(&optimum, &Options::default()).unwrap();
    assert_eq!((solution.status, solution.cost, solution.proof_checked), (Status::Optimum, Some(1), true));
}

// Cada resolucion tiene sus ficheros: a la vez y con trazas distintas, ninguna pisa a otra.
#[test]
#[ignore = "requiere veripb"]
fn several_certified_solves_can_run_at_the_same_time() {
    let handles: Vec<_> = (0..8).map(|index| std::thread::spawn(move || {
        let holes = 3 + index % 3;
        let solution = solve(&pigeonhole(holes), &Options::default()).unwrap();
        assert_eq!((solution.status, solution.proof_checked), (Status::Unsat, true), "{holes}");
    })).collect();
    for handle in handles {
        handle.join().unwrap();
    }
}

// R(3,5) en K14 es UNSAT, y parar a mitad de la comprobacion no puede salir como tal.
#[test]
#[ignore = "requiere veripb"]
fn stopping_during_the_check_gives_no_verdict() {
    let directory = std::env::temp_dir().join(format!("motor_api_check_{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("r35.cnf");
    let arguments = ["14".to_owned(), path.to_string_lossy().into_owned(), "--cliques".to_owned(), "3".to_owned(), "5".to_owned()];
    gen_ramsey::run(&arguments).unwrap();
    let problem = Problem::from_file(&path).unwrap();
    let certified = solve(&problem, &Options::default()).unwrap();
    assert_eq!((certified.status, certified.proof_checked), (Status::Unsat, true));
    // Con el tiempo justo para buscar y no para comprobar.
    let limit = Duration::from_secs_f64(certified.stats.search_ms / 1000.0 + certified.stats.symmetry.ms / 1000.0 + 0.05);
    let solution = solve(&problem, &Options { timeout: Some(limit), ..Options::default() }).unwrap();
    std::fs::remove_dir_all(&directory).unwrap();
    assert_eq!(solution.status, Status::Unknown, "{limit:?}: {}", solution.to_json());
    assert_eq!(solution.halt, Some(Halt::Timeout));
    assert!(!solution.proof_checked);
}
