fn run_case(
    name: &str,
    n: usize,
    formula: Formula,
    expected_sat: bool,
    require_skip: bool,
    parallel: bool,
) -> ParallelStats {
    use std::hint::black_box;
    use std::time::Instant;
    let clause_count = formula.len();
    let input = formula.clone();
    let start = Instant::now();
    let (result, stats) = if parallel {
        let workers = std::thread::available_parallelism().map_or(1, |n| n.get());
        black_box(solve_parallel_with_threads(black_box(input), Vec::new(), workers))
    } else {
        let (result, total) = black_box(solve_cdcl(black_box(input), Vec::new()));
        (result, ParallelStats { total, ..ParallelStats::default() })
    };
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(result.is_some(), expected_sat);
    if let Some(model) = &result {
        let mut values = vec![0_i8; n + 1];
        for &literal in model {
            let variable = literal.unsigned_abs() as usize;
            let value = if literal > 0 { 1 } else { -1 };
            assert!(values[variable] == 0 || values[variable] == value);
            values[variable] = value;
        }
        assert!(formula.iter().all(|clause| clause.iter().any(|&literal| {
            values[literal.unsigned_abs() as usize] == if literal > 0 { 1 } else { -1 }
        })));
    }
    if require_skip && !parallel {
        assert!(stats.total.nonchronological_backjumps > 0);
    }
    let status = if result.is_some() { "SAT" } else { "UNSAT" };
    let mode = if parallel { "parallel" } else { "serial" };
    let total = stats.total;
    println!("{name},{mode},{n},{clause_count},{status},{elapsed_ms:.6},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        total.decisions, total.propagations, total.conflicts, total.learned_clauses,
        total.backjumps, total.nonchronological_backjumps,
        total.deleted_clauses, total.database_reductions,
        total.symmetry_rows, total.symmetry_columns, total.symmetry_clauses,
        stats.workers, stats.cubes_generated, stats.cubes_started,
        stats.cubes_unsat, stats.cubes_cancelled,
        stats.lookahead_propagations, stats.lookahead_conflicts);
    stats
}

fn pigeonhole_3sat(holes: usize) -> (usize, Formula) {
    let pigeons = holes + 1;
    let mut variables = pigeons * holes;
    let mut formula = Formula::new();
    for pigeon in 0..pigeons {
        let positive: Vec<i32> = (0..holes)
            .map(|hole| (pigeon * holes + hole + 1) as i32).collect();
        if positive.len() <= 3 {
            formula.push(positive);
        } else {
            variables += 1;
            let mut auxiliary = variables as i32;
            formula.push(vec![positive[0], positive[1], auxiliary]);
            for &literal in &positive[2..positive.len() - 2] {
                variables += 1;
                let next = variables as i32;
                formula.push(vec![-auxiliary, literal, next]);
                auxiliary = next;
            }
            formula.push(vec![-auxiliary, positive[positive.len() - 2], positive[positive.len() - 1]]);
        }
    }
    for hole in 0..holes {
        for first in 0..pigeons {
            for second in first + 1..pigeons {
                formula.push(vec![-((first * holes + hole + 1) as i32),
                    -((second * holes + hole + 1) as i32)]);
            }
        }
    }
    (variables, formula)
}

fn main() {
    println!("case,mode,n,clauses,status,elapsed_ms,decisions,propagations,conflicts,learned,backjumps,nonchronological,deleted,reductions,symmetry_rows,symmetry_columns,symmetry_clauses,workers,cubes_generated,cubes_started,cubes_unsat,cubes_cancelled,lookahead_propagations,lookahead_conflicts");
    for n in [10_i32, 100, 1000] {
        let mut chain: Formula = (1..n).rev().map(|v| vec![-v, v + 1]).collect();
        chain.push(vec![1]);
        let independent: Formula = (1..n - 1).step_by(3)
            .map(|v| vec![v, v + 1, v + 2]).collect();
        let rollback: Formula = (1..n - 2).step_by(3).flat_map(|v| {
            [vec![v, v + 1], vec![-v, v + 2], vec![-v, -(v + 2)]]
        }).collect();
        for (name, formula) in [
            ("reverse_chain", chain), ("independent", independent), ("rollback", rollback),
        ] {
            for parallel in [false, true] {
                run_case(name, n as usize, formula.clone(), true, false, parallel);
            }
        }
    }
    let mut nonchronological = vec![
        vec![-1, -3, 4], vec![-3, 5], vec![-4, -5, 6], vec![-6, 7], vec![-4, -7],
    ];
    nonchronological.extend((8..=18).map(|v| vec![1, v]));
    nonchronological.extend((19..=28).map(|v| vec![2, v]));
    nonchronological.extend((29..=34).map(|v| vec![3, v]));
    for parallel in [false, true] {
        run_case("nonchronological", 34, nonchronological.clone(), true, true, parallel);
    }
    for holes in [3, 4, 5, 6, 14, 19] {
        let (n, formula) = pigeonhole_3sat(holes);
        for parallel in [false, true] {
            let stats = run_case(&format!("pigeonhole_{}_{holes}", holes + 1),
                n, formula.clone(), false, false, parallel);
            assert_eq!(stats.total.symmetry_rows, holes + 1);
            assert_eq!(stats.total.symmetry_columns, holes);
            assert_eq!(stats.total.decisions, 0);
            assert_eq!(stats.total.conflicts, 1);
        }
    }
    // La tautología conserva SAT/UNSAT y ejercita la ruta CDCL general.
    let (n, mut stress) = pigeonhole_3sat(7);
    stress.push(vec![1, -1]);
    for parallel in [false, true] {
        run_case("cdcl_pigeonhole_8_7", n, stress.clone(), false, false, parallel);
    }
}
