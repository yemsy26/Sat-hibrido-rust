fn run_case(name: &str, n: usize, formula: Formula, expected_sat: bool, require_skip: bool) -> SolveStats {
    use std::hint::black_box;
    use std::time::Instant;

    let clause_count = formula.len();
    let input = formula.clone();
    let start = Instant::now();
    let (result, stats) = black_box(solve_cdcl(black_box(input), Vec::new()));
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
    if require_skip {
        assert!(stats.nonchronological_backjumps > 0);
    }
    let status = if result.is_some() { "SAT" } else { "UNSAT" };
    println!("{name},{n},{clause_count},{status},{elapsed_ms:.6},{},{},{},{},{},{},{},{},{},{},{}",
        stats.decisions, stats.propagations, stats.conflicts, stats.learned_clauses,
        stats.backjumps, stats.nonchronological_backjumps,
        stats.deleted_clauses, stats.database_reductions,
        stats.symmetry_rows, stats.symmetry_columns, stats.symmetry_clauses);
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
    println!("case,n,clauses,status,elapsed_ms,decisions,propagations,conflicts,learned,backjumps,nonchronological,deleted,reductions,symmetry_rows,symmetry_columns,symmetry_clauses");
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
            run_case(name, n as usize, formula, true, false);
        }
    }
    let mut nonchronological = vec![
        vec![-1, -3, 4], vec![-3, 5], vec![-4, -5, 6], vec![-6, 7], vec![-4, -7],
    ];
    nonchronological.extend((8..=18).map(|v| vec![1, v]));
    nonchronological.extend((19..=28).map(|v| vec![2, v]));
    nonchronological.extend((29..=34).map(|v| vec![3, v]));
    run_case("nonchronological", 34, nonchronological, true, true);
    for holes in 3..=6 {
        let (n, formula) = pigeonhole_3sat(holes);
        run_case(&format!("pigeonhole_{}_{holes}", holes + 1), n, formula, false, false);
    }
    for holes in [14, 19] {
        let (n, formula) = pigeonhole_3sat(holes);
        let stats = run_case(&format!("pigeonhole_{}_{holes}", holes + 1), n, formula, false, false);
        assert_eq!(stats.symmetry_rows, holes + 1);
        assert_eq!(stats.symmetry_columns, holes);
        assert_eq!(stats.decisions, 0);
        assert_eq!(stats.conflicts, 1);
    }
}
