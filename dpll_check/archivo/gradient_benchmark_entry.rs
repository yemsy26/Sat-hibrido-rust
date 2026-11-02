fn run_case(name: &str, n: usize, formula: Formula, expected: &str, config: GradientConfig) {
    use std::hint::black_box;
    use std::time::Instant;
    let clauses = formula.len();
    let start = Instant::now();
    let report = black_box(solve_gradient(black_box(&formula), config));
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let status = match &report.outcome {
        Outcome::Sat(model) => {
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
            assert_ne!(expected, "UNSAT", "Unexpected verified model for an UNSAT benchmark");
            "SAT"
        }
        Outcome::Unknown => "UNKNOWN",
    };
    // expected describe la construcción del benchmark, no una prueba del optimizador.
    println!("{name},{n},{clauses},{expected},{status},{elapsed_ms:.6},{},{},{:.8},{:.12e},{:.12e},{}",
        report.epochs, report.restarts, report.learning_rate,
        report.loss, report.gradient_l2, report.violated_clauses);
}

fn pigeonhole_3sat(pigeons: usize, holes: usize) -> (usize, Formula) {
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
    println!("case,n,clauses,expected,status,elapsed_ms,epochs,restarts,learning_rate,loss,gradient_l2,violated_clauses");
    let config = GradientConfig::default();
    run_case("empty_formula", 0, vec![], "SAT", config);
    run_case("contradictory_units", 1, vec![vec![1], vec![-1]], "UNSAT", config);
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
            run_case(name, n as usize, formula, "SAT", config);
        }
    }
    for holes in [3, 4, 5, 6, 14, 19] {
        let (n, formula) = pigeonhole_3sat(holes + 1, holes);
        run_case(&format!("pigeonhole_{}_{holes}", holes + 1), n, formula, "UNSAT", config);
    }
    for (pigeons, holes) in [(3, 3), (4, 4), (6, 8)] {
        let (n, formula) = pigeonhole_3sat(pigeons, holes);
        run_case(&format!("pigeonhole_{pigeons}_{holes}"), n, formula, "SAT", config);
    }
}
