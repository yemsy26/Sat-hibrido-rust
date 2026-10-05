fn run_case(name: &str, n: usize, formula: Formula, expected_sat: bool, config: HybridConfig) {
    use std::hint::black_box;
    use std::time::Instant;
    let start = Instant::now();
    let (baseline_model, baseline) = black_box(cdcl::solve_cdcl(black_box(&formula), None, config.activity_boost));
    let baseline_ms = start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(baseline_model.is_some(), expected_sat);
    if let Some(model) = &baseline_model { assert!(verify_model(&formula, model)); }
    let start = Instant::now();
    let report = black_box(solve_hybrid(black_box(&formula), config));
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(report.model.is_some(), expected_sat);
    // Reutiliza exactamente el mismo tensor inicial para aislar el efecto del feedback.
    let (static_ms, static_stats) = if report.used_cdcl && config.reignition_interval != 0 {
        let prior: Vec<(u32, f64)> = report.oracle.variable_ids.iter().copied()
            .zip(report.oracle.continuous_values.iter().copied()).collect();
        let start = Instant::now();
        let (model, stats) = black_box(cdcl::solve_cdcl(black_box(&formula), Some(&prior), config.activity_boost));
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(model.is_some(), expected_sat);
        if let Some(model) = &model { assert!(verify_model(&formula, model)); }
        (elapsed, stats)
    } else { (report.cdcl_ms, report.cdcl) };
    let feedback_saved = static_stats.decisions as i128 - report.cdcl.decisions as i128;
    let feedback_reduction = if static_stats.decisions == 0 { "NA".to_owned() } else {
        format!("{:.4}", 100.0 * feedback_saved as f64 / static_stats.decisions as f64)
    };
    let saved = baseline.decisions as i128 - report.cdcl.decisions as i128;
    let reduction = if baseline.decisions == 0 { "NA".to_owned() } else {
        format!("{:.4}", 100.0 * saved as f64 / baseline.decisions as f64)
    };
    let status = if report.model.is_some() { "SAT" } else { "UNSAT" };
    let route = if report.used_cdcl { "oracle_cdcl" } else { "oracle_model" };
    // Diferencia real contra el mismo CDCL sin guía; valores negativos indican regresión.
    // reignition_ms forma parte de guided_cdcl_ms; no se suma de nuevo al total.
    println!("{name},{n},{},{status},{route},{baseline_ms:.6},{:.6},{:.6},{total_ms:.6},{},{:.12e},{},{},{saved},{reduction},{},{},{},{},{},{},{:.6},{static_ms:.6},{},{},{feedback_saved},{feedback_reduction}",
        formula.len(), report.oracle_ms, report.cdcl_ms,
        report.oracle.epochs, report.oracle.loss, baseline.decisions, report.cdcl.decisions,
        baseline.conflicts, report.cdcl.conflicts, report.cdcl.nonchronological_backjumps,
        report.cdcl.symmetry_clauses, report.cdcl.reignitions,
        report.cdcl.reignition_epochs, report.cdcl.reignition_ms,
        static_stats.decisions, static_stats.conflicts);
}

fn pigeonhole_3sat(pigeons: usize, holes: usize) -> (usize, Formula) {
    let mut variables = pigeons * holes;
    let mut formula = Formula::new();
    for pigeon in 0..pigeons {
        let positive: Vec<i32> = (0..holes)
            .map(|hole| (pigeon * holes + hole + 1) as i32).collect();
        if positive.len() <= 3 { formula.push(positive); } else {
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
    println!("case,n,clauses,status,route,baseline_cdcl_ms,oracle_ms,guided_cdcl_ms,total_ms,epochs,oracle_loss,baseline_decisions,guided_decisions,decisions_saved,decision_reduction_pct,baseline_conflicts,guided_conflicts,nonchronological,symmetry_clauses,reignitions,reignition_epochs,reignition_ms,static_guided_cdcl_ms,static_guided_decisions,static_guided_conflicts,feedback_decisions_saved,feedback_decision_reduction_pct");
    let config = HybridConfig::default();
    run_case("empty_formula", 0, vec![], true, config);
    run_case("contradictory_units", 1, vec![vec![1], vec![-1]], false, config);
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
        ] { run_case(name, n as usize, formula, true, config); }
    }
    let mut nonchronological = vec![
        vec![-1, -3, 4], vec![-3, 5], vec![-4, -5, 6], vec![-6, 7], vec![-4, -7],
    ];
    nonchronological.extend((8..=18).map(|v| vec![1, v]));
    nonchronological.extend((19..=28).map(|v| vec![2, v]));
    nonchronological.extend((29..=34).map(|v| vec![3, v]));
    run_case("nonchronological", 34, nonchronological, true, config);
    for holes in [3, 4, 5, 6, 14, 19] {
        let (n, formula) = pigeonhole_3sat(holes + 1, holes);
        run_case(&format!("pigeonhole_{}_{holes}", holes + 1), n, formula, false, config);
    }
    for (pigeons, holes) in [(3, 3), (4, 4), (6, 8)] {
        let (n, formula) = pigeonhole_3sat(pigeons, holes);
        run_case(&format!("pigeonhole_{pigeons}_{holes}"), n, formula, true, config);
    }
    // La tautología conserva SAT/UNSAT y fuerza la ruta general sin reconocimiento PHP.
    let (n, mut stress) = pigeonhole_3sat(8, 7);
    stress.push(vec![1, -1]);
    run_case("cdcl_pigeonhole_8_7", n, stress, false, config);
}
