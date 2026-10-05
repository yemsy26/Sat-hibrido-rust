fn run_case(name: &str, n: usize, formula: Formula, expected_sat: bool, compare_sharing: bool) {
    use std::hint::black_box;
    use std::time::Instant;
    let config = HybridConfig::default();
    let start = Instant::now();
    let serial = black_box(solve_hybrid(black_box(&formula), config));
    let serial_ms = start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(serial.model.is_some(), expected_sat);
    if let Some(model) = &serial.model { assert!(verify_model(&formula, model)); }
    let serial_status = if serial.model.is_some() { "SAT" } else { "UNSAT" };
    println!("{name},serial,{n},{},{serial_status},{serial_ms:.6},{:.6},{:.6},{},{},{},{},{:.6},1,0,0,0,0,0,0,0,0,0,1.000000,0,0.0000",
        formula.len(), serial.oracle_ms, serial.cdcl_ms, serial.cdcl.decisions,
        serial.cdcl.conflicts, serial.cdcl.reignitions, serial.cdcl.reignition_epochs,
        serial.cdcl.reignition_ms);
    let mut no_share_decisions = None;
    for share_enabled in [false, true] {
        if !share_enabled && !compare_sharing { continue; }
        let swarm_config = cdcl::SwarmConfig { share_enabled, ..cdcl::SwarmConfig::default() };
        let start = Instant::now();
        let report = black_box(solve_hybrid_parallel(black_box(&formula), config, swarm_config));
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(report.model.is_some(), expected_sat);
        let stats = &report.swarm;
        let total = stats.total;
        if !share_enabled { no_share_decisions = Some(total.decisions); }
        let reduction = no_share_decisions.map_or_else(|| "NA".to_owned(), |before| {
            if before == 0 { "NA".to_owned() } else {
                format!("{:.4}", 100.0 * (before as f64 - total.decisions as f64) / before as f64)
            }
        });
        let status = if report.model.is_some() { "SAT" } else { "UNSAT" };
        let mode = if share_enabled { "parallel_shared" } else { "parallel_private" };
        let speedup = serial_ms / elapsed_ms.max(f64::MIN_POSITIVE);
        // Trabajo total real: no se presupone que compartir reduzca decisiones o tiempo.
        println!("{name},{mode},{n},{},{status},{elapsed_ms:.6},{:.6},{:.6},{},{},{},{},{:.6},{},{},{},{},{},{},{},{},{},{},{speedup:.6},{},{reduction}",
            formula.len(), report.oracle_ms, report.conquer_ms,
            total.decisions, total.conflicts, total.reignitions, total.reignition_epochs,
            total.reignition_ms, stats.workers, stats.cubes_generated, stats.cubes_started,
            stats.cubes_unsat, stats.cubes_cancelled,
            total.exported_clauses, total.imported_clauses, total.dropped_exports,
            total.import_backjumps, stats.shared_clauses.len(),
            no_share_decisions.map_or_else(|| "NA".to_owned(), |value| value.to_string()));
    }
}

fn pigeonhole_3sat(pigeons: usize, holes: usize) -> (usize, Formula) {
    let mut variables = pigeons * holes;
    let mut formula = Formula::new();
    for pigeon in 0..pigeons {
        let row: Vec<i32> = (0..holes).map(|hole| (pigeon * holes + hole + 1) as i32).collect();
        if row.len() <= 3 { formula.push(row); } else {
            variables += 1;
            let mut auxiliary = variables as i32;
            formula.push(vec![row[0], row[1], auxiliary]);
            for &literal in &row[2..row.len() - 2] {
                variables += 1;
                let next = variables as i32;
                formula.push(vec![-auxiliary, literal, next]);
                auxiliary = next;
            }
            formula.push(vec![-auxiliary, row[row.len() - 2], row[row.len() - 1]]);
        }
    }
    for hole in 0..holes {
        for first in 0..pigeons {
            for second in first + 1..pigeons {
                formula.push(vec![-((first * holes + hole + 1) as i32), -((second * holes + hole + 1) as i32)]);
            }
        }
    }
    (variables, formula)
}

fn main() {
    println!("case,mode,n,clauses,status,elapsed_ms,oracle_ms,conquer_ms,decisions,conflicts,reignitions,reignition_epochs,reignition_ms,workers,cubes_generated,cubes_started,cubes_unsat,cubes_cancelled,exports,imports,sharing_drops,import_backjumps,pool_clauses,speedup_vs_serial,private_decisions,decision_reduction_vs_private_pct");
    run_case("empty_formula", 0, vec![], true, false);
    run_case("contradictory_units", 1, vec![vec![1], vec![-1]], false, false);
    for n in [10_i32, 100, 1000] {
        let mut chain: Formula = (1..n).rev().map(|v| vec![-v, v + 1]).collect();
        chain.push(vec![1]);
        let independent: Formula = (1..n - 1).step_by(3).map(|v| vec![v, v + 1, v + 2]).collect();
        let rollback: Formula = (1..n - 2).step_by(3).flat_map(|v| {
            [vec![v, v + 1], vec![-v, v + 2], vec![-v, -(v + 2)]]
        }).collect();
        for (name, formula) in [("reverse_chain", chain), ("independent", independent), ("rollback", rollback)] {
            run_case(name, n as usize, formula, true, false);
        }
    }
    for holes in [3, 4, 5, 6, 14, 19] {
        let (n, formula) = pigeonhole_3sat(holes + 1, holes);
        run_case(&format!("pigeonhole_{}_{holes}", holes + 1), n, formula, false, false);
    }
    for (pigeons, holes) in [(3, 3), (4, 4), (6, 8)] {
        let (n, formula) = pigeonhole_3sat(pigeons, holes);
        run_case(&format!("pigeonhole_{pigeons}_{holes}"), n, formula, true, false);
    }
    // cargo run --release -- 10 ejecuta 11/10; el valor por defecto ejecuta 10/9.
    let holes = std::env::args().nth(1).map_or(9, |value| value.parse::<usize>().expect("Expected hole count"));
    assert!((7..=10).contains(&holes), "Stress hole count must be 7..=10");
    let (n, mut stress) = pigeonhole_3sat(holes + 1, holes);
    stress.push(vec![1, -1]);
    run_case(&format!("cdcl_pigeonhole_{}_{holes}", holes + 1), n, stress, false, true);
}
