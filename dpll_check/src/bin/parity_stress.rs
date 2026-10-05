// Fase 14b: estres de paridad. Restricciones XOR camufladas como CNF, con y sin la capa
// algebraica. Sin la capa la resolucion crece exponencialmente y el motor no tiene limite
// de conflictos: cada familia fija hasta que tamano se mide ese caso.
#![forbid(unsafe_code)]
#![deny(warnings)]

#[allow(dead_code)]
#[path = "../hybrid.rs"]
mod hybrid;

use hybrid::{solve_hybrid, verify_model, HybridConfig};
use std::time::Instant;

type Formula = Vec<Vec<i32>>;

const BASELINE_BUDGET_MS: f64 = 2000.0;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
    fn shuffle<T>(&mut self, items: &mut [T]) {
        for index in (1..items.len()).rev() {
            items.swap(index, self.below(index + 1));
        }
    }
}

// CNF de x1 xor ... xor xk = value: una clausula por patron de signos de paridad opuesta.
fn xor_clauses(variables: &[i32], value: bool, formula: &mut Formula) {
    for pattern in 0..1_u32 << variables.len() {
        if (pattern.count_ones() & 1 == 1) != value {
            formula.push(variables.iter().enumerate().map(|(bit, &variable)| {
                if pattern >> bit & 1 == 1 { -variable } else { variable }
            }).collect());
        }
    }
}

// Una paridad ancha se corta en eslabones de tres variables con auxiliares frescas.
fn chained_xor(variables: &[i32], value: bool, next_id: &mut i32, formula: &mut Formula) {
    if variables.len() <= 3 {
        return xor_clauses(variables, value, formula);
    }
    let mut carry = variables[0];
    for &variable in &variables[1..variables.len() - 2] {
        *next_id += 1;
        xor_clauses(&[carry, variable, *next_id], false, formula);
        carry = *next_id;
    }
    xor_clauses(&[carry, variables[variables.len() - 2], variables[variables.len() - 1]], value, formula);
}

// Camuflaje: identificadores permutados, clausulas barajadas y literales en orden arbitrario.
fn camouflage(mut formula: Formula, rng: &mut Rng) -> Formula {
    let last = formula.iter().flatten().map(|literal| literal.unsigned_abs()).max().unwrap_or(0) as usize;
    let mut ids: Vec<i32> = (1..=last as i32).collect();
    rng.shuffle(&mut ids);
    for clause in &mut formula {
        for literal in clause.iter_mut() {
            let id = ids[literal.unsigned_abs() as usize - 1];
            *literal = if *literal > 0 { id } else { -id };
        }
        rng.shuffle(clause);
    }
    rng.shuffle(&mut formula);
    formula
}

// Formula de Tseitin sobre un grafo regular aleatorio con carga total impar: insatisfacible.
fn tseitin(vertices: usize, degree: usize, seed: u64) -> Formula {
    let mut rng = Rng(seed | 1);
    let incident = loop {
        // Modelo de configuracion: se reintenta hasta no tener lazos ni aristas repetidas.
        let mut stubs: Vec<usize> = (0..vertices * degree).map(|stub| stub / degree).collect();
        rng.shuffle(&mut stubs);
        let mut edges: Vec<(usize, usize)> = stubs.chunks(2).map(|pair| (pair[0].min(pair[1]), pair[0].max(pair[1]))).collect();
        edges.sort_unstable();
        if edges.iter().all(|&(a, b)| a != b) && edges.windows(2).all(|pair| pair[0] != pair[1]) {
            let mut incident = vec![Vec::new(); vertices];
            for (index, &(a, b)) in edges.iter().enumerate() {
                incident[a].push(index as i32 + 1);
                incident[b].push(index as i32 + 1);
            }
            break incident;
        }
    };
    let mut formula = Formula::new();
    for (vertex, edges) in incident.iter().enumerate() {
        xor_clauses(edges, vertex == 0, &mut formula);
    }
    camouflage(formula, &mut rng)
}

// Sistema lineal disperso con solucion plantada: satisfacible por construccion.
fn planted(variables: usize, equations: usize, seed: u64) -> Formula {
    let mut rng = Rng(seed | 1);
    let secret: Vec<bool> = (0..variables).map(|_| rng.next() & 1 == 1).collect();
    let mut formula = Formula::new();
    for _ in 0..equations {
        let mut chosen = [0_usize; 3];
        loop {
            for slot in &mut chosen {
                *slot = rng.below(variables);
            }
            if chosen[0] != chosen[1] && chosen[0] != chosen[2] && chosen[1] != chosen[2] {
                break;
            }
        }
        let value = chosen.iter().fold(false, |parity, &variable| parity ^ secret[variable]);
        xor_clauses(&chosen.map(|variable| variable as i32 + 1), value, &mut formula);
    }
    camouflage(formula, &mut rng)
}

// Paridades anchas cortadas con auxiliares. Con `contradiction` se anade la suma de todas
// con el termino cambiado: el sistema pasa a ser insatisfacible sin que ninguna clausula
// lo delate.
fn wide(variables: usize, equations: usize, width: usize, contradiction: bool, seed: u64) -> Formula {
    let mut rng = Rng(seed | 1);
    let secret: Vec<bool> = (0..variables).map(|_| rng.next() & 1 == 1).collect();
    let mut next_id = variables as i32;
    let mut formula = Formula::new();
    let mut total = vec![false; variables];
    let mut total_value = false;
    for _ in 0..equations {
        let mut chosen: Vec<usize> = (0..variables).collect();
        rng.shuffle(&mut chosen);
        chosen.truncate(width);
        let value = chosen.iter().fold(false, |parity, &variable| parity ^ secret[variable]);
        for &variable in &chosen {
            total[variable] ^= true;
        }
        total_value ^= value;
        let row: Vec<i32> = chosen.iter().map(|&variable| variable as i32 + 1).collect();
        chained_xor(&row, value, &mut next_id, &mut formula);
    }
    if contradiction {
        let row: Vec<i32> = (0..variables).filter(|&variable| total[variable]).map(|variable| variable as i32 + 1).collect();
        if row.is_empty() {
            // La suma se anula: basta una unidad contradictoria con el sistema.
            formula.push(vec![1]);
            formula.push(vec![-1]);
        } else {
            chained_xor(&row, !total_value, &mut next_id, &mut formula);
        }
    }
    camouflage(formula, &mut rng)
}

fn run(family: &str, size: usize, formula: &Formula, expected_sat: bool, baseline: &mut bool, limit: usize) {
    let variables = formula.iter().flatten().map(|literal| literal.unsigned_abs()).max().unwrap_or(0);
    for parity in [true, false] {
        if !parity && (!*baseline || size > limit) {
            println!("{family},{size},{variables},{},NA,parity_off,skipped,NA,NA,0,0,0,0,0,0,false,0.000000", formula.len());
            continue;
        }
        let config = HybridConfig { parity, ..HybridConfig::default() };
        let start = Instant::now();
        let report = solve_hybrid(formula, config);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(report.model.is_some(), expected_sat, "{family} {size} parity={parity}");
        if let Some(model) = &report.model {
            assert!(verify_model(formula, model));
        }
        if !parity && elapsed_ms > BASELINE_BUDGET_MS {
            *baseline = false;
        }
        let stats = report.parity;
        println!("{family},{size},{variables},{},{},{},{elapsed_ms:.6},{},{},{},{},{},{},{},{},{},{:.6}",
            formula.len(), if expected_sat { "SAT" } else { "UNSAT" },
            if parity { "parity_on" } else { "parity_off" },
            report.cdcl.decisions, report.cdcl.conflicts,
            stats.equations, stats.rank, stats.units, stats.equivalences, stats.clauses,
            stats.auxiliaries, stats.refuted, stats.ms);
    }
}

fn main() {
    println!("family,size,variables,clauses,status,mode,elapsed_ms,decisions,conflicts,parity_equations,parity_rank,parity_units,parity_equivalences,parity_clauses,parity_auxiliaries,parity_refuted,parity_ms");
    // Limites del caso sin capa, tomados de mediciones locales para que termine en segundos.
    let mut baseline = true;
    for vertices in [10, 14, 18, 22, 26, 30, 40, 60, 100, 200, 400] {
        run("tseitin_3_regular", vertices, &tseitin(vertices, 3, 0x51ed_2701), false, &mut baseline, 60);
    }
    let mut baseline = true;
    for vertices in [8, 10, 12, 14, 16, 20, 30, 60, 120] {
        run("tseitin_5_regular", vertices, &tseitin(vertices, 5, 0x0b5e_55ed), false, &mut baseline, 14);
    }
    let mut baseline = true;
    for variables in [40, 60, 80, 100, 140, 200, 400, 800, 1600] {
        run("planted_3xor_full", variables, &planted(variables, variables + variables / 5, 0x7ea5_e11e), true, &mut baseline, 800);
    }
    let mut baseline = true;
    for variables in [40, 60, 80, 100, 140, 200, 400, 800] {
        run("planted_3xor_under", variables, &planted(variables, variables * 4 / 5, 0x0dd_c0de), true, &mut baseline, 800);
    }
    let mut baseline = true;
    for variables in [16, 20, 24, 28, 32, 40, 60, 100, 200] {
        run("wide_parity_sat", variables, &wide(variables, variables * 3 / 4, 8, false, 0xfeed_f00d), true, &mut baseline, 100);
    }
    let mut baseline = true;
    for variables in [16, 20, 24, 28, 32, 40, 60, 100, 200] {
        run("wide_parity_unsat", variables, &wide(variables, variables * 3 / 4, 8, true, 0xdead_beef), false, &mut baseline, 24);
    }
}
