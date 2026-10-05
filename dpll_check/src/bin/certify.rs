// Fase 16: certificacion pseudo-booleana (formato 3.0 de VeriPB) con la simetria activada.
//
//   certify <entrada.cnf> <salida.pbp>   resuelve un DIMACS y deja la traza, que se
//                                        comprueba con `veripb --cnf entrada.cnf salida.pbp`
//   certify                              mide el sobrecoste de emitir la traza (CSV)
//   certify --emit <directorio>          lo mismo, y escribe cada caso como .cnf y .pbp
#![forbid(unsafe_code)]
#![deny(warnings)]

#[allow(dead_code)]
#[path = "../hybrid.rs"]
mod hybrid;

use hybrid::proof::PbProof;
use hybrid::{solve_hybrid, solve_hybrid_with_proof, verify_model, HybridConfig};
use std::fs::File;
use std::hint::black_box;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

type Formula = Vec<Vec<i32>>;

// Mediana de RUNS ejecuciones: los casos duran menos de un milisegundo.
const RUNS: usize = 31;

struct Count(u64);
impl Write for Count {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.0 += buffer.len() as u64;
        Ok(buffer.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

// El numero de clausulas de la traza es el de clausulas leidas, no el de la cabecera.
fn parse_dimacs(text: &str) -> Result<Formula, String> {
    let mut formula = Formula::new();
    let mut clause = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('%') {
            break;
        }
        if line.starts_with('c') || line.starts_with('p') {
            continue;
        }
        for token in line.split_whitespace() {
            let literal: i32 = token.parse().map_err(|_| format!("Literal no valido: {token}"))?;
            if literal == i32::MIN {
                return Err(format!("Literal fuera de rango: {token}"));
            }
            if literal == 0 {
                formula.push(std::mem::take(&mut clause));
            } else {
                clause.push(literal);
            }
        }
    }
    if clause.is_empty() { Ok(formula) } else { Err("La ultima clausula no termina en 0".into()) }
}

fn write_dimacs(path: &Path, formula: &Formula) -> io::Result<()> {
    let variables = formula.iter().flatten().map(|literal| literal.unsigned_abs()).max().unwrap_or(0);
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "p cnf {variables} {}", formula.len())?;
    for clause in formula {
        for literal in clause {
            write!(out, "{literal} ")?;
        }
        writeln!(out, "0")?;
    }
    out.flush()
}

fn pigeonhole(pigeons: usize, holes: usize) -> Formula {
    let mut formula: Formula = (0..pigeons)
        .map(|pigeon| (0..holes).map(|hole| (pigeon * holes + hole + 1) as i32).collect())
        .collect();
    for hole in 0..holes {
        for first in 0..pigeons {
            for second in first + 1..pigeons {
                formula.push(vec![-((first * holes + hole + 1) as i32), -((second * holes + hole + 1) as i32)]);
            }
        }
    }
    formula
}

// La codificacion del benchmark principal: cada fila partida en clausulas de tres literales.
fn pigeonhole_3sat(pigeons: usize, holes: usize) -> Formula {
    let mut variables = pigeons * holes;
    let mut formula = Formula::new();
    for pigeon in 0..pigeons {
        let row: Vec<i32> = (0..holes).map(|hole| (pigeon * holes + hole + 1) as i32).collect();
        if row.len() <= 3 {
            formula.push(row);
            continue;
        }
        variables += 1;
        let mut auxiliary = variables as i32;
        formula.push(vec![row[0], row[1], auxiliary]);
        for &literal in &row[2..row.len() - 2] {
            variables += 1;
            formula.push(vec![-auxiliary, literal, variables as i32]);
            auxiliary = variables as i32;
        }
        formula.push(vec![-auxiliary, row[row.len() - 2], row[row.len() - 1]]);
    }
    formula.extend(pigeonhole(pigeons, holes).into_iter().skip(pigeons));
    formula
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_unstable_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn measure(name: &str, formula: &Formula, symmetry: bool, emit: Option<&Path>) -> io::Result<()> {
    let config = HybridConfig { symmetry, ..HybridConfig::default() };
    let (mut plain_ms, mut certified_ms) = (Vec::with_capacity(RUNS), Vec::with_capacity(RUNS));
    let mut row = String::new();
    for _ in 0..RUNS {
        let start = Instant::now();
        let plain = black_box(solve_hybrid(black_box(formula), config));
        plain_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        // Con traza: todo lo que se emite, hasta entregar el ultimo byte.
        let start = Instant::now();
        let mut proof = PbProof::new(Count(0), formula.len());
        let report = black_box(solve_hybrid_with_proof(black_box(formula), config, &mut proof));
        let (steps, dominances, deletions) = (proof.steps(), proof.dominances(), proof.deletions());
        let bytes = proof.finish()?.0;
        certified_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(report.model.is_some(), plain.model.is_some());
        if let Some(model) = &report.model {
            assert!(verify_model(formula, model));
        }
        let variables = formula.iter().flatten().map(|literal| literal.unsigned_abs()).max().unwrap_or(0);
        row = format!(
            "{variables},{},{},{bytes},{steps},{dominances},{deletions},{},{},{},{},{},{}",
            formula.len(), if report.model.is_some() { "SAT" } else { "UNSAT" },
            report.symmetry.generators, report.symmetry.clauses, plain.symmetry.generators,
            plain.symmetry.clauses, report.cdcl.decisions, report.cdcl.conflicts,
        );
    }
    let (plain, certified) = (median(&mut plain_ms), median(&mut certified_ms));
    // A disco: la misma traza, con el coste real de escribirla.
    let mut file_ms = String::from("NA");
    if let Some(directory) = emit {
        write_dimacs(&directory.join(format!("{name}.cnf")), formula)?;
        let start = Instant::now();
        let mut proof = PbProof::new(File::create(directory.join(format!("{name}.pbp")))?, formula.len());
        black_box(solve_hybrid_with_proof(black_box(formula), config, &mut proof));
        proof.finish()?.sync_all()?;
        file_ms = format!("{:.6}", start.elapsed().as_secs_f64() * 1000.0);
    }
    println!(
        "{name},{},{plain:.6},{certified:.6},{:.6},{file_ms},{row}",
        if symmetry { "on" } else { "off" }, certified - plain,
    );
    Ok(())
}

fn benchmark(emit: Option<&Path>) -> io::Result<()> {
    if let Some(directory) = emit {
        std::fs::create_dir_all(directory)?;
    }
    println!("case,symmetry,plain_ms,certified_ms,overhead_ms,file_ms,variables,clauses,status,trace_bytes,steps,dominances,deletions,certified_generators,certified_predicates,plain_generators,plain_predicates,decisions,conflicts");
    // El caso de estres del benchmark principal: PHP 10/9 en 3-SAT mas una tautologia.
    let mut stress = pigeonhole_3sat(10, 9);
    stress.push(vec![1, -1]);
    measure("cdcl_pigeonhole_10_9", &stress, true, emit)?;
    for holes in [14, 19] {
        measure(&format!("pigeonhole_3sat_{}_{holes}", holes + 1), &pigeonhole_3sat(holes + 1, holes), true, emit)?;
    }
    for holes in [9, 19, 29, 39, 49] {
        measure(&format!("pigeonhole_{}_{holes}", holes + 1), &pigeonhole(holes + 1, holes), true, emit)?;
    }
    // Sin simetria la traza es toda busqueda: clausulas aprendidas y borradas.
    for holes in [7, 8] {
        measure(&format!("search_pigeonhole_{}_{holes}", holes + 1), &pigeonhole(holes + 1, holes), false, emit)?;
    }
    Ok(())
}

fn certify(input: &Path, output: &Path) -> Result<(), String> {
    let text = std::fs::read_to_string(input).map_err(|error| format!("{}: {error}", input.display()))?;
    let formula = parse_dimacs(&text)?;
    let failed = |error: io::Error| format!("{}: {error}", output.display());
    let start = Instant::now();
    let mut proof = PbProof::new(File::create(output).map_err(failed)?, formula.len());
    let report = solve_hybrid_with_proof(&formula, HybridConfig::default(), &mut proof);
    let (steps, dominances, deletions, bytes) = (proof.steps(), proof.dominances(), proof.deletions(), proof.bytes());
    proof.finish().map_err(failed)?;
    println!(
        "c {:.3} ms, {} clauses, {} symmetry generators, {} decisions, {} conflicts",
        start.elapsed().as_secs_f64() * 1000.0, formula.len(), report.symmetry.generators,
        report.cdcl.decisions, report.cdcl.conflicts,
    );
    println!("c trace: {steps} steps ({dominances} dominance), {deletions} deletions, {bytes} bytes before the footer");
    match report.model {
        Some(model) => {
            println!("s SATISFIABLE");
            let line: Vec<String> = model.iter().map(i32::to_string).collect();
            println!("v {} 0", line.join(" "));
        }
        None => println!("s UNSATISFIABLE"),
    }
    Ok(())
}

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let outcome = match arguments.as_slice() {
        [] => benchmark(None).map_err(|error| error.to_string()),
        [flag, directory] if flag == "--emit" => benchmark(Some(Path::new(directory))).map_err(|error| error.to_string()),
        [input, output] => certify(Path::new(input), Path::new(output)),
        _ => Err("Uso: certify [<entrada.cnf> <salida.pbp> | --emit <directorio>]".into()),
    };
    if let Err(error) = outcome {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
