//! Audita las bases r55_42 con el lector y la referencia del propio motor, sin CUDA.
//! Cada base debe pasar `check_reference` como `Open` (sin K5 en la parte fija),
//! con exactamente la estrella del vértice 0 libre (42 aristas).
//! Compilar: rustc --edition 2021 -O scripts/r55_check_bases.rs -o <exe>
#![allow(dead_code)]

#[path = "../src/ramsey43_gpu.rs"]
mod ramsey43_gpu;
use ramsey43_gpu::{check_reference, parse_partial_dimacs, CheckResult, VERTICES};

fn main() {
    let dir = std::env::args().nth(1).expect("uso: r55_check_bases <directorio de bases>");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("directorio de bases ilegible")
        .map(|entry| entry.expect("entrada ilegible").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "sat"))
        .collect();
    paths.sort();
    let mut failures = 0_u32;
    println!("base,check_reference,free_edges,star0_free,red_edges,ok");
    for path in &paths {
        let text = std::fs::read_to_string(path).expect("base ilegible");
        let (check, free, star_free, red) = match parse_partial_dimacs(&text) {
            Ok(graph) => {
                let rows = graph.rows();
                let red: u32 = rows[..VERTICES].iter().map(|row| row.count_ones()).sum();
                (
                    format!("{:?}", check_reference(&graph)).replace(',', ";"),
                    graph.free_edges(),
                    rows[0] | rows[VERTICES] == 0,
                    red / 2,
                )
            }
            Err(error) => (format!("parse error: {error}"), 0, false, 0),
        };
        let ok = check == format!("{:?}", CheckResult::Open) && free == 42 && star_free;
        failures += u32::from(!ok);
        let name = path.file_stem().unwrap().to_string_lossy();
        println!("{name},{check},{free},{star_free},{red},{ok}");
    }
    eprintln!("c bases={}; failures={failures}", paths.len());
    std::process::exit(i32::from(failures != 0 || paths.is_empty()));
}
