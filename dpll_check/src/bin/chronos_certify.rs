// Fase 17: puente SAT <-> Chronos-VFS v2. El emisor PbProof del motor no escribe en disco:
// su flujo VeriPB entra en el anillo SPSC en RAM de la libreria chronos-vfs y un hilo
// consumidor lo drena segun la ruta:
//
//   --route compress   (por defecto) zstd al vuelo hacia un .pbp.zst; el flujo crudo nunca
//                      toca disco y VeriPB lee el .zst tal cual. Es la ruta certificable.
//   --route discard    cuenta y hashea (FNV-1a) y tira los bytes. Techo de throughput, SIN
//                      certificado: una traza descartada no es una prueba.
//   --route check      clasifica cada linea por regla (rup/red/dom/del) in situ. NO es un
//                      chequeo de solidez (eso lo hace veripb), solo consumo sin disco.
//
//   chronos_certify <formula.cnf> [--route R] [--out F] [--zstd N] [--threads T] [--ring-mib M]
//
// La RAM del puente es la del anillo (64 MiB por defecto), con contrapresion cuando se
// llena: no depende de que la traza total sean MB o TB.
#![forbid(unsafe_code)]
#![deny(warnings)]

#[allow(dead_code)]
#[path = "../hybrid.rs"]
mod hybrid;

use chronos_vfs::{Consumer, RingBuffer, RingWriter, ZstdConfig, CHUNK_SIZE};
use hybrid::proof::PbProof;
use hybrid::{solve_hybrid_with_proof, verify_model, HybridConfig};
use std::time::Instant;

type Formula = Vec<Vec<i32>>;

// Tope por pasada del consumidor: el espacio vuelve al productor por lotes, no al vaciar
// el anillo entero.
const DRAIN_BATCH: usize = 64 * CHUNK_SIZE;

#[derive(Clone, Copy, PartialEq)]
enum Route {
    Compress,
    Discard,
    Check,
}

#[derive(Default)]
struct Consumed {
    bytes: u64,
    batches: u64,
    hash: u64,
    lines: u64,
    rup: u64,
    red: u64,
    dom: u64,
    del: u64,
    other: u64,
}

// Clasifica una linea VeriPB por su primera palabra. Parse estructural, no semantico.
fn classify(line: &[u8], tally: &mut Consumed) {
    tally.lines += 1;
    match line.split(|&byte| byte == b' ').next().unwrap_or(b"") {
        b"rup" => tally.rup += 1,
        b"red" => tally.red += 1,
        b"dom" => tally.dom += 1,
        b"del" => tally.del += 1,
        _ => tally.other += 1,
    }
}

// Rutas sin compresor: lee la region publicada in situ hasta que el productor cierre.
// `pending` guarda el resto de linea que queda entre dos lotes.
fn consume(mut consumer: Consumer, check: bool) -> Consumed {
    let mut tally = Consumed { hash: 0xcbf2_9ce4_8422_2325, ..Consumed::default() };
    let mut pending: Vec<u8> = Vec::new();
    while consumer.wait_for_data() {
        let (front, back) = consumer.readable();
        let front = &front[..front.len().min(DRAIN_BATCH)];
        let back = &back[..back.len().min(DRAIN_BATCH - front.len())];
        for data in [front, back] {
            if check {
                let mut rest = data;
                while let Some(position) = rest.iter().position(|&byte| byte == b'\n') {
                    if pending.is_empty() {
                        classify(&rest[..position], &mut tally);
                    } else {
                        pending.extend_from_slice(&rest[..position]);
                        classify(&pending, &mut tally);
                        pending.clear();
                    }
                    rest = &rest[position + 1..];
                }
                pending.extend_from_slice(rest);
            } else {
                // FNV-1a, para que el consumo no se optimice a nada.
                for &byte in data {
                    tally.hash = (tally.hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
        }
        let count = front.len() + back.len();
        consumer.release(count);
        tally.bytes += count as u64;
        tally.batches += 1;
    }
    if !pending.is_empty() {
        classify(&pending, &mut tally);
    }
    tally
}

struct Solved {
    satisfiable: bool,
    decisions: usize,
    conflicts: usize,
    dominances: u64,
    // Veces que el motor encontro el anillo lleno y espero al consumidor.
    stalls: u64,
}

// El motor escribe la traza en el anillo. Al terminar, `finish` emite el pie y devuelve el
// productor, y cerrarlo es lo que le dice al consumidor que el flujo acabo.
fn solve(formula: &Formula, config: HybridConfig, ring_writer: RingWriter) -> Result<Solved, String> {
    let mut proof = PbProof::new(ring_writer, formula.len());
    let report = solve_hybrid_with_proof(formula, config, &mut proof);
    let dominances = proof.dominances();
    let ring_writer = proof.finish().map_err(|error| format!("El anillo dejo de aceptar la traza: {error}"))?;
    let stalls = ring_writer.stalls();
    ring_writer.close();
    if let Some(model) = &report.model {
        assert!(verify_model(formula, model), "El modelo no satisface la formula");
    }
    Ok(Solved {
        satisfiable: report.model.is_some(), decisions: report.cdcl.decisions,
        conflicts: report.cdcl.conflicts, dominances, stalls,
    })
}

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

struct Options {
    input: String,
    route: Route,
    out: Option<String>,
    zstd_level: i32,
    threads: u32,
    ring_bytes: usize,
}

fn parse_options(arguments: &[String]) -> Result<Options, String> {
    let mut options = Options {
        input: String::new(), route: Route::Compress, out: None, zstd_level: 3, threads: 4,
        ring_bytes: 64 << 20,
    };
    let mut index = 0;
    while index < arguments.len() {
        let flag = arguments[index].as_str();
        index += 1;
        let mut value = || -> Result<&String, String> {
            let found = arguments.get(index).ok_or(format!("{flag} necesita un valor"))?;
            index += 1;
            Ok(found)
        };
        match flag {
            "--route" => options.route = match value()?.as_str() {
                "compress" => Route::Compress,
                "discard" => Route::Discard,
                "check" => Route::Check,
                other => return Err(format!("Ruta desconocida: {other}")),
            },
            "--out" => options.out = Some(value()?.clone()),
            "--zstd" => options.zstd_level = value()?.parse().map_err(|_| "Nivel zstd no valido")?,
            "--threads" => options.threads = value()?.parse().map_err(|_| "Numero de hilos no valido")?,
            "--ring-mib" => {
                let mebibytes: usize = value()?.parse().map_err(|_| "Tamano de anillo no valido")?;
                if !mebibytes.is_power_of_two() || mebibytes > 1 << 14 {
                    return Err("--ring-mib debe ser una potencia de dos de MiB".into());
                }
                options.ring_bytes = mebibytes << 20;
            }
            _ if flag.starts_with("--") => return Err(format!("Opcion desconocida: {flag}")),
            _ if options.input.is_empty() => options.input = flag.to_owned(),
            _ => return Err(format!("Sobra el argumento {flag}")),
        }
    }
    if options.input.is_empty() {
        return Err("Uso: chronos_certify <formula.cnf> [--route compress|discard|check] [--out F] [--zstd N] [--threads T] [--ring-mib M]".into());
    }
    Ok(options)
}

fn run(options: &Options) -> Result<i32, String> {
    let started = Instant::now();
    let text = std::fs::read_to_string(&options.input).map_err(|error| format!("{}: {error}", options.input))?;
    let formula = parse_dimacs(&text)?;
    drop(text);
    let route_name = match options.route {
        Route::Compress => "compress",
        Route::Discard => "discard",
        Route::Check => "check",
    };
    eprintln!(
        "c {}: {} clausulas, ruta {route_name}, anillo de {} MiB",
        options.input, formula.len(), options.ring_bytes >> 20,
    );

    let bridge = Instant::now();
    // (bytes drenados del anillo, resumen de la ruta)
    let (solved, drained, detail) = if options.route == Route::Compress {
        let out = options.out.clone()
            .unwrap_or_else(|| format!("{}.pbp.zst", options.input.trim_end_matches(".cnf")));
        let config = ZstdConfig { level: options.zstd_level, workers: options.threads, checksum: true };
        let (ring_writer, sink) = chronos_vfs::zstd_bridge(&out, options.ring_bytes, config)
            .map_err(|error| format!("{out}: {error}"))?;
        let solved = solve(&formula, HybridConfig::default(), ring_writer);
        // Si el consumidor murio (disco lleno, p. ej.), su error es la causa y sale aqui.
        let stats = sink.join().expect("Fallo crítico en hilo ZSTD");
        let detail = format!(
            "  compress: {} -> {} bytes ({:.2}x) en {out}; comprobable con veripb sobre el .zst",
            stats.raw_bytes, stats.compressed_bytes,
            stats.raw_bytes as f64 / stats.compressed_bytes.max(1) as f64,
        );
        (solved?, stats.raw_bytes, detail)
    } else {
        let (producer, consumer) = RingBuffer::with_capacity(options.ring_bytes);
        let check = options.route == Route::Check;
        let drain = std::thread::spawn(move || consume(consumer, check));
        let solved = solve(&formula, HybridConfig::default(), RingWriter::new(producer));
        let tally = drain.join().map_err(|_| "El hilo consumidor entro en panico")?;
        let detail = if check {
            format!(
                "  check: {} lineas = {} rup + {} red + {} dom + {} del + {} otras, en {} lotes",
                tally.lines, tally.rup, tally.red, tally.dom, tally.del, tally.other, tally.batches,
            )
        } else {
            format!("  discard: hash_fnv1a=0x{:016x} en {} lotes (prueba de consumo, no de validez)", tally.hash, tally.batches)
        };
        (solved?, tally.bytes, detail)
    };
    let seconds = bridge.elapsed().as_secs_f64();

    println!(
        "ruta={route_name} estado={} traza_bytes={drained} MB/s={:.1} esperas_contrapresion={} segundos={seconds:.3} total_segundos={:.3}",
        if solved.satisfiable { "SAT" } else { "UNSAT" }, drained as f64 / 1e6 / seconds, solved.stalls,
        started.elapsed().as_secs_f64(),
    );
    println!("{detail}");
    eprintln!(
        "c motor: {} decisiones, {} conflictos, {} dominancias",
        solved.decisions, solved.conflicts, solved.dominances,
    );
    // Codigos de salida de la competicion SAT.
    Ok(if solved.satisfiable { 10 } else { 20 })
}

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match parse_options(&arguments).and_then(|options| run(&options)) {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // La misma traza, escrita directamente en memoria sin pasar por el anillo.
    fn direct(formula: &Formula, config: HybridConfig) -> Vec<u8> {
        let mut proof = PbProof::new(Vec::new(), formula.len());
        solve_hybrid_with_proof(formula, config, &mut proof);
        proof.finish().unwrap()
    }

    #[test]
    fn the_ring_delivers_the_trace_byte_for_byte() {
        let search = HybridConfig { symmetry: false, ..HybridConfig::default() };
        for (formula, config) in [(pigeonhole(10, 9), HybridConfig::default()), (pigeonhole(7, 6), search)] {
            let expected = direct(&formula, config);
            // Dos bloques de anillo: la traza lo envuelve muchas veces y hay contrapresion.
            assert!(expected.len() > 3 * 2 * CHUNK_SIZE);
            let (producer, mut consumer) = RingBuffer::with_capacity(2 * CHUNK_SIZE);
            let drain = std::thread::spawn(move || {
                let mut received = Vec::new();
                while consumer.wait_for_data() {
                    let (front, back) = consumer.readable();
                    received.extend_from_slice(front);
                    received.extend_from_slice(back);
                    let count = front.len() + back.len();
                    consumer.release(count);
                }
                received
            });
            let solved = solve(&formula, config, RingWriter::new(producer)).unwrap();
            assert!(!solved.satisfiable);
            assert!(drain.join().unwrap() == expected, "El anillo altero la traza");
        }
    }

    #[test]
    fn the_consumer_routes_see_every_byte_and_every_line() {
        let formula = pigeonhole(7, 6);
        let config = HybridConfig { symmetry: false, ..HybridConfig::default() };
        let expected = direct(&formula, config);
        for check in [false, true] {
            let (producer, consumer) = RingBuffer::with_capacity(4 * CHUNK_SIZE);
            let drain = std::thread::spawn(move || consume(consumer, check));
            solve(&formula, config, RingWriter::new(producer)).unwrap();
            let tally = drain.join().unwrap();
            assert_eq!(tally.bytes, expected.len() as u64);
            if check {
                let lines: Vec<&[u8]> = expected.split(|&byte| byte == b'\n').filter(|line| !line.is_empty()).collect();
                let count = |rule: &[u8]| lines.iter().filter(|line| line.starts_with(rule)).count() as u64;
                assert_eq!(tally.lines, lines.len() as u64);
                assert_eq!((tally.rup, tally.del), (count(b"rup "), count(b"del ")));
                assert!(tally.rup > 0 && tally.del > 0);
                assert_eq!(tally.rup + tally.red + tally.dom + tally.del + tally.other, tally.lines);
            } else {
                let hash = expected.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, &byte| {
                    (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                });
                assert_eq!(tally.hash, hash);
            }
        }
    }

    #[test]
    fn the_zstd_sink_accounts_for_every_byte_of_the_trace() {
        let formula = pigeonhole(7, 6);
        let config = HybridConfig { symmetry: false, ..HybridConfig::default() };
        let expected = direct(&formula, config);
        let path = std::env::temp_dir().join(format!("chronos_certify_{}.pbp.zst", std::process::id()));
        let zstd = ZstdConfig { level: 3, workers: 2, checksum: true };
        let (ring_writer, sink) = chronos_vfs::zstd_bridge(&path, 2 * CHUNK_SIZE, zstd).unwrap();
        solve(&formula, config, ring_writer).unwrap();
        let stats = sink.join().unwrap();
        assert_eq!(stats.raw_bytes, expected.len() as u64);
        let file = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(file.len() as u64, stats.compressed_bytes);
        // Numero magico de un frame zstd, y una traza de texto repetitivo comprime.
        assert_eq!(file[..4], [0x28, 0xb5, 0x2f, 0xfd]);
        assert!(file.len() < expected.len() / 2);
    }

    #[test]
    fn options_default_to_the_certifiable_route() {
        let arguments = |text: &str| -> Vec<String> { text.split_whitespace().map(str::to_owned).collect() };
        let options = parse_options(&arguments("f.cnf")).unwrap();
        assert!(options.route == Route::Compress);
        assert_eq!((options.zstd_level, options.threads, options.ring_bytes), (3, 4, 64 << 20));
        let options = parse_options(&arguments("f.cnf --route check --ring-mib 8 --out t.zst")).unwrap();
        assert!(options.route == Route::Check && options.ring_bytes == 8 << 20);
        assert_eq!(options.out.as_deref(), Some("t.zst"));
        assert!(parse_options(&arguments("f.cnf --ring-mib 3")).is_err());
        assert!(parse_options(&arguments("f.cnf --route disk")).is_err());
        assert!(parse_options(&arguments("--route check")).is_err());
    }
}
