// Fase 17: generador de Ramsey. Escribe en DIMACS la formula cuyos modelos son exactamente
// las coloraciones con dos colores de las aristas de K_N sin K_5 monocromatico: es
// satisfacible si y solo si N < R(5,5).
//
//   gen_ramsey <N> [salida.cnf] [--cliques <S> <T>]
//
// Sin fichero de salida escribe en la salida estandar. `--cliques S T` cambia los tamanos
// prohibidos (sin K_S del color 1 y sin K_T del color 0), para calibrar con numeros de
// Ramsey conocidos: R(3,3)=6, R(3,4)=9, R(3,5)=14, R(4,4)=18, R(3,6)=18.
//
// Variables: la arista {i,j}, con 0 <= i < j < N, es la variable i*N - i*(i+1)/2 + (j-i),
// es decir, las aristas en orden lexicografico desde 1. Verdadero es el color 1.
// Clausulas: por cada S-subconjunto de vertices, alguna de sus aristas no es del color 1;
// por cada T-subconjunto, alguna de sus aristas es del color 1. No se anade ninguna
// ruptura de simetria: eso es trabajo del motor, que ademas la certifica.
#![forbid(unsafe_code)]
#![deny(warnings)]

use std::fs::File;
use std::io::{self, BufWriter, Write};

fn edge(vertices: usize, first: usize, second: usize) -> usize {
    debug_assert!(first < second && second < vertices);
    first * vertices - first * (first + 1) / 2 + (second - first)
}

// Satura en lugar de desbordar: quien llama rechaza las formulas desmesuradas.
fn binomial(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    // Cada producto parcial es un coeficiente binomial: la division es exacta.
    let value = (0..k.min(n - k)).fold(1_u128, |value, index| {
        value.saturating_mul((n - index) as u128) / (index as u128 + 1)
    });
    u64::try_from(value).unwrap_or(u64::MAX)
}

// Una clausula por cada subconjunto de `size` vertices, con sus aristas en el signo dado.
fn forbid<W: Write>(out: &mut W, vertices: usize, size: usize, literals: &[Vec<u8>]) -> io::Result<u64> {
    if size > vertices {
        return Ok(0);
    }
    let mut chosen: Vec<usize> = (0..size).collect();
    let mut clauses = 0;
    loop {
        for (position, &first) in chosen.iter().enumerate() {
            for &second in &chosen[position + 1..] {
                out.write_all(&literals[edge(vertices, first, second)])?;
            }
        }
        out.write_all(b"0\n")?;
        clauses += 1;
        // Siguiente subconjunto en orden lexicografico.
        let Some(pivot) = (0..size).rev().find(|&index| chosen[index] != index + vertices - size) else {
            return Ok(clauses);
        };
        chosen[pivot] += 1;
        for index in pivot + 1..size {
            chosen[index] = chosen[index - 1] + 1;
        }
    }
}

// Devuelve (variables, clausulas) de la formula escrita.
fn write_formula<W: Write>(out: &mut W, vertices: usize, first: usize, second: usize) -> io::Result<(u64, u64)> {
    let variables = binomial(vertices, 2);
    let expected = binomial(vertices, first).saturating_add(binomial(vertices, second));
    writeln!(out, "c Ramsey: 2-coloraciones de las aristas de K_{vertices} sin K_{first} del color 1 ni K_{second} del color 0")?;
    writeln!(out, "c arista {{i,j}}, 0 <= i < j < {vertices}: variable i*{vertices} - i*(i+1)/2 + (j-i); verdadero = color 1")?;
    writeln!(out, "p cnf {variables} {expected}")?;
    let literal = |sign: &str| -> Vec<Vec<u8>> {
        (0..=variables).map(|variable| format!("{sign}{variable} ").into_bytes()).collect()
    };
    let clauses = forbid(out, vertices, first, &literal("-"))? + forbid(out, vertices, second, &literal(""))?;
    assert_eq!(clauses, expected);
    Ok((variables, clauses))
}

fn run(arguments: &[String]) -> Result<(), String> {
    let usage = "Uso: gen_ramsey <N> [salida.cnf] [--cliques <S> <T>]";
    let number = |text: &String| text.parse::<usize>().map_err(|_| format!("Numero no valido: {text}\n{usage}"));
    let (mut vertices, mut output, mut sizes) = (None, None, (5, 5));
    let mut index = 0;
    while index < arguments.len() {
        if arguments[index] == "--cliques" {
            let (Some(first), Some(second)) = (arguments.get(index + 1), arguments.get(index + 2)) else {
                return Err(usage.into());
            };
            sizes = (number(first)?, number(second)?);
            index += 3;
            continue;
        }
        if vertices.is_none() {
            vertices = Some(number(&arguments[index])?);
        } else if output.is_none() {
            output = Some(arguments[index].clone());
        } else {
            return Err(usage.into());
        }
        index += 1;
    }
    let Some(vertices) = vertices else { return Err(usage.into()) };
    // Un clique de menos de dos vertices no tiene aristas: daria la clausula vacia.
    if sizes.0 < 2 || sizes.1 < 2 {
        return Err("Los cliques prohibidos tienen al menos dos vertices".into());
    }
    // N = 43 con K_5 son 903 variables y 1 925 196 clausulas.
    let clauses = binomial(vertices, sizes.0).saturating_add(binomial(vertices, sizes.1));
    if binomial(vertices, 2) > i32::MAX as u64 || clauses > 1 << 36 {
        return Err("La formula no cabe en un fichero DIMACS razonable".into());
    }
    let failed = |error: io::Error| error.to_string();
    let (variables, clauses) = match &output {
        Some(path) => {
            let mut out = BufWriter::with_capacity(1 << 20, File::create(path).map_err(failed)?);
            let counts = write_formula(&mut out, vertices, sizes.0, sizes.1).map_err(failed)?;
            out.flush().map_err(failed)?;
            counts
        }
        None => {
            let mut out = BufWriter::with_capacity(1 << 20, io::stdout().lock());
            let counts = write_formula(&mut out, vertices, sizes.0, sizes.1).map_err(failed)?;
            out.flush().map_err(failed)?;
            counts
        }
    };
    eprintln!("c K_{vertices}, cliques ({}, {}): {variables} variables, {clauses} clausulas", sizes.0, sizes.1);
    Ok(())
}

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if let Err(error) = run(&arguments) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn formula(vertices: usize, first: usize, second: usize) -> (u64, Vec<Vec<i32>>) {
        let mut bytes = Vec::new();
        let (variables, clauses) = write_formula(&mut bytes, vertices, first, second).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        let parsed: Vec<Vec<i32>> = text.lines().filter(|line| !line.starts_with(['c', 'p'])).map(|line| {
            let mut literals: Vec<i32> = line.split_whitespace().map(|token| token.parse().unwrap()).collect();
            assert_eq!(literals.pop(), Some(0));
            literals
        }).collect();
        assert_eq!(parsed.len() as u64, clauses);
        assert!(text.contains(&format!("p cnf {variables} {clauses}\n")));
        (variables, parsed)
    }

    fn satisfiable(variables: u64, clauses: &[Vec<i32>]) -> bool {
        (0..1_u64 << variables).any(|bits| clauses.iter().all(|clause| {
            clause.iter().any(|&literal| (bits >> (literal.unsigned_abs() - 1) & 1 == 1) == (literal > 0))
        }))
    }

    #[test]
    fn edges_are_numbered_consecutively_from_one() {
        for vertices in [2, 3, 7, 43] {
            let mut expected = 0;
            for first in 0..vertices {
                for second in first + 1..vertices {
                    expected += 1;
                    assert_eq!(edge(vertices, first, second), expected);
                }
            }
            assert_eq!(expected as u64, binomial(vertices, 2));
        }
    }

    #[test]
    fn the_formula_has_two_clauses_of_ten_edges_per_five_vertices() {
        let (variables, clauses) = formula(9, 5, 5);
        assert_eq!((variables, clauses.len() as u64), (36, 2 * binomial(9, 5)));
        let mut seen = std::collections::BTreeSet::new();
        for clause in &clauses {
            assert_eq!(clause.len(), 10);
            // Diez aristas distintas, todas del mismo signo, sobre cinco vertices.
            assert!(clause.iter().all(|&literal| (literal > 0) == (clause[0] > 0)));
            let edges: std::collections::BTreeSet<u32> = clause.iter().map(|literal| literal.unsigned_abs()).collect();
            assert_eq!(edges.len(), 10);
            assert!(seen.insert((clause[0] > 0, edges)));
        }
        assert_eq!((binomial(43, 2), 2 * binomial(43, 5)), (903, 1_925_196));
    }

    #[test]
    fn small_ramsey_numbers_are_reproduced_exhaustively() {
        // R(3,3) = 6.
        let (variables, clauses) = formula(5, 3, 3);
        assert!(satisfiable(variables, &clauses));
        let (variables, clauses) = formula(6, 3, 3);
        assert!(!satisfiable(variables, &clauses));
        // R(2,k) = k: sin aristas del color 1, todo K_k es del color 0.
        let (variables, clauses) = formula(3, 2, 4);
        assert!(satisfiable(variables, &clauses));
        let (variables, clauses) = formula(4, 2, 4);
        assert!(!satisfiable(variables, &clauses));
        // R(3,4) = 9 por debajo: K_6 admite una coloracion.
        let (variables, clauses) = formula(6, 3, 4);
        assert!(satisfiable(variables, &clauses));
        // Menos vertices que el clique prohibido: ninguna clausula.
        let (variables, clauses) = formula(4, 5, 5);
        assert_eq!((variables, clauses.len()), (6, 0));
    }
}
