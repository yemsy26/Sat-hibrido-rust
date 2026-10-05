// Fase 17: generador de la Discrepancia de Erdos. Escribe en DIMACS la formula cuyos
// modelos son las secuencias x_1..x_N en {+1,-1} tales que para todo paso d >= 1 y toda
// longitud k con d*k <= N, |x_d + x_2d + ... + x_kd| <= C.
//
//   gen_erdos <N> [salida.cnf] [--c <C>]
//
// Variable i (1..N) es x_i; verdadero = +1. A lo largo de cada progresion d se sigue la
// suma parcial con un automata de estados one-hot: variable auxiliar S[d][t][v] = "la suma
// tras t terminos vale v", con v en [-C, C]. Las clausulas propagan el estado real hacia
// delante desde la suma 0 y prohiben el paso que saldria de [-C, C]; por induccion la
// cadena de estados reales queda forzada por propagacion unitaria, asi que la prohibicion
// salta exactamente cuando la discrepancia supera C. Los estados auxiliares sobrantes no
// afectan a x. Para C=2 existen secuencias hasta N=1160 (Konev-Lisitsa 2014), de modo que
// N=500 es satisfacible.
//
// Solo se codifican las progresiones con floor(N/d) >= C+1: con menos terminos la suma no
// puede alcanzar C+1.
#![forbid(unsafe_code)]
#![deny(warnings)]

use std::fs::File;
use std::io::{self, BufWriter, Write};

struct Builder {
    variables: usize,
    clauses: Vec<Vec<i32>>,
}

impl Builder {
    fn new(sequence: usize) -> Self {
        Self { variables: sequence, clauses: Vec::new() }
    }
    fn fresh(&mut self) -> i32 {
        self.variables += 1;
        self.variables as i32
    }
    fn clause(&mut self, literals: Vec<i32>) {
        self.clauses.push(literals);
    }
}

// Construye la formula; devuelve el constructor con las clausulas listas.
fn build(sequence: usize, bound: usize) -> Builder {
    assert!(bound >= 1, "La cota de discrepancia es al menos 1");
    let mut builder = Builder::new(sequence);
    let signed = bound as i32;
    for step in 1..=sequence {
        let terms = sequence / step;
        if terms < bound + 1 {
            continue; // No puede alcanzar C+1: progresion trivial.
        }
        // state[v + C] guarda el literal de "suma actual = v"; None mientras no exista.
        let width = 2 * bound + 1;
        let mut previous: Vec<Option<i32>> = vec![None; width];
        for position in 1..=terms {
            let term = (step * position) as i32; // x_{d*position}
            let mut current: Vec<Option<i32>> = vec![None; width];
            // get_or_create para el estado v en el nivel actual.
            let ensure = |builder: &mut Builder, current: &mut Vec<Option<i32>>, value: i32| -> i32 {
                let slot = (value + signed) as usize;
                if let Some(literal) = current[slot] {
                    literal
                } else {
                    let literal = builder.fresh();
                    current[slot] = Some(literal);
                    literal
                }
            };
            if position == 1 {
                // Desde la suma 0: +1 lleva a estado +1, -1 a estado -1.
                let up = ensure(&mut builder, &mut current, 1);
                let down = ensure(&mut builder, &mut current, -1);
                builder.clause(vec![-term, up]); // x => suma +1
                builder.clause(vec![term, down]); // ¬x => suma -1
            } else {
                for value in -signed..=signed {
                    let Some(state) = previous[(value + signed) as usize] else { continue };
                    // Paso +1 (x verdadero).
                    if value + 1 <= signed {
                        let next = ensure(&mut builder, &mut current, value + 1);
                        builder.clause(vec![-state, -term, next]);
                    } else {
                        builder.clause(vec![-state, -term]); // Saldria por arriba: prohibido.
                    }
                    // Paso -1 (x falso).
                    if value - 1 >= -signed {
                        let next = ensure(&mut builder, &mut current, value - 1);
                        builder.clause(vec![-state, term, next]);
                    } else {
                        builder.clause(vec![-state, term]); // Saldria por abajo: prohibido.
                    }
                }
            }
            previous = current;
        }
    }
    builder
}

fn write_formula<W: Write>(out: &mut W, sequence: usize, bound: usize) -> io::Result<(usize, usize)> {
    let builder = build(sequence, bound);
    writeln!(out, "c Discrepancia de Erdos: secuencia +-1 de longitud {sequence} con discrepancia <= {bound}")?;
    writeln!(out, "c variable i (1..{sequence}) es x_i, verdadero = +1; el resto son estados auxiliares del automata")?;
    writeln!(out, "p cnf {} {}", builder.variables, builder.clauses.len())?;
    for clause in &builder.clauses {
        for literal in clause {
            write!(out, "{literal} ")?;
        }
        writeln!(out, "0")?;
    }
    Ok((builder.variables, builder.clauses.len()))
}

fn run(arguments: &[String]) -> Result<(), String> {
    let usage = "Uso: gen_erdos <N> [salida.cnf] [--c <C>]";
    let number = |text: &String| text.parse::<usize>().map_err(|_| format!("Numero no valido: {text}\n{usage}"));
    let (mut sequence, mut output, mut bound) = (None, None, 2usize);
    let mut index = 0;
    while index < arguments.len() {
        if arguments[index] == "--c" {
            bound = number(arguments.get(index + 1).ok_or("--c necesita un valor")?)?;
            index += 2;
            continue;
        }
        if sequence.is_none() {
            sequence = Some(number(&arguments[index])?);
        } else if output.is_none() {
            output = Some(arguments[index].clone());
        } else {
            return Err(usage.into());
        }
        index += 1;
    }
    let Some(sequence) = sequence else { return Err(usage.into()) };
    if sequence == 0 || bound == 0 {
        return Err("N y C son positivos".into());
    }
    let failed = |error: io::Error| error.to_string();
    let (variables, clauses) = match &output {
        Some(path) => {
            let mut out = BufWriter::with_capacity(1 << 20, File::create(path).map_err(failed)?);
            let counts = write_formula(&mut out, sequence, bound).map_err(failed)?;
            out.flush().map_err(failed)?;
            counts
        }
        None => {
            let mut out = BufWriter::with_capacity(1 << 20, io::stdout().lock());
            write_formula(&mut out, sequence, bound).map_err(failed)?
        }
    };
    eprintln!("c Erdos N={sequence} C={bound}: {variables} variables ({sequence} de secuencia), {clauses} clausulas");
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

    // Discrepancia real de una secuencia (true=+1): max_{d,k} |suma de los primeros k de la
    // progresion de paso d|.
    fn discrepancy(sequence: &[bool]) -> usize {
        let n = sequence.len();
        let mut worst = 0_i32;
        for step in 1..=n {
            let mut sum = 0_i32;
            let mut position = step;
            while position <= n {
                sum += if sequence[position - 1] { 1 } else { -1 };
                worst = worst.max(sum.abs());
                position += step;
            }
        }
        worst as usize
    }

    // Evalua la formula fijando x a la secuencia dada y los auxiliares a su valor forzado
    // (el automata es determinista): satisfacible con esta x sii ninguna prohibicion salta.
    // Como los auxiliares quedan forzados por propagacion, basta comprobar que el camino
    // real nunca sale de [-C, C], que es justo lo que mide `discrepancy`.
    fn encoding_matches_brute_force(n: usize, bound: usize) {
        for bits in 0..1_u32 << n {
            let sequence: Vec<bool> = (0..n).map(|i| bits >> i & 1 == 1).collect();
            let satisfiable = discrepancy(&sequence) <= bound;
            // Reconstruye el modelo canonico y comprueba TODAS las clausulas directamente.
            assert_eq!(satisfiable, canonical_model_satisfies(n, bound, &sequence),
                "N={n} C={bound} secuencia {sequence:?}");
        }
    }

    // Reconstruye el modelo canonico (x + camino real) y evalua cada clausula. Devuelve
    // false solo si alguna prohibicion no puede satisfacerse, es decir, si x excede C.
    fn canonical_model_satisfies(n: usize, bound: usize, sequence: &[bool]) -> bool {
        let builder = build(n, bound);
        // Valor de cada variable: x_i conocido; un auxiliar es true sii su estado coincide
        // con el camino real. Recomputamos el camino real para asignar los auxiliares.
        let mut values = vec![false; builder.variables + 1];
        for (i, &bit) in sequence.iter().enumerate() {
            values[i + 1] = bit;
        }
        // Repite la construccion siguiendo los ids para marcar el estado real de cada celda.
        let signed = bound as i32;
        let mut next_id = n;
        for step in 1..=n {
            let terms = n / step;
            if terms < bound + 1 {
                continue;
            }
            let width = 2 * bound + 1;
            let mut previous: Vec<Option<i32>> = vec![None; width];
            let mut real_prev = 0_i32;
            for position in 1..=terms {
                let mut current: Vec<Option<i32>> = vec![None; width];
                let mut ids = std::collections::HashMap::new();
                let mut ensure = |value: i32| -> i32 {
                    let slot = (value + signed) as usize;
                    if let Some(&literal) = ids.get(&slot) {
                        return literal;
                    }
                    next_id += 1;
                    ids.insert(slot, next_id as i32);
                    next_id as i32
                };
                let bit = sequence[step * position - 1];
                let real_now = real_prev + if bit { 1 } else { -1 };
                if position == 1 {
                    let up = ensure(1);
                    let down = ensure(-1);
                    current[(1 + signed) as usize] = Some(up);
                    current[(-1 + signed) as usize] = Some(down);
                } else {
                    for value in -signed..=signed {
                        if previous[(value + signed) as usize].is_none() {
                            continue;
                        }
                        if value + 1 <= signed {
                            let next = ensure(value + 1);
                            current[(value + 1 + signed) as usize] = Some(next);
                        }
                        if value - 1 >= -signed {
                            let next = ensure(value - 1);
                            current[(value - 1 + signed) as usize] = Some(next);
                        }
                    }
                }
                // El estado real, si cabe en [-C,C], se marca true.
                if real_now.abs() <= signed {
                    if let Some(literal) = current[(real_now + signed) as usize] {
                        values[literal as usize] = true;
                    }
                }
                previous = current;
                real_prev = real_now;
            }
        }
        builder.clauses.iter().all(|clause| {
            clause.iter().any(|&literal| values[literal.unsigned_abs() as usize] == (literal > 0))
        })
    }

    #[test]
    fn the_encoding_agrees_with_brute_force_discrepancy() {
        for n in 1..=11 {
            encoding_matches_brute_force(n, 1);
        }
        for n in 1..=11 {
            encoding_matches_brute_force(n, 2);
        }
    }

    #[test]
    fn discrepancy_one_is_unsatisfiable_at_twelve() {
        // Erdos: ninguna secuencia +-1 de longitud 12 tiene discrepancia 1.
        assert!((0..1_u32 << 12).all(|bits| {
            let sequence: Vec<bool> = (0..12).map(|i| bits >> i & 1 == 1).collect();
            discrepancy(&sequence) > 1
        }));
        // Y hay al menos una de longitud 11 con discrepancia 1.
        assert!((0..1_u32 << 11).any(|bits| {
            let sequence: Vec<bool> = (0..11).map(|i| bits >> i & 1 == 1).collect();
            discrepancy(&sequence) <= 1
        }));
    }

    #[test]
    fn trivial_progressions_are_skipped() {
        // Con C=2, solo los pasos d con floor(N/d) >= 3 generan clausulas.
        let builder = build(5, 2);
        // d=1 (5 terminos) y d=2... floor(5/2)=2 < 3, no. Solo d=1 aporta.
        assert!(builder.variables > 5);
        let small = build(2, 2);
        assert_eq!(small.clauses.len(), 0);
        assert_eq!(small.variables, 2);
    }
}
