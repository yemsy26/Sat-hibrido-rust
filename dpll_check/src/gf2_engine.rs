// Mejora 1: nucleo matricial sobre GF(2). Un sistema de ecuaciones XOR
//
//     x_a ^ x_b ^ ... = b
//
// se guarda como matriz aumentada densa de bits, una fila por ecuacion, y se reduce por
// Gauss-Jordan sumando filas con XOR palabra a palabra. La forma reducida da el rango, las
// variables que quedan forzadas y las filas 0 = 1, que demuestran que el sistema no tiene
// solucion.
//
// En esta fase es una herramienta aislada: no lee ni toca el CDCL. El motor ya hace una
// reduccion gaussiana como preproceso (hybrid_parity.rs); este modulo es la base de un
// propagador que viva dentro de la busqueda.
//
// Apagado (`new(false)`) no reserva memoria y todas sus operaciones vuelven sin hacer nada.
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gf2Outcome {
    // El modulo esta apagado: no se ha mirado nada.
    Disabled,
    // El sistema tiene solucion; `rank` ecuaciones son independientes.
    Consistent { rank: usize },
    // Alguna combinacion de las ecuaciones da 0 = 1.
    Contradiction,
}

pub struct Gf2Propagator {
    enabled: bool,
    // Variable de cada columna, por orden de primera aparicion, y su inversa.
    variables: Vec<u32>,
    column_of: HashMap<u32, u32>,
    // Ecuaciones tal como llegan: las columnas de todas, seguidas; donde acaba cada una; y
    // su termino independiente.
    terms: Vec<u32>,
    ends: Vec<u32>,
    constants: Vec<bool>,
    // Forma reducida: `words` palabras de 64 bits por fila. El bit de la columna
    // `variables.len()` es el termino independiente.
    words: usize,
    matrix: Vec<u64>,
    // Columna pivote de cada fila de la forma reducida.
    pivots: Vec<u32>,
    inconsistent: bool,
}

impl Gf2Propagator {
    pub fn new(enabled: bool) -> Self {
        // Ninguno de estos contenedores reserva memoria hasta recibir su primer elemento.
        Self {
            enabled, variables: Vec::new(), column_of: HashMap::new(), terms: Vec::new(),
            ends: Vec::new(), constants: Vec::new(), words: 0, matrix: Vec::new(),
            pivots: Vec::new(), inconsistent: false,
        }
    }
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    pub fn equations(&self) -> usize {
        self.ends.len()
    }
    pub fn variables(&self) -> usize {
        self.variables.len()
    }
    // Filas no nulas de la forma reducida calculada por la ultima `eliminate`.
    pub fn rank(&self) -> usize {
        self.pivots.len()
    }
    // Anade la ecuacion: el XOR de `variables` vale `constant`. Una variable repetida se
    // cancela consigo misma; una ecuacion sin variables y con `constant` es ya 0 = 1.
    pub fn add_xor(&mut self, variables: &[u32], constant: bool) {
        if !self.enabled {
            return;
        }
        let start = self.terms.len();
        for &variable in variables {
            let next = self.variables.len() as u32;
            let column = *self.column_of.entry(variable).or_insert(next);
            if column == next {
                self.variables.push(variable);
            }
            match self.terms[start..].iter().position(|&seen| seen == column) {
                Some(position) => {
                    self.terms.swap_remove(start + position);
                }
                None => self.terms.push(column),
            }
        }
        self.ends.push(self.terms.len() as u32);
        self.constants.push(constant);
    }
    // Gauss-Jordan sobre todas las ecuaciones recibidas. Deja la forma escalonada reducida:
    // cada fila tiene un pivote que no aparece en ninguna otra.
    pub fn eliminate(&mut self) -> Gf2Outcome {
        if !self.enabled {
            return Gf2Outcome::Disabled;
        }
        let columns = self.variables.len();
        // Un bit mas que columnas: el termino independiente.
        let words = columns / 64 + 1;
        let rows = self.ends.len();
        let mut matrix = vec![0_u64; rows * words];
        let mut start = 0;
        for (row, (&end, &constant)) in self.ends.iter().zip(&self.constants).enumerate() {
            let bits = &mut matrix[row * words..(row + 1) * words];
            for &column in &self.terms[start..end as usize] {
                bits[column as usize / 64] |= 1 << (column % 64);
            }
            if constant {
                bits[columns / 64] |= 1 << (columns % 64);
            }
            start = end as usize;
        }
        self.pivots.clear();
        let mut rank = 0;
        for column in 0..columns {
            let (word, mask) = (column / 64, 1_u64 << (column % 64));
            let Some(found) = (rank..rows).find(|&row| matrix[row * words + word] & mask != 0) else {
                continue;
            };
            if found != rank {
                for offset in 0..words {
                    matrix.swap(found * words + offset, rank * words + offset);
                }
            }
            // La fila pivote se suma a todas las demas que tengan la columna: por encima
            // y por debajo, que es lo que distingue a Gauss-Jordan de la triangulacion.
            let (above, rest) = matrix.split_at_mut(rank * words);
            let (pivot, below) = rest.split_at_mut(words);
            for other in above.chunks_exact_mut(words).chain(below.chunks_exact_mut(words)) {
                if other[word] & mask != 0 {
                    for (target, &source) in other.iter_mut().zip(pivot.iter()) {
                        *target ^= source;
                    }
                }
            }
            self.pivots.push(column as u32);
            rank += 1;
        }
        // Bajo el rango ya no quedan coeficientes: una fila con termino independiente es 0 = 1.
        let constant = |row: usize| matrix[row * words + columns / 64] >> (columns % 64) & 1 == 1;
        self.inconsistent = (rank..rows).any(constant);
        matrix.truncate(rank * words);
        self.matrix = matrix;
        self.words = words;
        if self.inconsistent { Gf2Outcome::Contradiction } else { Gf2Outcome::Consistent { rank } }
    }
    // Fila `index` de la forma reducida: sus variables, con el pivote entre ellas, y su
    // termino independiente.
    pub fn row(&self, index: usize) -> (Vec<u32>, bool) {
        let bits = &self.matrix[index * self.words..(index + 1) * self.words];
        let columns = self.variables.len();
        let mut variables = Vec::new();
        for (word, &value) in bits.iter().enumerate() {
            let mut rest = value;
            while rest != 0 {
                let column = word * 64 + rest.trailing_zeros() as usize;
                if column < columns {
                    variables.push(self.variables[column]);
                }
                rest &= rest - 1;
            }
        }
        (variables, bits[columns / 64] >> (columns % 64) & 1 == 1)
    }
    // Variables que la forma reducida deja forzadas, con su valor: filas de una sola variable.
    pub fn units(&self) -> Vec<(u32, bool)> {
        (0..self.rank()).filter_map(|index| {
            let (variables, constant) = self.row(index);
            (variables.len() == 1).then(|| (variables[0], constant))
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    // Asignaciones (bit i = variable i + 1) que cumplen todas las ecuaciones.
    fn solutions(variables: u32, system: &[(Vec<u32>, bool)]) -> Vec<u32> {
        (0..1_u32 << variables).filter(|bits| {
            system.iter().all(|(row, constant)| {
                row.iter().fold(false, |parity, &variable| parity ^ (bits >> (variable - 1) & 1 == 1)) == *constant
            })
        }).collect()
    }

    #[test]
    fn a_disabled_module_does_nothing_and_holds_nothing() {
        let mut gf2 = Gf2Propagator::new(false);
        gf2.add_xor(&[1, 2, 3], true);
        gf2.add_xor(&[], true);
        assert_eq!(gf2.eliminate(), Gf2Outcome::Disabled);
        assert_eq!((gf2.equations(), gf2.variables(), gf2.rank()), (0, 0, 0));
        assert!(gf2.units().is_empty() && !gf2.is_enabled());
        let reserved = gf2.variables.capacity() + gf2.column_of.capacity() + gf2.terms.capacity()
            + gf2.ends.capacity() + gf2.constants.capacity() + gf2.matrix.capacity() + gf2.pivots.capacity();
        assert_eq!(reserved, 0);
    }

    #[test]
    fn an_odd_cycle_is_a_contradiction() {
        // a ^ b = 1, b ^ c = 1, a ^ c = 1: la suma de las tres es 0 = 1.
        let mut gf2 = Gf2Propagator::new(true);
        gf2.add_xor(&[10, 20], true);
        gf2.add_xor(&[20, 30], true);
        assert_eq!(gf2.eliminate(), Gf2Outcome::Consistent { rank: 2 });
        gf2.add_xor(&[10, 30], true);
        assert_eq!(gf2.eliminate(), Gf2Outcome::Contradiction);
        assert_eq!((gf2.equations(), gf2.variables(), gf2.rank()), (3, 3, 2));
    }

    #[test]
    fn elimination_exposes_forced_variables() {
        // x1 ^ x2 = 1 y x1 ^ x2 ^ x3 = 0 fuerzan x3 = 1.
        let mut gf2 = Gf2Propagator::new(true);
        gf2.add_xor(&[1, 2], true);
        gf2.add_xor(&[1, 2, 3], false);
        assert_eq!(gf2.eliminate(), Gf2Outcome::Consistent { rank: 2 });
        assert_eq!(gf2.units(), vec![(3, true)]);
        assert_eq!(gf2.row(0), (vec![1, 2], true));
    }

    #[test]
    fn repeated_variables_cancel_and_an_empty_equation_can_refute() {
        let mut gf2 = Gf2Propagator::new(true);
        gf2.add_xor(&[5, 7, 5], false);
        assert_eq!(gf2.eliminate(), Gf2Outcome::Consistent { rank: 1 });
        assert_eq!(gf2.units(), vec![(7, false)]);
        gf2.add_xor(&[9, 9], false);
        assert_eq!(gf2.eliminate(), Gf2Outcome::Consistent { rank: 1 });
        gf2.add_xor(&[], true);
        assert_eq!(gf2.eliminate(), Gf2Outcome::Contradiction);
    }

    #[test]
    fn the_reduced_form_has_the_solutions_of_the_system() {
        let mut seed = 0x9e37_79b9_7f4a_7c15;
        let (mut refuted, mut consistent) = (0, 0);
        for _ in 0..600 {
            let variables = 1 + (next(&mut seed) % 10) as u32;
            let system: Vec<(Vec<u32>, bool)> = (0..1 + next(&mut seed) % 12).map(|_| {
                let row = (0..next(&mut seed) % 5).map(|_| 1 + (next(&mut seed) % u64::from(variables)) as u32).collect();
                (row, next(&mut seed) & 1 == 1)
            }).collect();
            let mut gf2 = Gf2Propagator::new(true);
            for (row, constant) in &system {
                gf2.add_xor(row, *constant);
            }
            let expected = solutions(variables, &system);
            match gf2.eliminate() {
                Gf2Outcome::Contradiction => {
                    assert!(expected.is_empty(), "{system:?}");
                    refuted += 1;
                }
                Gf2Outcome::Consistent { rank } => {
                    let reduced: Vec<(Vec<u32>, bool)> = (0..rank).map(|index| gf2.row(index)).collect();
                    assert_eq!(solutions(variables, &reduced), expected, "{system:?}");
                    // Cada ecuacion independiente divide entre dos el espacio de soluciones.
                    assert_eq!(expected.len(), 1 << (variables as usize - rank));
                    // Forma reducida: el pivote de una fila no aparece en ninguna otra.
                    for (index, (row, _)) in reduced.iter().enumerate() {
                        assert!(reduced.iter().enumerate().all(|(other, (variables, _))| other == index || !variables.contains(&row[0])));
                    }
                    for (variable, value) in gf2.units() {
                        assert!(expected.iter().all(|bits| (bits >> (variable - 1) & 1 == 1) == value));
                    }
                    consistent += 1;
                }
                Gf2Outcome::Disabled => unreachable!("El modulo esta encendido"),
            }
        }
        assert!(refuted > 50 && consistent > 50, "{refuted} refutados, {consistent} consistentes");
    }

    #[test]
    fn rows_wider_than_a_word_reduce_across_word_boundaries() {
        // x_1 = 1 y x_i ^ x_(i+1) = 1 para 300 variables: la cadena fija todas, alternando.
        let mut gf2 = Gf2Propagator::new(true);
        for variable in (1..300_u32).rev() {
            gf2.add_xor(&[variable, variable + 1], true);
        }
        gf2.add_xor(&[1], true);
        assert_eq!(gf2.eliminate(), Gf2Outcome::Consistent { rank: 300 });
        let mut units = gf2.units();
        units.sort_unstable();
        assert_eq!(units.len(), 300);
        assert!(units.iter().all(|&(variable, value)| value == (variable % 2 == 1)));
        // Una ecuacion sobre las 300 variables que contradice la cadena: 150 valen 1.
        let all: Vec<u32> = (1..=300).collect();
        gf2.add_xor(&all, true);
        assert_eq!(gf2.eliminate(), Gf2Outcome::Contradiction);
    }
}
