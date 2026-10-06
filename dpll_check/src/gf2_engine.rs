// Mejora 1: nucleo matricial sobre GF(2). Un sistema de ecuaciones XOR
//
//     x_a ^ x_b ^ ... = b
//
// se guarda como matriz aumentada densa de bits, una fila por ecuacion, y se reduce por
// Gauss-Jordan sumando filas con XOR palabra a palabra. La forma reducida da el rango, las
// variables que quedan forzadas y las filas 0 = 1, que demuestran que el sistema no tiene
// solucion.
//
// Dentro de la busqueda la matriz no se toca: `bind` traduce una vez la forma reducida a
// los indices de variable del motor y `propagate_trail` atiende solo a las filas que
// vigilan una variable recien asignada (dos vigiladas por fila). Sigue siendo propagacion
// fila a fila sobre una matriz fija, no eliminacion incremental: lo que solo se deduce
// combinando filas bajo una asignacion parcial no se ve.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gf2PropagationResult {
    // Nada de lo asignado desde la ultima llamada fuerza ni contradice ninguna fila.
    Idle,
    // A la fila `row` le queda una sola variable libre, que debe valer `value`.
    Unit { row: usize, variable: u32, value: bool },
    // La fila `row` esta asignada por completo y su XOR no da el termino independiente.
    Conflict { row: usize },
}

#[derive(Clone)]
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
    // La forma reducida sobre los indices de variable del motor: las variables de todas las
    // filas, seguidas; donde empieza cada fila; y su termino independiente.
    bound: Vec<u32>,
    bound_starts: Vec<u32>,
    bound_constants: Vec<bool>,
    // Vigilancia: las filas que vigila cada variable del motor y las dos variables que
    // vigila cada fila.
    watches: Vec<Vec<u32>>,
    watched_by_row: Vec<[u32; 2]>,
    // Filas de una sola variable: unidades de la raiz, de las que `settled` ya estan fijadas.
    singles: Vec<u32>,
    settled: usize,
    // Cuanto del trail se ha procesado, y donde empezaba cada nivel de decision la ultima
    // vez que se miro: `starts[n]` es el principio del nivel n + 1.
    cursor: usize,
    starts: Vec<usize>,
}

impl Gf2Propagator {
    pub fn new(enabled: bool) -> Self {
        // Ninguno de estos contenedores reserva memoria hasta recibir su primer elemento.
        Self {
            enabled, variables: Vec::new(), column_of: HashMap::new(), terms: Vec::new(),
            ends: Vec::new(), constants: Vec::new(), words: 0, matrix: Vec::new(),
            pivots: Vec::new(), inconsistent: false,
            bound: Vec::new(), bound_starts: Vec::new(), bound_constants: Vec::new(),
            watches: Vec::new(), watched_by_row: Vec::new(), singles: Vec::new(), settled: 0,
            cursor: 0, starts: Vec::new(),
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
    // La ultima `eliminate` encontro una fila 0 = 1.
    pub fn is_contradictory(&self) -> bool {
        self.inconsistent
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
    // Enlaza la forma reducida con el motor, que tiene `variables` variables; `index_of` da
    // el indice interno de cada una. Cada fila pasa a vigilar sus dos primeras variables. Se
    // llama una vez, despues de `eliminate`; la busqueda ya no toca la matriz.
    pub fn bind(&mut self, variables: usize, index_of: impl Fn(u32) -> u32) {
        self.bound.clear();
        self.bound_starts.clear();
        self.bound_constants.clear();
        self.watches.clear();
        self.watched_by_row.clear();
        self.singles.clear();
        self.starts.clear();
        (self.settled, self.cursor) = (0, 0);
        if !self.enabled || self.inconsistent || self.rank() == 0 {
            return;
        }
        self.watches.resize(variables, Vec::new());
        self.bound_starts.push(0);
        for index in 0..self.rank() {
            let (row_variables, constant) = self.row(index);
            let row = self.bound_constants.len() as u32;
            let start = self.bound.len();
            self.bound.extend(row_variables.into_iter().map(&index_of));
            self.bound_starts.push(self.bound.len() as u32);
            self.bound_constants.push(constant);
            // Una fila reducida tiene al menos su pivote; con una sola variable es una unidad.
            let first = self.bound[start];
            match self.bound.get(start + 1).copied() {
                Some(second) => {
                    self.watches[first as usize].push(row);
                    self.watches[second as usize].push(row);
                    self.watched_by_row.push([first, second]);
                }
                None => {
                    self.singles.push(row);
                    self.watched_by_row.push([first, first]);
                }
            }
        }
    }
    pub fn bound_rows(&self) -> usize {
        self.bound_constants.len()
    }
    // Variables, en indices del motor, de la fila enlazada `row`.
    pub fn bound_row(&self, row: usize) -> &[u32] {
        &self.bound[self.bound_starts[row] as usize..self.bound_starts[row + 1] as usize]
    }
    // Procesa lo que el motor ha asignado desde la ultima llamada y devuelve la primera fila
    // que fuerza o contradice; `Idle` cuando el trail esta visto entero. El motor la llama
    // en cada punto fijo de sus clausulas, antes de decidir, y repite hasta `Idle`.
    //   values        indexado por literal: el de la variable v es `values[2 * v]`, con 1
    //                 verdadero, -1 falso y 0 libre
    //   trail         literales asignados, en orden
    //   level         nivel de decision actual; `level_start` es donde empieza en el trail
    // Cada fila vigila dos variables. Mientras le queden dos libres, sus vigiladas lo estan
    // y nadie la mira. Al asignarse una vigilada se busca otra libre que la sustituya, y
    // solo si no la hay la fila puede forzar o contradecir. Al retroceder no se deshace
    // nada: las vigiladas de una fila son sus ultimas asignadas, las primeras en liberarse.
    pub fn propagate_trail(
        &mut self, values: &[i8], trail: &[u32], level: usize, level_start: usize,
    ) -> Gf2PropagationResult {
        if self.bound_constants.is_empty() {
            return Gf2PropagationResult::Idle;
        }
        // Unidades de la formula. Fijadas en la raiz no se vuelven a mirar; si alguna se
        // asigno mas arriba, se repasa hasta que la raiz la fije.
        let mut single = self.settled;
        while single < self.singles.len() {
            let row = self.singles[single] as usize;
            let variable = self.bound[self.bound_starts[row] as usize];
            let wanted = self.bound_constants[row];
            match values[(variable << 1) as usize] {
                0 => return Gf2PropagationResult::Unit { row, variable, value: wanted },
                value if (value > 0) != wanted => return Gf2PropagationResult::Conflict { row },
                _ if level == 0 && single == self.settled => self.settled += 1,
                _ => {}
            }
            single += 1;
        }
        // Si la busqueda retrocedio, el trail se corto donde empezaba el nivel siguiente al
        // actual: todo lo anterior sigue visto y se retoma justo ahi.
        if level < self.starts.len() {
            self.cursor = self.cursor.min(self.starts[level]);
            self.starts.truncate(level);
        }
        // Un nivel nuevo empieza donde dice el motor; si se abrieron varios sin mirar, de
        // los intermedios solo se sabe que empezaron despues de lo ya visto.
        while self.starts.len() < level {
            let known = if self.starts.len() + 1 == level { level_start } else { self.cursor.min(level_start) };
            self.starts.push(known);
        }
        self.cursor = self.cursor.min(trail.len());
        while self.cursor < trail.len() {
            let variable = trail[self.cursor] >> 1;
            let mut index = 0;
            while index < self.watches[variable as usize].len() {
                let row = self.watches[variable as usize][index] as usize;
                let [first, second] = self.watched_by_row[row];
                let other = if first == variable { second } else { first };
                // Un solo recorrido: la paridad de lo asignado y, si aparece, un sustituto.
                let mut parity = self.bound_constants[row];
                let mut replacement = None;
                for &candidate in &self.bound[self.bound_starts[row] as usize..self.bound_starts[row + 1] as usize] {
                    match values[(candidate << 1) as usize] {
                        0 if candidate != other => {
                            replacement = Some(candidate);
                            break;
                        }
                        0 => {}
                        value => parity ^= value > 0,
                    }
                }
                if let Some(candidate) = replacement {
                    // La fila deja de vigilar `variable`; otra ocupa su hueco en la lista.
                    self.watches[variable as usize].swap_remove(index);
                    self.watches[candidate as usize].push(row as u32);
                    self.watched_by_row[row] = [other, candidate];
                    continue;
                }
                // Todo asignado salvo, quiza, la otra vigilada: `parity` es lo que falta.
                if values[(other << 1) as usize] == 0 {
                    return Gf2PropagationResult::Unit { row, variable: other, value: parity };
                }
                if parity {
                    return Gf2PropagationResult::Conflict { row };
                }
                index += 1;
            }
            self.cursor += 1;
        }
        Gf2PropagationResult::Idle
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
        gf2.bind(1000, |id| id);
        assert_eq!(gf2.propagate_trail(&[], &[], 0, 0), Gf2PropagationResult::Idle);
        let reserved = gf2.variables.capacity() + gf2.column_of.capacity() + gf2.terms.capacity()
            + gf2.ends.capacity() + gf2.constants.capacity() + gf2.matrix.capacity() + gf2.pivots.capacity()
            + gf2.bound.capacity() + gf2.bound_starts.capacity() + gf2.bound_constants.capacity()
            + gf2.watches.capacity() + gf2.watched_by_row.capacity() + gf2.singles.capacity()
            + gf2.starts.capacity();
        assert_eq!(reserved, 0);
    }

    // Motor minimo: valores por literal, trail y niveles, con el protocolo del CDCL.
    struct Mini {
        values: Vec<i8>,
        trail: Vec<u32>,
        limits: Vec<usize>,
    }
    impl Mini {
        fn new(variables: usize) -> Self {
            Self { values: vec![0; 2 * variables], trail: Vec::new(), limits: Vec::new() }
        }
        fn value(&self, variable: u32) -> i8 {
            self.values[(variable << 1) as usize]
        }
        fn assign(&mut self, variable: u32, value: bool) {
            assert_eq!(self.value(variable), 0);
            self.values[(variable << 1) as usize] = if value { 1 } else { -1 };
            self.values[(variable << 1 | 1) as usize] = if value { -1 } else { 1 };
            self.trail.push((variable << 1) | u32::from(!value));
        }
        fn decide(&mut self, variable: u32, value: bool) {
            self.limits.push(self.trail.len());
            self.assign(variable, value);
        }
        fn backjump(&mut self, level: usize) {
            for literal in self.trail.drain(self.limits[level]..) {
                self.values[literal as usize] = 0;
                self.values[(literal ^ 1) as usize] = 0;
            }
            self.limits.truncate(level);
        }
        // Lleva las ecuaciones a su punto fijo: literales forzados, o la fila en conflicto.
        fn settle(&mut self, gf2: &mut Gf2Propagator) -> Result<usize, usize> {
            let mut forced = 0;
            loop {
                let start = self.limits.last().copied().unwrap_or(0);
                match gf2.propagate_trail(&self.values, &self.trail, self.limits.len(), start) {
                    Gf2PropagationResult::Idle => return Ok(forced),
                    Gf2PropagationResult::Unit { variable, value, .. } => {
                        self.assign(variable, value);
                        forced += 1;
                    }
                    Gf2PropagationResult::Conflict { row } => return Err(row),
                }
            }
        }
    }

    // Referencia sin vigilancia: filas con una sola variable libre y filas contradichas.
    fn scan(gf2: &Gf2Propagator, values: &[i8]) -> (Vec<usize>, Vec<usize>) {
        let (mut units, mut conflicts) = (Vec::new(), Vec::new());
        for row in 0..gf2.bound_rows() {
            let variables = gf2.bound_row(row);
            let open = variables.iter().filter(|&&variable| values[(variable << 1) as usize] == 0).count();
            let parity = variables.iter().fold(gf2.bound_constants[row], |parity, &variable| {
                parity ^ (values[(variable << 1) as usize] > 0)
            });
            match open {
                0 if parity => conflicts.push(row),
                1 => units.push(row),
                _ => {}
            }
        }
        (units, conflicts)
    }

    #[test]
    fn watched_rows_force_and_contradict_as_the_trail_grows_and_shrinks() {
        // Dos filas sobre los indices 0..5: x0 ^ x1 ^ x2 = 1 y x3 ^ x4 = 0.
        let mut gf2 = Gf2Propagator::new(true);
        gf2.add_xor(&[10, 11, 12], true);
        gf2.add_xor(&[13, 14], false);
        assert_eq!(gf2.eliminate(), Gf2Outcome::Consistent { rank: 2 });
        gf2.bind(5, |id| id - 10);
        assert_eq!((gf2.bound_row(0), gf2.bound_row(1)), (&[0, 1, 2][..], &[3, 4][..]));
        assert_eq!(gf2.watched_by_row, vec![[0, 1], [3, 4]]);
        let mut engine = Mini::new(5);
        assert_eq!(engine.settle(&mut gf2), Ok(0));
        // x0 era vigilada: al asignarse, la fila pasa a vigilar x2 y no dice nada.
        engine.decide(0, true);
        assert_eq!(engine.settle(&mut gf2), Ok(0));
        assert_eq!(gf2.watched_by_row[0], [1, 2]);
        assert!(gf2.watches[0].is_empty());
        // x2 = 1 no tiene sustituto: fuerza x1 = 1 ^ 1 ^ 1 = 1.
        engine.decide(2, true);
        assert_eq!(engine.settle(&mut gf2), Ok(1));
        assert_eq!(engine.value(1), 1);
        // Retroceder no toca la vigilancia y la fila vuelve a responder.
        engine.backjump(1);
        assert_eq!(engine.settle(&mut gf2), Ok(0));
        engine.decide(1, false);
        assert_eq!(engine.settle(&mut gf2), Ok(1));
        assert_eq!(engine.value(2), -1);
        engine.decide(3, true);
        assert_eq!(engine.settle(&mut gf2), Ok(1));
        assert_eq!(engine.value(4), 1);
        // Las clausulas asignan toda una fila dentro de un mismo punto fijo: conflicto.
        engine.backjump(0);
        engine.decide(0, true);
        engine.assign(1, true);
        engine.assign(2, false);
        assert_eq!(engine.settle(&mut gf2), Err(0));
        // Nada de esto cambia la matriz.
        assert_eq!(gf2.row(0), (vec![10, 11, 12], true));
    }

    #[test]
    fn single_variable_rows_are_root_units() {
        // x5 = 1 queda como fila de una variable; x6 ^ x7 = 1 se vigila con normalidad.
        let system = |gf2: &mut Gf2Propagator| {
            gf2.add_xor(&[5], true);
            gf2.add_xor(&[6, 7], true);
            assert_eq!(gf2.eliminate(), Gf2Outcome::Consistent { rank: 2 });
            gf2.bind(3, |id| id - 5);
        };
        let mut gf2 = Gf2Propagator::new(true);
        system(&mut gf2);
        let mut engine = Mini::new(3);
        assert_eq!(engine.settle(&mut gf2), Ok(1));
        assert_eq!(engine.value(0), 1);
        engine.decide(1, false);
        assert_eq!(engine.settle(&mut gf2), Ok(1));
        assert_eq!(engine.value(2), 1);
        // Si la raiz ya dice lo contrario, es un conflicto.
        let mut gf2 = Gf2Propagator::new(true);
        system(&mut gf2);
        let mut engine = Mini::new(3);
        engine.assign(0, false);
        assert_eq!(engine.settle(&mut gf2), Err(0));
    }

    #[test]
    fn watching_never_misses_what_a_full_scan_would_find() {
        // Paseos aleatorios con decisiones, lotes de asignaciones como los de las clausulas
        // y retrocesos: en cada punto fijo, un recorrido completo no debe encontrar nada.
        let mut seed = 0xc2b2_ae3d_27d4_eb4f;
        let (mut forced, mut conflicts, mut points) = (0, 0, 0);
        for _ in 0..300 {
            let variables = 4 + (next(&mut seed) % 10) as u32;
            let mut gf2 = Gf2Propagator::new(true);
            for _ in 0..1 + next(&mut seed) % u64::from(variables) {
                let row: Vec<u32> = (0..1 + next(&mut seed) % 5).map(|_| 1 + (next(&mut seed) % u64::from(variables)) as u32).collect();
                gf2.add_xor(&row, next(&mut seed) & 1 == 1);
            }
            if gf2.eliminate() == Gf2Outcome::Contradiction {
                continue;
            }
            gf2.bind(variables as usize, |id| id - 1);
            let mut engine = Mini::new(variables as usize);
            for _ in 0..120 {
                match engine.settle(&mut gf2) {
                    Ok(count) => {
                        forced += count;
                        points += 1;
                        assert_eq!(scan(&gf2, &engine.values), (Vec::new(), Vec::new()), "trail {:?}", engine.trail);
                    }
                    Err(row) => {
                        conflicts += 1;
                        assert!(scan(&gf2, &engine.values).1.contains(&row));
                        // Un conflicto en la raiz no tiene arreglo: se acaba el paseo.
                        if engine.limits.is_empty() {
                            break;
                        }
                        let level = (next(&mut seed) % engine.limits.len() as u64) as usize;
                        engine.backjump(level);
                        continue;
                    }
                }
                let free: Vec<u32> = (0..variables).filter(|&variable| engine.value(variable) == 0).collect();
                if free.is_empty() || next(&mut seed) % 4 == 0 {
                    if engine.limits.is_empty() {
                        if free.is_empty() {
                            break;
                        }
                    } else {
                        let level = (next(&mut seed) % engine.limits.len() as u64) as usize;
                        engine.backjump(level);
                        continue;
                    }
                }
                // Una decision y, a veces, mas asignaciones del mismo nivel antes de mirar.
                fn pick(engine: &Mini, variables: u32, seed: &mut u64) -> Option<u32> {
                    let free: Vec<u32> = (0..variables).filter(|&variable| engine.value(variable) == 0).collect();
                    free.get((next(seed) % free.len().max(1) as u64) as usize).copied()
                }
                let Some(variable) = pick(&engine, variables, &mut seed) else { continue };
                engine.decide(variable, next(&mut seed) & 1 == 1);
                for _ in 0..next(&mut seed) % 3 {
                    if let Some(variable) = pick(&engine, variables, &mut seed) {
                        engine.assign(variable, next(&mut seed) & 1 == 1);
                    }
                }
                // Las clausulas tambien encuentran conflictos antes de que se mire aqui.
                if next(&mut seed) % 6 == 0 {
                    let level = (next(&mut seed) % engine.limits.len() as u64) as usize;
                    engine.backjump(level);
                }
            }
        }
        assert!(forced > 500 && conflicts > 100 && points > 3000, "{forced} forced, {conflicts} conflicts, {points} points");
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
