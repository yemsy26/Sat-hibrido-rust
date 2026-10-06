// Fase 16: trazas certificables. DRAT para busqueda y paridad; pseudo-booleana (formato 3.0 de
// VeriPB) con regla de dominancia para romper simetrias sin perder el certificado. PHP 10/9 de
// estres con la simetria activa: 0.83 ms sin traza, 0.92 ms con traza (37 KB, 10 dominancias),
// aceptada por VeriPB 3.0.2 (s VERIFIED UNSATISFIABLE). El CDCL no reconoce ninguna familia.
// El oraculo solo escribe fases; SAT verifica la formula original. Compartir no garantiza mejoras.
#![forbid(unsafe_code)]
#![deny(warnings)]
pub mod proof {
// Trazas de prueba. El motor solo escribe; la comprobación corresponde a un verificador
// externo.
//
// DRAT: cada línea añade una cláusula que es RUP respecto de las anteriores, o RAT sobre
// su primer literal, o borra una cláusula. No tiene regla para una ruptura de simetría.
//
// Pseudo-booleano (formato 3.0 de VeriPB): además de RUP y de la regla de redundancia, la
// regla de dominancia justifica z ≤lex σ(z) para un automorfismo σ respecto de un orden
// cargado en la prueba, y de esa desigualdad salen por RUP los predicados de ruptura.
use std::collections::HashMap;
use std::io::{self, Write};

pub trait ProofLog {
    // Cláusula que se sigue por propagación unitaria de lo ya presente.
    fn add(&mut self, clause: &[i32]);
    // Cláusula de una definición: su primer literal es el de la variable nueva.
    fn define(&mut self, clause: &[i32]) {
        self.add(clause);
    }
    fn delete(&mut self, clause: &[i32]);
    fn supports_dominance(&self) -> bool {
        false
    }
    // Orden lexicográfico sobre estas variables, de la más a la menos significativa.
    fn order(&mut self, _variables: &[i32]) {}
    // z ≤lex σ(z) sobre las variables ordenadas, para el automorfismo dado por todos sus
    // pares (variable, imagen).
    fn dominance(&mut self, _pairs: &[(i32, i32)]) {}
    // La fórmula ya trae la cláusula vacía: DRAT no necesita ningún paso.
    fn input_refuted(&mut self) {}
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProofFormat {
    Text,
    Binary,
}

// Una llamada al sistema por cada 64 KiB de traza, no por paso.
const FLUSH_AT: usize = 1 << 16;

struct Buffered<W: Write> {
    sink: W,
    buffer: Vec<u8>,
    bytes: u64,
    error: Option<io::Error>,
}
impl<W: Write> Buffered<W> {
    fn new(sink: W) -> Self {
        Self { sink, buffer: Vec::with_capacity(FLUSH_AT + 4096), bytes: 0, error: None }
    }
    fn text(&mut self, text: &str) {
        self.buffer.extend_from_slice(text.as_bytes());
    }
    fn named(&mut self, prefix: &str, index: u32) {
        self.text(prefix);
        self.number(index);
    }
    fn number(&mut self, mut value: u32) {
        let mut digits = [0_u8; 10];
        let mut start = digits.len();
        loop {
            start -= 1;
            digits[start] = b'0' + (value % 10) as u8;
            value /= 10;
            if value == 0 {
                break;
            }
        }
        self.buffer.extend_from_slice(&digits[start..]);
    }
    fn commit(&mut self) {
        if self.buffer.len() >= FLUSH_AT {
            self.drain();
        }
    }
    fn drain(&mut self) {
        if self.error.is_none() {
            if let Err(error) = self.sink.write_all(&self.buffer) {
                self.error = Some(error);
            }
        }
        self.bytes += self.buffer.len() as u64;
        self.buffer.clear();
    }
    fn written(&self) -> u64 {
        self.bytes + self.buffer.len() as u64
    }
    // Entrega lo pendiente y devuelve el destino; un error de escritura invalida la traza.
    fn finish(mut self) -> io::Result<W> {
        self.drain();
        if let Some(error) = self.error {
            return Err(error);
        }
        self.sink.flush()?;
        Ok(self.sink)
    }
}

pub struct Proof<W: Write> {
    out: Buffered<W>,
    format: ProofFormat,
    additions: u64,
    deletions: u64,
}
impl<W: Write> Proof<W> {
    pub fn new(sink: W, format: ProofFormat) -> Self {
        Self { out: Buffered::new(sink), format, additions: 0, deletions: 0 }
    }
    pub fn additions(&self) -> u64 {
        self.additions
    }
    pub fn deletions(&self) -> u64 {
        self.deletions
    }
    // Bytes ya entregados al destino más los que esperan en el búfer.
    pub fn bytes(&self) -> u64 {
        self.out.written()
    }
    fn line(&mut self, clause: &[i32], deletion: bool) {
        match self.format {
            ProofFormat::Text => {
                if deletion {
                    self.out.text("d ");
                }
                for &literal in clause {
                    if literal < 0 {
                        self.out.buffer.push(b'-');
                    }
                    self.out.number(literal.unsigned_abs());
                    self.out.buffer.push(b' ');
                }
                self.out.text("0\n");
            }
            ProofFormat::Binary => {
                // DRAT binario: 'a' o 'd', literales 2|x| + signo en base 128, y un cero.
                self.out.buffer.push(if deletion { b'd' } else { b'a' });
                for &literal in clause {
                    let mut code = 2 * u64::from(literal.unsigned_abs()) + u64::from(literal < 0);
                    while code >= 128 {
                        self.out.buffer.push((code & 127) as u8 | 128);
                        code >>= 7;
                    }
                    self.out.buffer.push(code as u8);
                }
                self.out.buffer.push(0);
            }
        }
        self.out.commit();
    }
    pub fn finish(self) -> io::Result<W> {
        self.out.finish()
    }
}
impl<W: Write> ProofLog for Proof<W> {
    fn add(&mut self, clause: &[i32]) {
        self.additions += 1;
        self.line(clause, false);
    }
    fn delete(&mut self, clause: &[i32]) {
        self.deletions += 1;
        self.line(clause, true);
    }
}

// Potencias de dos en decimal: los pesos del orden lexicográfico no caben en una palabra.
fn powers_of_two(count: usize) -> Vec<String> {
    let mut digits = vec![1_u8];
    let mut powers = Vec::with_capacity(count);
    for _ in 0..count {
        powers.push(digits.iter().rev().map(|&digit| (b'0' + digit) as char).collect());
        let mut carry = 0;
        for digit in &mut digits {
            let doubled = *digit * 2 + carry;
            *digit = doubled % 10;
            carry = doubled / 10;
        }
        if carry > 0 {
            digits.push(carry);
        }
    }
    powers
}

// Resta de dos decimales, el primero mayor que el segundo.
fn difference(larger: &str, smaller: &str) -> String {
    let mut digits: Vec<u8> = Vec::with_capacity(larger.len());
    let mut low = smaller.bytes().rev();
    let mut borrow = 0;
    for high in larger.bytes().rev() {
        let take = low.next().map_or(0, |digit| digit - b'0') + borrow;
        let have = high - b'0';
        borrow = u8::from(have < take);
        digits.push(have + 10 * borrow - take);
    }
    while digits.len() > 1 && digits.last() == Some(&0) {
        digits.pop();
    }
    digits.iter().rev().map(|&digit| (b'0' + digit) as char).collect()
}

// Traza pseudo-booleana en el formato 3.0 de VeriPB: cada regla termina en punto y coma.
// `clauses` es el número de cláusulas del fichero CNF que acompaña a la prueba; las
// variables se nombran x<id>, como hace el verificador al leer DIMACS. Las cláusulas de la
// fórmula forman el núcleo; todo lo derivado aquí queda en el conjunto derivado.
pub struct PbProof<W: Write> {
    out: Buffered<W>,
    // Posición de cada variable en el orden cargado y potencias de dos en decimal.
    positions: HashMap<i32, usize>,
    powers: Vec<String>,
    steps: u64,
    dominances: u64,
    deletions: u64,
    refuted: bool,
}
impl<W: Write> PbProof<W> {
    pub fn new(sink: W, clauses: usize) -> Self {
        let mut out = Buffered::new(sink);
        out.text("pseudo-Boolean proof version 3.0\nf ");
        out.number(clauses as u32);
        out.text(";\n");
        Self {
            out, positions: HashMap::new(), powers: Vec::new(),
            steps: 0, dominances: 0, deletions: 0, refuted: false,
        }
    }
    // Restricciones derivadas: RUP, redundancia y dominancia.
    pub fn steps(&self) -> u64 {
        self.steps
    }
    pub fn dominances(&self) -> u64 {
        self.dominances
    }
    pub fn deletions(&self) -> u64 {
        self.deletions
    }
    pub fn bytes(&self) -> u64 {
        self.out.written()
    }
    // Una cláusula es la restricción «suma de sus literales >= 1».
    fn clause(&mut self, clause: &[i32]) {
        for &literal in clause {
            self.out.named(if literal < 0 { "1 ~x" } else { "1 x" }, literal.unsigned_abs());
            self.out.text(" ");
        }
        self.out.text(">= 1");
    }
    fn names(&mut self, keyword: &str, prefix: &str, count: usize) {
        self.out.text(keyword);
        for index in 1..=count {
            self.out.text(" ");
            self.out.named(prefix, index as u32);
        }
        self.out.text(";\n");
    }
    // Cierra la traza: la conclusión es UNSAT solo si lo último derivado fue 0 >= 1.
    pub fn finish(mut self) -> io::Result<W> {
        self.out.text("output NONE;\n");
        self.out.text(if self.refuted { "conclusion UNSAT : -1;\n" } else { "conclusion NONE;\n" });
        self.out.text("end pseudo-Boolean proof;\n");
        self.out.finish()
    }
}
impl<W: Write> ProofLog for PbProof<W> {
    fn add(&mut self, clause: &[i32]) {
        self.steps += 1;
        self.refuted = clause.is_empty();
        self.out.text("rup ");
        self.clause(clause);
        self.out.text(";\n");
        self.out.commit();
    }
    // Redundancia con testigo: la variable nueva toma el valor que satisface la cláusula.
    // No pertenece al orden cargado, así que las metas del orden son triviales.
    fn define(&mut self, clause: &[i32]) {
        self.steps += 1;
        self.refuted = false;
        self.out.text("red ");
        self.clause(clause);
        self.out.named(" : x", clause[0].unsigned_abs());
        self.out.text(if clause[0] > 0 { " -> 1;\n" } else { " -> 0;\n" });
        self.out.commit();
    }
    // Borrado por especificación: solo se borran cláusulas aprendidas, que son derivadas.
    fn delete(&mut self, clause: &[i32]) {
        self.deletions += 1;
        self.out.text("del spec ");
        self.clause(clause);
        self.out.text(";\n");
        self.out.commit();
    }
    fn supports_dominance(&self) -> bool {
        true
    }
    // La conclusión señala una restricción contradictoria: se deriva de la que ya lo es.
    fn input_refuted(&mut self) {
        self.add(&[]);
    }
    // u ≼ v  :=  suma de 2^(n-1-i) (v_i - u_i) >= 0. Transitividad: las premisas 1 y 2 son
    // u ≼ v y v ≼ w, la 3 es la negación de u ≼ w, y las tres suman 0 >= 1. La
    // reflexividad es trivial: la negación de u ≼ u ya es 0 >= 1.
    fn order(&mut self, variables: &[i32]) {
        let count = variables.len();
        self.powers = powers_of_two(count);
        self.positions = variables.iter().enumerate().map(|(position, &id)| (id, position)).collect();
        self.out.text("def_order lex\nvars\n");
        self.names("left", "u", count);
        self.names("right", "v", count);
        self.out.text("aux;\nend vars;\ndef\n");
        for position in 0..count {
            let weight = &self.powers[count - 1 - position];
            self.out.text("-");
            self.out.text(weight);
            self.out.named(" u", position as u32 + 1);
            self.out.text(" +");
            self.out.text(weight);
            self.out.named(" v", position as u32 + 1);
            self.out.text(" ");
        }
        self.out.text(">= 0;\nend def;\ntransitivity\nvars\n");
        self.names("fresh_right", "w", count);
        self.out.text("end vars;\nproof\nproofgoal #1\npol 1 2 + 3 +;\nqed #1 : -1;\nqed proof;\n");
        self.out.text("end transitivity;\nend def_order;\nload_order lex");
        for &id in variables {
            self.out.named(" x", id.unsigned_abs());
        }
        self.out.text(";\n");
        self.out.commit();
    }
    // C := suma de peso(z) (σ(z) - z) >= 0 sobre las variables ordenadas que σ mueve, con
    // testigo σ y un solo término por variable: lo que gana como imagen menos lo que pierde
    // en su posición. Las cláusulas del núcleo van a cláusulas del núcleo porque σ es un
    // automorfismo, y σ(z) ≼ z se sigue de la negación de C: ambas metas son automáticas.
    // La meta estricta #2 se da hecha: su premisa z ≼ σ(z) es C, y con ¬C suma 0 >= 1.
    fn dominance(&mut self, pairs: &[(i32, i32)]) {
        self.steps += 1;
        self.dominances += 1;
        self.refuted = false;
        let count = self.powers.len();
        // Exponente con que cada variable aparece como imagen: el de su preimagen.
        let gained: HashMap<i32, usize> = pairs.iter().filter_map(|&(variable, image)| {
            self.positions.get(&variable).map(|&position| (image, count - 1 - position))
        }).collect();
        self.out.text("dom ");
        for &(variable, _) in pairs {
            let lost = self.positions.get(&variable).map(|&position| count - 1 - position);
            match (gained.get(&variable).copied(), lost) {
                (Some(plus), Some(minus)) if plus > minus => {
                    self.out.text("+");
                    self.out.text(&difference(&self.powers[plus], &self.powers[minus]));
                }
                (Some(plus), Some(minus)) if plus < minus => {
                    self.out.text("-");
                    self.out.text(&difference(&self.powers[minus], &self.powers[plus]));
                }
                (Some(plus), None) => {
                    self.out.text("+");
                    self.out.text(&self.powers[plus]);
                }
                (None, Some(minus)) => {
                    self.out.text("-");
                    self.out.text(&self.powers[minus]);
                }
                _ => continue,
            }
            self.out.named(" x", variable.unsigned_abs());
            self.out.text(" ");
        }
        self.out.text(">= 0 :");
        for &(variable, image) in pairs {
            self.out.named(" x", variable.unsigned_abs());
            self.out.named(" -> x", image.unsigned_abs());
        }
        self.out.text(" : subproof\nproofgoal #2\npol -1 -2 +;\nqed #2 : -1;\nqed dom;\n");
        self.out.commit();
    }
}

}

pub mod parity {
// Inprocessing: extracción de paridad y eliminación de Gauss-Jordan sobre GF(2).
//
// Corrección. Una ecuación x1 ⊕ ... ⊕ xk = b se extrae solo cuando la fórmula contiene sus
// 2^(k-1) cláusulas; la conjunción de esas cláusulas equivale a la ecuación, luego F la
// implica. Toda combinación lineal de ecuaciones implicadas está implicada, así que cada
// fila de la forma reducida es consecuencia de F y añadirla conserva exactamente sus
// modelos. Una fila 0 = 1 demuestra que F es insatisfacible. Las filas largas se expresan
// con variables auxiliares definidas (t = a ⊕ b): es una extensión conservadora, con los
// mismos modelos sobre las variables originales.
//
// Certificación. Con una traza DRAT, cada fila se deriva sumando una a una las ecuaciones
// originales que la componen. Toda restricción vive en forma de cadena: un literal por
// prefijo, definido por RAT, y una unidad sobre el último. La suma de dos cadenas avanza
// variable a variable con lemas RUP de tres literales, así que la traza crece linealmente
// con el soporte y no exponencialmente. Lo que no cabe en el presupuesto no se inyecta.
use super::proof::ProofLog;

pub type Clause = Vec<i32>;

#[derive(Clone, Debug, Default)]
pub struct ParityReduction {
    // Consecuencias lineales en CNF; se añaden a la fórmula.
    pub clauses: Vec<Clause>,
    pub equations: usize,
    pub rank: usize,
    pub units: usize,
    pub equivalences: usize,
    // Variables nuevas, con identificadores por encima de todos los originales.
    pub auxiliaries: usize,
    pub refuted: bool,
}

// La reducción es opcional: cualquier tope solo la debilita, nunca compromete la corrección.
const MAX_ARITY: usize = 8;
const DIRECT_ARITY: usize = 4;
const ELIMINATION_WORK: u64 = 1 << 31;
const INJECTION_FACTOR: usize = 4;
const INJECTION_FLOOR: usize = 1024;
const PROOF_BUDGET: usize = 1 << 21;

// Ecuación sobre índices de variable en orden creciente: su XOR vale `1`.
type Equation = (Vec<u32>, bool);

// Fila reducida, las ecuaciones originales cuya suma la produce y el ancho de su componente.
struct Row {
    equation: Equation,
    sources: Vec<u32>,
    width: usize,
}

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

// Una cláusula con n negaciones prohíbe la asignación con n unos, de paridad n. La ecuación
// de término b prohíbe las asignaciones de paridad distinta de b: sus cláusulas son todas
// las de ese soporte cuyo número de negaciones tiene paridad 1 - b.
fn extract(ids: &[u32], formula: &[Clause]) -> Vec<Equation> {
    // Con identificadores densos basta una tabla directa; si no, búsqueda binaria.
    let dense = ids.last().is_some_and(|&last| last as usize <= 4 * ids.len() + 1024);
    let mut table = vec![0_u32; if dense { ids[ids.len() - 1] as usize + 1 } else { 0 }];
    for (index, &id) in ids.iter().enumerate().take(if dense { ids.len() } else { 0 }) {
        table[id as usize] = index as u32;
    }
    // Cláusulas candidatas en un arreglo plano, con la huella de su soporte para agruparlas.
    let mut offsets = vec![0_u32];
    let mut literals: Vec<u32> = Vec::new();
    let mut keys: Vec<(u64, u32)> = Vec::new();
    let mut packed: Vec<u32> = Vec::new();
    for clause in formula {
        if clause.len() > 2 * MAX_ARITY {
            continue;
        }
        packed.clear();
        packed.extend(clause.iter().map(|&literal| {
            let variable = if dense {
                table[literal.unsigned_abs() as usize]
            } else {
                ids.binary_search(&literal.unsigned_abs()).expect("Variable was indexed") as u32
            };
            (variable << 1) | u32::from(literal < 0)
        }));
        packed.sort_unstable();
        packed.dedup();
        if packed.len() > MAX_ARITY || packed.windows(2).any(|pair| pair[0] ^ 1 == pair[1]) {
            continue;
        }
        let support = packed.iter().fold(packed.len() as u64, |print, &literal| {
            mix(print ^ u64::from(literal >> 1))
        });
        keys.push((support, keys.len() as u32));
        literals.extend_from_slice(&packed);
        offsets.push(literals.len() as u32);
    }
    keys.sort_unstable();
    let clause = |index: u32| &literals[offsets[index as usize] as usize..offsets[index as usize + 1] as usize];
    let same_support = |first: u32, second: u32| {
        let (first, second) = (clause(first), clause(second));
        first.len() == second.len() && first.iter().zip(second).all(|(a, b)| a >> 1 == b >> 1)
    };
    let mut equations: Vec<Equation> = Vec::new();
    let mut pending: Vec<u32> = Vec::new();
    let mut group: Vec<u32> = Vec::new();
    let mut seen: Vec<bool> = Vec::new();
    let mut start = 0;
    while start < keys.len() {
        let mut end = start + 1;
        while end < keys.len() && keys[end].0 == keys[start].0 {
            end += 1;
        }
        // Misma huella no garantiza mismo soporte: se compara antes de agrupar.
        pending.clear();
        pending.extend(keys[start..end].iter().map(|&(_, index)| index));
        while let Some(&leader) = pending.first() {
            group.clear();
            let mut kept = 0;
            for position in 0..pending.len() {
                if same_support(leader, pending[position]) {
                    group.push(pending[position]);
                } else {
                    pending[kept] = pending[position];
                    kept += 1;
                }
            }
            pending.truncate(kept);
            let arity = clause(leader).len();
            let half = 1_usize << (arity - 1);
            if group.len() < half {
                continue;
            }
            // Patrones de signo presentes, separados por su número de negaciones.
            seen.clear();
            seen.resize(2 * half, false);
            let mut present = [0_usize; 2];
            for &member in &group {
                let pattern = clause(member).iter().enumerate()
                    .fold(0_usize, |pattern, (bit, &literal)| pattern | ((literal & 1) as usize) << bit);
                if !seen[pattern] {
                    seen[pattern] = true;
                    present[pattern.count_ones() as usize & 1] += 1;
                }
            }
            let variables: Vec<u32> = clause(leader).iter().map(|&literal| literal >> 1).collect();
            for negations in 0..2 {
                if present[negations] == half {
                    equations.push((variables.clone(), negations == 0));
                }
            }
        }
        start = end;
    }
    equations
}

// Matriz densa por filas de palabras de 64 bits. Tras las columnas de variables va el
// término independiente y, si se pide, una identidad que recuerda qué filas se sumaron.
struct Matrix {
    words: usize,
    columns: usize,
    data: Vec<u64>,
}
impl Matrix {
    fn bit(&self, row: usize, column: usize) -> bool {
        self.data[row * self.words + column / 64] >> (column % 64) & 1 == 1
    }
    fn set(&mut self, row: usize, column: usize) {
        self.data[row * self.words + column / 64] ^= 1 << (column % 64);
    }
    // Gauss-Jordan: cada columna pivote queda con un único uno. Devuelve el rango.
    fn eliminate(&mut self, rows: usize) -> usize {
        let words = self.words;
        let mut pivot_row = vec![0_u64; words];
        let mut rank = 0;
        for column in 0..self.columns {
            let Some(found) = (rank..rows).find(|&row| self.bit(row, column)) else { continue; };
            for word in 0..words {
                self.data.swap(found * words + word, rank * words + word);
            }
            pivot_row.copy_from_slice(&self.data[rank * words..(rank + 1) * words]);
            for row in 0..rows {
                if row != rank && self.bit(row, column) {
                    for (target, &source) in self.data[row * words..(row + 1) * words].iter_mut().zip(&pivot_row) {
                        *target ^= source;
                    }
                }
            }
            rank += 1;
        }
        rank
    }
}

// Forma reducida de un componente conexo. `Err` lleva las ecuaciones que suman 0 = 1.
fn reduce(
    component: &[Equation],
    offset: usize,
    track: bool,
    derived: &mut Vec<Row>,
    rank: &mut usize,
) -> Result<(), Vec<u32>> {
    let mut columns: Vec<u32> = component.iter().flat_map(|(variables, _)| variables.iter().copied()).collect();
    columns.sort_unstable();
    columns.dedup();
    let rows = component.len();
    let value_column = columns.len();
    let words = (value_column + 1 + if track { rows } else { 0 }) / 64 + 1;
    // Un componente demasiado grande se deja como está: solo se pierde inferencia.
    if (rows as u64).saturating_mul(rows as u64).saturating_mul(words as u64) > ELIMINATION_WORK {
        return Ok(());
    }
    let mut matrix = Matrix { words, columns: columns.len(), data: vec![0; rows * words] };
    for (row, (variables, value)) in component.iter().enumerate() {
        for variable in variables {
            matrix.set(row, columns.binary_search(variable).expect("Column was indexed"));
        }
        if *value {
            matrix.set(row, value_column);
        }
        if track {
            matrix.set(row, value_column + 1 + row);
        }
    }
    let independent = matrix.eliminate(rows);
    *rank += independent;
    let sources = |row: usize| -> Vec<u32> {
        if !track {
            return Vec::new();
        }
        (0..rows).filter(|&source| matrix.bit(row, value_column + 1 + source))
            .map(|source| (offset + source) as u32).collect()
    };
    // Tras el rango no quedan variables: un término independiente a uno es 0 = 1.
    if let Some(row) = (independent..rows).find(|&row| matrix.bit(row, value_column)) {
        return Err(sources(row));
    }
    let mut known: Vec<&Equation> = component.iter().collect();
    known.sort_unstable();
    for row in 0..independent {
        let variables: Vec<u32> = (0..columns.len()).filter(|&column| matrix.bit(row, column))
            .map(|column| columns[column]).collect();
        let equation = (variables, matrix.bit(row, value_column));
        // Una fila que ya era una ecuación de la fórmula no aporta nada nuevo.
        if known.binary_search(&&equation).is_err() {
            derived.push(Row { equation, sources: sources(row), width: columns.len() });
        }
    }
    Ok(())
}

// CNF de x1 ⊕ ... ⊕ xk = b: una cláusula por patrón de signos con paridad 1 - b.
fn encode(variables: &[i32], value: bool, clauses: &mut Vec<Clause>) {
    for pattern in 0..1_u32 << variables.len() {
        if (pattern.count_ones() & 1 == 1) != value {
            clauses.push(variables.iter().enumerate().map(|(bit, &variable)| {
                if pattern >> bit & 1 == 1 { -variable } else { variable }
            }).collect());
        }
    }
}

// Restricción en forma de cadena: `links[i]` vale la paridad de las primeras i + 1
// variables (el primero es la propia variable) y el último está fijado a `value`.
#[derive(Clone)]
struct Chain {
    variables: Vec<u32>,
    links: Vec<i32>,
    value: bool,
}

// Cláusulas de la cadena: una definición por eslabón y la unidad final.
fn chain_clauses(row: &[i32], links: &[i32], value: bool, clauses: &mut Vec<Clause>) {
    for position in 1..row.len() {
        encode(&[links[position], links[position - 1], row[position]], false, clauses);
    }
    let last = links[links.len() - 1];
    clauses.push(vec![if value { last } else { -last }]);
}

struct Prover<'a> {
    log: &'a mut dyn ProofLog,
    ids: &'a [u32],
    equations: &'a [Equation],
    chains: Vec<Option<Chain>>,
    next_id: u64,
    emitted: usize,
}
impl Prover<'_> {
    fn emit(&mut self, clause: &[i32]) {
        self.emitted += 1;
        self.log.add(clause);
    }
    fn fresh(&mut self) -> i32 {
        self.next_id += 1;
        (self.next_id - 1) as i32
    }
    // out ↔ previous ⊕ variable, con el literal nuevo delante: las cuatro cláusulas son RAT
    // sobre él (las dos primeras porque `out` aún no aparece; las otras por resolventes
    // tautológicas).
    fn define(&mut self, out: i32, previous: i32, variable: i32) {
        self.emitted += 4;
        self.log.define(&[-out, previous, variable]);
        self.log.define(&[-out, -previous, -variable]);
        self.log.define(&[out, -previous, variable]);
        self.log.define(&[out, previous, -variable]);
    }
    // Cadena de una ecuación original. La unidad final sale por casos sobre sus variables:
    // con todas asignadas la propagación recorre la cadena o falsea una cláusula original,
    // y cada nivel superior se obtiene resolviendo los dos casos del siguiente.
    fn chain_of(&mut self, index: usize) -> Chain {
        if let Some(chain) = &self.chains[index] {
            return chain.clone();
        }
        let equations = self.equations;
        let (variables, value) = &equations[index];
        let row: Vec<i32> = variables.iter().map(|&variable| self.ids[variable as usize] as i32).collect();
        let mut links = vec![row[0]];
        for &variable in &row[1..] {
            let out = self.fresh();
            self.define(out, links[links.len() - 1], variable);
            links.push(out);
        }
        if row.len() > 1 {
            let last = links[links.len() - 1];
            let target = if *value { last } else { -last };
            let mut clause = Vec::with_capacity(row.len() + 1);
            for depth in (0..=row.len()).rev() {
                for pattern in 0..1_u32 << depth {
                    clause.clear();
                    clause.push(target);
                    clause.extend((0..depth).map(|bit| if pattern >> bit & 1 == 1 { -row[bit] } else { row[bit] }));
                    self.emit(&clause);
                }
            }
        }
        let chain = Chain { variables: variables.clone(), links, value: *value };
        self.chains[index] = Some(chain.clone());
        chain
    }
    // Prohíbe cada combinación de valores de los tres eslabones con paridad impar; con
    // `split`, además bajo cada valor de esa variable. Un eslabón ausente vale falso.
    fn lemma(&mut self, links: [Option<i32>; 3], split: Option<i32>) {
        let mut clause: Vec<i32> = Vec::with_capacity(4);
        for values in 0..8_u32 {
            if values.count_ones() & 1 == 0 {
                continue;
            }
            // Un eslabón constante no puede valer verdadero: esa combinación no existe.
            if (0..3).any(|slot| links[slot].is_none() && values >> slot & 1 == 1) {
                continue;
            }
            for case in 0..if split.is_some() { 2 } else { 1 } {
                clause.clear();
                for slot in 0..3 {
                    if let Some(link) = links[slot] {
                        clause.push(if values >> slot & 1 == 1 { -link } else { link });
                    }
                }
                if let Some(variable) = split {
                    clause.push(if case == 1 { -variable } else { variable });
                }
                clause.sort_unstable_by_key(|literal| (literal.unsigned_abs(), *literal));
                clause.dedup();
                // Dos literales opuestos: la combinación es imposible y la cláusula, trivial.
                if clause.windows(2).all(|pair| pair[0] != -pair[1]) {
                    self.emit(&clause);
                }
            }
        }
    }
    // Suma de dos cadenas. En cada variable del soporte unido se mantiene el invariante
    // suma ⊕ a ⊕ b = 0 entre los eslabones vigentes: primero bajo cada valor de la variable
    // (RUP por las definiciones y el invariante anterior) y después sin ella. Devuelve la
    // cadena de la suma, o `None` si su soporte es vacío; en ese caso un término
    // independiente a uno ya ha producido la cláusula vacía.
    fn add(&mut self, a: &Chain, b: &Chain) -> Option<Chain> {
        let mut sum = Chain { variables: Vec::new(), links: Vec::new(), value: a.value ^ b.value };
        let (mut ia, mut ib) = (0, 0);
        let (mut la, mut lb, mut lc): (Option<i32>, Option<i32>, Option<i32>) = (None, None, None);
        while ia < a.variables.len() || ib < b.variables.len() {
            let variable = match (a.variables.get(ia), b.variables.get(ib)) {
                (Some(&first), Some(&second)) => first.min(second),
                (Some(&only), None) | (None, Some(&only)) => only,
                (None, None) => unreachable!("The loop condition guarantees a variable"),
            };
            let in_a = a.variables.get(ia) == Some(&variable);
            let in_b = b.variables.get(ib) == Some(&variable);
            let id = self.ids[variable as usize] as i32;
            if in_a {
                la = Some(a.links[ia]);
                ia += 1;
            }
            if in_b {
                lb = Some(b.links[ib]);
                ib += 1;
            }
            if in_a != in_b {
                let link = match lc {
                    None => id,
                    Some(previous) => {
                        let out = self.fresh();
                        self.define(out, previous, id);
                        out
                    }
                };
                lc = Some(link);
                sum.variables.push(variable);
                sum.links.push(link);
            }
            self.lemma([lc, la, lb], Some(id));
            self.lemma([lc, la, lb], None);
        }
        match lc {
            None => {
                if sum.value {
                    self.emit(&[]);
                }
                None
            }
            Some(last) => {
                self.emit(&[if sum.value { last } else { -last }]);
                Some(sum)
            }
        }
    }
    // Suma las ecuaciones indicadas, enlazando siempre con una que comparta variables para
    // que el soporte intermedio no crezca más de lo necesario.
    fn derive(&mut self, sources: &[u32]) -> Option<Chain> {
        let mut remaining: Vec<u32> = sources.to_vec();
        let mut total: Option<Chain> = None;
        while !remaining.is_empty() {
            let equations = self.equations;
            let choice = total.as_ref().and_then(|chain| {
                remaining.iter().position(|&index| {
                    equations[index as usize].0.iter().any(|variable| chain.variables.binary_search(variable).is_ok())
                })
            }).unwrap_or(0);
            let next = self.chain_of(remaining.swap_remove(choice) as usize);
            total = match total {
                None => Some(next),
                Some(chain) => self.add(&chain, &next),
            };
            // Soporte vacío a mitad de camino: el resto vuelve a empezar desde cero.
        }
        total
    }
}

// Las ecuaciones XOR que la fórmula contiene, sobre sus identificadores originales: las
// mismas que reduce esta capa, sin reducir ni inyectar nada. Es la puerta para que otro
// módulo haga su propia álgebra sobre ellas.
pub fn xor_equations(formula: &[Clause]) -> Vec<(Vec<u32>, bool)> {
    if formula.iter().any(Vec::is_empty) {
        return Vec::new();
    }
    let mut ids: Vec<u32> = formula.iter().flatten().map(|literal| literal.unsigned_abs()).collect();
    ids.sort_unstable();
    ids.dedup();
    extract(&ids, formula).into_iter().map(|(variables, constant)| {
        (variables.into_iter().map(|index| ids[index as usize]).collect(), constant)
    }).collect()
}

pub fn reduce_parity(formula: &[Clause]) -> ParityReduction {
    reduce_parity_with_proof(formula, None)
}

pub fn reduce_parity_with_proof(
    formula: &[Clause], proof: Option<&mut (dyn ProofLog + '_)>,
) -> ParityReduction {
    let mut reduction = ParityReduction::default();
    if formula.iter().any(Vec::is_empty) {
        return reduction;
    }
    let mut ids: Vec<u32> = formula.iter().flatten().map(|literal| literal.unsigned_abs()).collect();
    ids.sort_unstable();
    ids.dedup();
    let mut equations = extract(&ids, formula);
    reduction.equations = equations.len();
    if equations.is_empty() {
        return reduction;
    }
    let first_free = u64::from(*ids.last().expect("Equations have variables")) + 1;
    // Las derivaciones necesitan auxiliares: sin sitio para ellas no se certifica nada.
    if proof.is_some() && first_free > (i32::MAX / 2) as u64 {
        return reduction;
    }
    // Componentes conexos: ecuaciones que comparten variables se reducen juntas.
    let mut parent: Vec<u32> = (0..ids.len() as u32).collect();
    fn find(parent: &mut [u32], mut variable: u32) -> u32 {
        while parent[variable as usize] != variable {
            parent[variable as usize] = parent[parent[variable as usize] as usize];
            variable = parent[variable as usize];
        }
        variable
    }
    for (variables, _) in &equations {
        let root = find(&mut parent, variables[0]);
        for &variable in &variables[1..] {
            let other = find(&mut parent, variable);
            parent[other as usize] = root;
        }
    }
    equations.sort_by_cached_key(|(variables, _)| find(&mut parent, variables[0]));
    let mut derived: Vec<Row> = Vec::new();
    let mut contradiction: Option<Vec<u32>> = None;
    let mut start = 0;
    while start < equations.len() && contradiction.is_none() {
        let root = find(&mut parent, equations[start].0[0]);
        let mut end = start + 1;
        while end < equations.len() && find(&mut parent, equations[end].0[0]) == root {
            end += 1;
        }
        contradiction = reduce(&equations[start..end], start, proof.is_some(), &mut derived, &mut reduction.rank).err();
        start = end;
    }
    let mut prover = proof.map(|log| Prover {
        log, ids: &ids, equations: &equations,
        chains: vec![None; equations.len()],
        next_id: first_free, emitted: 0,
    });
    if let Some(sources) = contradiction {
        // La suma de esas ecuaciones es 0 = 1: la derivación termina en la cláusula vacía.
        if let Some(prover) = prover.as_mut() {
            prover.derive(&sources);
        }
        reduction.clauses = vec![Vec::new()];
        reduction.refuted = true;
        return reduction;
    }
    // Primero las filas cortas: son las más restrictivas y las más baratas de codificar.
    derived.sort_by_key(|row| row.equation.0.len());
    let limit = INJECTION_FACTOR * formula.len() + INJECTION_FLOOR;
    let mut next_id = first_free;
    for Row { equation: (variables, value), sources, width } in derived {
        let row: Vec<i32> = variables.iter().map(|&variable| ids[variable as usize] as i32).collect();
        let cost = if row.len() <= DIRECT_ARITY { 1 << (row.len() - 1) } else { 4 * (row.len() - 1) + 1 };
        if reduction.clauses.len() + cost > limit {
            break;
        }
        // Con traza, la fila solo entra si su derivación cabe en el presupuesto.
        let chain = match prover.as_mut() {
            None => None,
            Some(prover) => {
                if prover.emitted + 16 * sources.len() * width > PROOF_BUDGET {
                    continue;
                }
                match prover.derive(&sources) {
                    Some(chain) => Some(chain),
                    None => continue,
                }
            }
        };
        if row.len() <= DIRECT_ARITY {
            match row.len() {
                1 => reduction.units += 1,
                2 => reduction.equivalences += 1,
                _ => {}
            }
            let before = reduction.clauses.len();
            encode(&row, value, &mut reduction.clauses);
            // Con todas las variables asignadas la cadena propaga hasta su unidad: RUP.
            if let Some(prover) = prover.as_mut() {
                if row.len() > 1 {
                    for clause in &reduction.clauses[before..] {
                        prover.emit(clause);
                    }
                }
            }
            continue;
        }
        let links = match chain {
            // La cadena de la traza es la que se inyecta: sus definiciones ya están probadas.
            Some(chain) => chain.links,
            None => {
                if next_id + row.len() as u64 > i32::MAX as u64 {
                    continue;
                }
                let mut links = vec![row[0]];
                for _ in 1..row.len() {
                    links.push(next_id as i32);
                    next_id += 1;
                }
                links
            }
        };
        reduction.auxiliaries += row.len() - 1;
        chain_clauses(&row, &links, value, &mut reduction.clauses);
    }
    reduction
}

}

pub mod symmetry {
// Inprocessing: ruptura estática de simetrías por automorfismos del grafo de la fórmula.
//
// Corrección. Un automorfismo es una permutación de variables σ con σ(F) = F como conjunto
// de cláusulas; entonces α satisface F si y solo si α∘σ la satisface. Fijado un orden total
// de las variables, la menor asignación de cada órbita cumple α ≤lex α∘σ para todo σ del
// grupo, y por tanto toda cláusula implicada por ese predicado. Añadir tales cláusulas, para
// cualquier conjunto de automorfismos y siempre con el mismo orden, conserva la
// satisfacibilidad. Cada permutación candidata se comprueba cláusula a cláusula antes de
// usarse: la búsqueda es heurística, la verificación no.
use std::collections::HashMap;
use super::proof::ProofLog;
use std::hash::{BuildHasherDefault, Hasher};

pub type Clause = Vec<i32>;

pub struct SymmetryBreaking {
    // Pares (variable, imagen) movidos por cada automorfismo verificado, con IDs originales.
    pub generators: Vec<Vec<(u32, u32)>>,
    // Predicados lex-leader: se añaden a la fórmula y no introducen variables.
    pub clauses: Vec<Clause>,
}

// La ruptura es opcional: cualquier tope solo la debilita, nunca compromete la corrección.
const REFINEMENT_ROUNDS: usize = 64;
const MAX_STEPS: usize = 256;
const MAX_FAILURES: usize = 4;
const MAX_CHAIN: usize = 64;
const MAX_GROUP: usize = 64;
const WORK_PER_LITERAL: u64 = 4096;
const WORK_FLOOR: u64 = 1 << 20;
const WORK_CEILING: u64 = 1 << 28;

// Pares (variable, imagen) ordenados por variable; solo figuran las variables movidas.
type Permutation = Vec<(u32, u32)>;

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

// Las tablas de colores solo guardan enteros ya mezclados: basta un hasher multiplicativo.
#[derive(Default)]
struct Mixer(u64);
impl Hasher for Mixer {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = mix(self.0 ^ u64::from(byte));
        }
    }
    fn write_u32(&mut self, value: u32) {
        self.0 = mix(self.0 ^ u64::from(value));
    }
    fn write_u64(&mut self, value: u64) {
        self.0 = mix(self.0 ^ value);
    }
}
type Table<K, V> = HashMap<K, V, BuildHasherDefault<Mixer>>;

// Grafo bipartito de la fórmula normalizada: variables frente a cláusulas, aristas con signo.
// Todo en arreglos planos: una fórmula grande no paga una reserva por cláusula.
struct Graph {
    ids: Vec<u32>,
    // Cláusulas distintas, cada una con sus literales en orden creciente.
    clause_offsets: Vec<u32>,
    clause_literals: Vec<u32>,
    // (huella, cláusula) ordenado: la pertenencia compara los literales tras la huella.
    index: Vec<(u64, u32)>,
    // Ocurrencias por variable: (cláusula << 1) | signo.
    occurrence_offsets: Vec<u32>,
    occurrence_edges: Vec<u32>,
}
fn fingerprint(clause: &[u32]) -> u64 {
    clause.iter().fold(clause.len() as u64, |print, &literal| mix(print ^ u64::from(literal)))
}
impl Graph {
    fn new(formula: &[Clause]) -> Self {
        let mut ids: Vec<u32> = formula.iter().flatten().map(|literal| literal.unsigned_abs()).collect();
        assert!(u32::try_from(ids.len()).is_ok(), "Formula exceeds 2^32 literals");
        ids.sort_unstable();
        ids.dedup();
        // Con identificadores densos basta una tabla directa; si no, búsqueda binaria.
        let dense = ids.last().is_some_and(|&last| last as usize <= 4 * ids.len() + 1024);
        let mut table = vec![0_u32; if dense { ids[ids.len() - 1] as usize + 1 } else { 0 }];
        for (index, &id) in ids.iter().enumerate().take(if dense { ids.len() } else { 0 }) {
            table[id as usize] = index as u32;
        }
        let mut clause_offsets = vec![0_u32];
        let mut clause_literals: Vec<u32> = Vec::new();
        let mut index: Vec<(u64, u32)> = Vec::new();
        let mut packed: Vec<u32> = Vec::new();
        for clause in formula {
            packed.clear();
            packed.extend(clause.iter().map(|&literal| {
                let variable = if dense {
                    table[literal.unsigned_abs() as usize]
                } else {
                    ids.binary_search(&literal.unsigned_abs()).expect("Variable was indexed") as u32
                };
                (variable << 1) | u32::from(literal < 0)
            }));
            packed.sort_unstable();
            packed.dedup();
            // Una tautología se satisface siempre: no forma parte de la estructura.
            if packed.windows(2).any(|pair| pair[0] ^ 1 == pair[1]) {
                continue;
            }
            index.push((fingerprint(&packed), index.len() as u32));
            clause_literals.extend_from_slice(&packed);
            clause_offsets.push(clause_literals.len() as u32);
        }
        index.sort_unstable();
        let mut graph = Self {
            ids, clause_offsets, clause_literals, index,
            occurrence_offsets: Vec::new(), occurrence_edges: Vec::new(),
        };
        graph.drop_repeated_clauses();
        graph.index_occurrences();
        graph
    }
    // Una cláusula repetida comparte huella y literales con otra anterior: sobra.
    fn drop_repeated_clauses(&mut self) {
        let mut repeated = vec![false; self.clause_count()];
        let mut start = 0;
        while start < self.index.len() {
            let mut end = start + 1;
            while end < self.index.len() && self.index[end].0 == self.index[start].0 {
                end += 1;
            }
            for later in start + 1..end {
                let clause = self.index[later].1 as usize;
                repeated[clause] = (start..later).any(|earlier| {
                    let other = self.index[earlier].1 as usize;
                    !repeated[other] && self.clause(other) == self.clause(clause)
                });
            }
            start = end;
        }
        if !repeated.contains(&true) {
            return;
        }
        let mut offsets = vec![0_u32];
        let mut literals = Vec::with_capacity(self.clause_literals.len());
        let mut renamed = vec![0_u32; repeated.len()];
        for clause in 0..repeated.len() {
            if !repeated[clause] {
                renamed[clause] = (offsets.len() - 1) as u32;
                literals.extend_from_slice(self.clause(clause));
                offsets.push(literals.len() as u32);
            }
        }
        self.index.retain(|&(_, clause)| !repeated[clause as usize]);
        for entry in &mut self.index {
            entry.1 = renamed[entry.1 as usize];
        }
        self.clause_offsets = offsets;
        self.clause_literals = literals;
    }
    fn index_occurrences(&mut self) {
        let mut offsets = vec![0_u32; self.ids.len() + 1];
        for &literal in &self.clause_literals {
            offsets[(literal >> 1) as usize + 1] += 1;
        }
        for variable in 0..self.ids.len() {
            offsets[variable + 1] += offsets[variable];
        }
        let mut cursor = offsets.clone();
        let mut edges = vec![0_u32; self.clause_literals.len()];
        for clause in 0..self.clause_count() {
            for &literal in self.clause(clause) {
                let slot = &mut cursor[(literal >> 1) as usize];
                edges[*slot as usize] = ((clause as u32) << 1) | (literal & 1);
                *slot += 1;
            }
        }
        self.occurrence_offsets = offsets;
        self.occurrence_edges = edges;
    }
    fn clause_count(&self) -> usize {
        self.clause_offsets.len() - 1
    }
    fn clause(&self, clause: usize) -> &[u32] {
        &self.clause_literals[self.clause_offsets[clause] as usize..self.clause_offsets[clause + 1] as usize]
    }
    fn occurrences(&self, variable: usize) -> &[u32] {
        let range = self.occurrence_offsets[variable] as usize..self.occurrence_offsets[variable + 1] as usize;
        &self.occurrence_edges[range]
    }
    fn literals(&self) -> u64 {
        self.clause_literals.len() as u64
    }
    fn nodes(&self) -> usize {
        self.ids.len() + self.clause_count()
    }
    fn contains(&self, clause: &[u32]) -> bool {
        let print = fingerprint(clause);
        let start = self.index.partition_point(|&(other, _)| other < print);
        self.index[start..].iter().take_while(|&&(other, _)| other == print)
            .any(|&(_, known)| self.clause(known as usize) == clause)
    }
    fn degree(&self, node: usize) -> usize {
        let variables = self.ids.len();
        if node < variables { self.occurrences(node).len() } else { self.clause(node - variables).len() }
    }
    // Vecinos de un nodo con el signo de la arista.
    fn edges(&self, node: usize) -> impl Iterator<Item = (usize, u32)> + '_ {
        let variables = self.ids.len();
        let (edges, offset) = if node < variables {
            (self.occurrences(node), variables)
        } else {
            (self.clause(node - variables), 0)
        };
        edges.iter().map(move |&edge| (offset + (edge >> 1) as usize, edge & 1))
    }
    // Suma conmutativa: la firma no depende del orden de los vecinos.
    fn signature(&self, colors: &[u32], node: usize) -> u64 {
        self.edges(node).fold(0, |sum, (neighbor, sign)| {
            sum.wrapping_add(mix(u64::from(colors[neighbor]) << 1 | u64::from(sign)))
        })
    }
}

// Partición equitativa de partida. Color inicial: variables por frecuencia y polaridad,
// cláusulas por tamaño y número de literales negativos; después, refinamiento iterativo.
fn base_partition(graph: &Graph) -> (Vec<u32>, usize) {
    let variables = graph.ids.len();
    let mut names: Table<(u64, u64), u32> = Table::default();
    let mut colors: Vec<u32> = (0..graph.nodes()).map(|node| {
        let negative = graph.edges(node).filter(|&(_, sign)| sign == 1).count() as u64;
        let key = (u64::from(node >= variables), (graph.degree(node) as u64) << 32 | negative);
        let next = names.len() as u32;
        *names.entry(key).or_insert(next)
    }).collect();
    let mut count = names.len();
    let mut sizes = vec![0_u32; count];
    for _ in 0..REFINEMENT_ROUNDS {
        sizes.clear();
        sizes.resize(count, 0);
        for &color in &colors {
            sizes[color as usize] += 1;
        }
        names.clear();
        let mut next = 0_u32;
        let mut refined = Vec::with_capacity(graph.nodes());
        for node in 0..graph.nodes() {
            // Las variables van primero: si ya tienen todas un color propio, ninguna
            // permutación no trivial conserva los colores y no hay nada más que refinar.
            if node == variables && next as usize == variables {
                return (Vec::new(), 0);
            }
            // Una celda unitaria no puede dividirse: conserva su identidad sin firmar.
            let fresh = sizes[colors[node] as usize] == 1 || {
                let key = (u64::from(colors[node]), graph.signature(&colors, node));
                match names.get(&key) {
                    Some(&known) => {
                        refined.push(known);
                        false
                    }
                    None => {
                        names.insert(key, next);
                        true
                    }
                }
            };
            if fresh {
                refined.push(next);
                next += 1;
            }
        }
        let stable = next as usize == count;
        count = next as usize;
        colors = refined;
        if stable {
            break;
        }
    }
    (colors, count)
}

// Dos variables coinciden si comparten alguna cláusula. Es el criterio estructural que
// sustituye al orden de los identificadores: no cambia al renombrar las variables.
fn cooccur(graph: &Graph, first: u32, second: u32) -> bool {
    graph.occurrences(first as usize).iter().any(|&edge| {
        graph.clause((edge >> 1) as usize).iter().any(|&literal| literal >> 1 == second)
    })
}

fn apply(permutation: &Permutation, variable: u32) -> u32 {
    permutation.binary_search_by_key(&variable, |&(source, _)| source)
        .map_or(variable, |index| permutation[index].1)
}

// σ es biyectiva sobre su soporte y lleva cada cláusula afectada a otra cláusula de F.
// Una inyección de un conjunto finito en sí mismo es biyectiva: σ(F) = F.
fn is_automorphism(graph: &Graph, permutation: &Permutation) -> bool {
    let mut images: Vec<u32> = permutation.iter().map(|&(_, image)| image).collect();
    images.sort_unstable();
    if !images.iter().eq(permutation.iter().map(|(variable, _)| variable)) {
        return false;
    }
    let mut affected: Vec<u32> = permutation.iter().flat_map(|&(variable, _)| {
        graph.occurrences(variable as usize).iter().map(|&edge| edge >> 1)
    }).collect();
    affected.sort_unstable();
    affected.dedup();
    let mut mapped = Vec::new();
    affected.iter().all(|&clause| {
        mapped.clear();
        mapped.extend(graph.clause(clause as usize).iter().map(|&literal| {
            (apply(permutation, literal >> 1) << 1) | (literal & 1)
        }));
        mapped.sort_unstable();
        graph.contains(&mapped)
    })
}

fn is_involution(permutation: &Permutation) -> bool {
    permutation.iter().all(|&(variable, image)| apply(permutation, image) == variable)
}

// a∘b∘a: si a y b intercambian un mismo punto con w_a y w_b, esta intercambia w_a con w_b.
fn conjugate(a: &Permutation, b: &Permutation) -> Permutation {
    let mut support: Vec<u32> = a.iter().chain(b).map(|&(variable, _)| variable).collect();
    support.sort_unstable();
    support.dedup();
    support.into_iter().filter_map(|variable| {
        let image = apply(a, apply(b, apply(a, variable)));
        (image != variable).then_some((variable, image))
    }).collect()
}

// Individualización-refinamiento sobre dos copias del grafo: una coloración por lado y
// colores compartidos. Dos nodos de lados distintos con el mismo color son candidatos a
// corresponderse; una celda con distinto tamaño en cada lado descarta la búsqueda.
struct Search<'a> {
    graph: &'a Graph,
    base: &'a [u32],
    base_sizes: Vec<u32>,
    colors: [Vec<u32>; 2],
    sizes: [Vec<u32>; 2],
    touched: Vec<u32>,
    is_touched: Vec<bool>,
    dirty: [Vec<u32>; 2],
    is_dirty: [Vec<bool>; 2],
    entries: Vec<(u32, u64, u8, u32)>,
    work: u64,
    limit: u64,
}
impl<'a> Search<'a> {
    fn new(graph: &'a Graph, base: &'a [u32], count: usize) -> Self {
        let mut base_sizes = vec![0_u32; count];
        for &color in base {
            base_sizes[color as usize] += 1;
        }
        let nodes = graph.nodes();
        Self {
            graph, base,
            colors: [base.to_vec(), base.to_vec()],
            sizes: [base_sizes.clone(), base_sizes.clone()],
            base_sizes,
            touched: Vec::new(),
            is_touched: vec![false; nodes],
            dirty: [Vec::new(), Vec::new()],
            is_dirty: [vec![false; nodes], vec![false; nodes]],
            entries: Vec::new(),
            work: 0,
            limit: (WORK_FLOOR + WORK_PER_LITERAL * graph.literals()).min(WORK_CEILING),
        }
    }
    fn exhausted(&self) -> bool {
        self.work > self.limit
    }
    // Coste proporcional a lo tocado en la búsqueda anterior, no al tamaño del grafo.
    fn reset(&mut self) {
        for side in 0..2 {
            self.sizes[side].truncate(self.base_sizes.len());
            for &node in &self.dirty[side] {
                self.is_dirty[side][node as usize] = false;
            }
            self.dirty[side].clear();
        }
        for &node in &self.touched {
            let node = node as usize;
            let color = self.base[node] as usize;
            for side in 0..2 {
                self.colors[side][node] = self.base[node];
                self.sizes[side][color] = self.base_sizes[color];
            }
            self.is_touched[node] = false;
        }
        self.work += self.touched.len() as u64 + 1;
        self.touched.clear();
    }
    fn fresh(&mut self) -> u32 {
        self.sizes[0].push(0);
        self.sizes[1].push(0);
        (self.sizes[0].len() - 1) as u32
    }
    fn recolor(&mut self, side: usize, node: usize, color: u32) {
        let graph = self.graph;
        self.sizes[side][self.colors[side][node] as usize] -= 1;
        self.sizes[side][color as usize] += 1;
        self.colors[side][node] = color;
        if !self.is_touched[node] {
            self.is_touched[node] = true;
            self.touched.push(node as u32);
        }
        // Solo los vecinos de un nodo recoloreado pueden cambiar de firma.
        for (neighbor, _) in graph.edges(node) {
            if !self.is_dirty[side][neighbor] {
                self.is_dirty[side][neighbor] = true;
                self.dirty[side].push(neighbor as u32);
            }
        }
    }
    fn individualize(&mut self, source: usize, target: usize) {
        let color = self.fresh();
        self.recolor(0, source, color);
        self.recolor(1, target, color);
    }
    // Refinamiento incremental conjunto. Devuelve falso si alguna celda queda con distinto
    // tamaño en cada lado (ningún automorfismo extiende lo fijado) o se agota el presupuesto.
    fn refine(&mut self) -> bool {
        let graph = self.graph;
        while !(self.dirty[0].is_empty() && self.dirty[1].is_empty()) {
            let mut entries = std::mem::take(&mut self.entries);
            entries.clear();
            for side in 0..2 {
                let mut nodes = std::mem::take(&mut self.dirty[side]);
                for &node in &nodes {
                    let node = node as usize;
                    self.is_dirty[side][node] = false;
                    self.work += graph.degree(node) as u64 + 1;
                    let signature = graph.signature(&self.colors[side], node);
                    entries.push((self.colors[side][node], signature, side as u8, node as u32));
                }
                nodes.clear();
                self.dirty[side] = nodes;
            }
            entries.sort_unstable();
            let mut balanced = !self.exhausted();
            let mut start = 0;
            while balanced && start < entries.len() {
                let color = entries[start].0 as usize;
                let mut end = start;
                let mut dirty = [0_u32; 2];
                while end < entries.len() && entries[end].0 as usize == color {
                    dirty[entries[end].2 as usize] += 1;
                    end += 1;
                }
                // Toda la celda vio el mismo cambio: sigue siendo una celda, sin renombrar.
                let whole = dirty[0] == self.sizes[0][color] && dirty[1] == self.sizes[1][color];
                if !(whole && entries[start].1 == entries[end - 1].1) {
                    let mut run = start;
                    while run < end {
                        let mut stop = run;
                        let mut members = [0_u32; 2];
                        while stop < end && entries[stop].1 == entries[run].1 {
                            members[entries[stop].2 as usize] += 1;
                            stop += 1;
                        }
                        // Si nadie conserva el color, la primera parte se queda con él.
                        if !(whole && run == start) {
                            if members[0] != members[1] {
                                balanced = false;
                                break;
                            }
                            let fresh = self.fresh();
                            for &(_, _, side, node) in &entries[run..stop] {
                                self.recolor(side as usize, node as usize, fresh);
                            }
                        }
                        run = stop;
                    }
                    balanced = balanced && self.sizes[0][color] == self.sizes[1][color];
                }
                start = end;
            }
            self.entries = entries;
            if !balanced {
                return false;
            }
        }
        true
    }
    // Busca un automorfismo que lleve `source` a `target` moviendo lo menos posible: una
    // variable con el mismo color en ambos lados se deja fija y las desplazadas se emparejan
    // entre sí, prefiriendo la pareja que comparte cláusula. En modo `batch` cada celda
    // empareja de golpe todas sus desplazadas; si no, solo la menor. El resultado es un
    // candidato hasta que `is_automorphism` lo confirma.
    fn find(&mut self, source: usize, target: usize, batch: bool) -> Option<Permutation> {
        let graph = self.graph;
        let variables = graph.ids.len();
        self.reset();
        self.individualize(source, target);
        let mut displaced: Vec<u32> = Vec::new();
        for _ in 0..MAX_STEPS {
            if !self.refine() {
                return None;
            }
            displaced.clear();
            displaced.extend(self.touched.iter().copied().filter(|&node| {
                let node = node as usize;
                node < variables && self.colors[0][node] != self.colors[1][node]
            }));
            displaced.sort_unstable();
            let pending = displaced.iter().copied().find(|&node| {
                self.sizes[0][self.colors[0][node as usize] as usize] > 1
            });
            let Some(pivot) = pending else {
                // Celdas unitarias: el color determina la imagen de cada variable desplazada.
                let targets: Table<u32, u32> = displaced.iter()
                    .map(|&node| (self.colors[1][node as usize], node)).collect();
                return displaced.iter().map(|&node| {
                    targets.get(&self.colors[0][node as usize]).map(|&image| (node, image))
                }).collect();
            };
            if !batch {
                let color = self.colors[0][pivot as usize];
                let mut candidates = displaced.iter().copied()
                    .filter(|&node| self.colors[1][node as usize] == color);
                let partner = candidates.clone().find(|&node| cooccur(graph, pivot, node))
                    .or_else(|| candidates.next())?;
                self.individualize(pivot as usize, partner as usize);
                continue;
            }
            let mut sources: Vec<(u32, u32)> = displaced.iter()
                .map(|&node| (self.colors[0][node as usize], node))
                .filter(|&(color, _)| self.sizes[0][color as usize] > 1).collect();
            let mut targets: Vec<(u32, u32)> = displaced.iter()
                .map(|&node| (self.colors[1][node as usize], node))
                .filter(|&(color, _)| self.sizes[1][color as usize] > 1).collect();
            sources.sort_unstable();
            targets.sort_unstable();
            if sources.len() != targets.len() {
                return None;
            }
            let mut used = vec![false; targets.len()];
            let mut start = 0;
            while start < sources.len() {
                let color = sources[start].0;
                let mut end = start;
                while end < sources.len() && sources[end].0 == color {
                    end += 1;
                }
                if targets[start].0 != color || targets[end - 1].0 != color {
                    return None;
                }
                for index in start..end {
                    let node = sources[index].1;
                    let near = if end - start <= MAX_GROUP {
                        (start..end).find(|&slot| !used[slot] && cooccur(graph, node, targets[slot].1))
                    } else {
                        None
                    };
                    let slot = near.or_else(|| (start..end).find(|&slot| !used[slot]))?;
                    used[slot] = true;
                    self.individualize(node as usize, targets[slot].1 as usize);
                }
                start = end;
            }
        }
        None
    }
}

struct Orbits(Vec<u32>);
impl Orbits {
    fn find(&mut self, mut variable: u32) -> u32 {
        while self.0[variable as usize] != variable {
            self.0[variable as usize] = self.0[self.0[variable as usize] as usize];
            variable = self.0[variable as usize];
        }
        variable
    }
    fn join(&mut self, first: u32, second: u32) {
        let (first, second) = (self.find(first), self.find(second));
        self.0[first.max(second) as usize] = first.min(second);
    }
}

// Una familia de involuciones (v w1), (v w2), ... que solo comparten el bloque de v describe
// filas intercambiables: bloque, g1(bloque), g2(bloque), ... Se colocan consecutivas en el
// orden global, alineadas por columnas, y dentro de cada fila van primero las columnas
// cuyo par ya excluye la fórmula, porque su conjunto lex-leader colapsa a una cláusula más
// corta. El orden solo afecta a la fuerza de los predicados, nunca a su validez.
fn place_rows(
    graph: &Graph,
    members: &[(u32, Permutation)],
    order: &mut Vec<u32>,
    placed: &mut [bool],
) {
    let (first, second) = (&members[0].1, &members[1].1);
    let mut columns: Vec<u32> = first.iter().map(|&(variable, _)| variable)
        .filter(|&variable| apply(second, variable) != variable).collect();
    let mut rows: Vec<u32> = columns.clone();
    for (_, involution) in members {
        rows.extend(columns.iter().map(|&variable| apply(involution, variable)));
    }
    let distinct = {
        let mut sorted = rows.clone();
        sorted.sort_unstable();
        sorted.windows(2).all(|pair| pair[0] != pair[1])
    };
    if columns.is_empty() || !distinct || rows.iter().any(|&variable| placed[variable as usize]) {
        return;
    }
    columns.sort_unstable_by_key(|&variable| {
        let mut exclusion = [(variable << 1) | 1, (apply(first, variable) << 1) | 1];
        exclusion.sort_unstable();
        (!graph.contains(&exclusion), variable)
    });
    order.extend(&columns);
    for (_, involution) in members {
        order.extend(columns.iter().map(|&variable| apply(involution, variable)));
    }
    for variable in rows {
        placed[variable as usize] = true;
    }
}

// Agrupamiento avaro. Para la menor variable v de cada celda aún sin mover se buscan
// automorfismos v → w: primero hacia las w de su celda que comparten cláusula con v
// (variables intercambiables dentro de las mismas cláusulas) y después hacia las que
// quedan fuera de la órbita ya generada. De cada estrella de involuciones (v w1),
// (v w2), ... se derivan por conjugación las transposiciones adyacentes (w1 w2),
// (w2 w3), ..., cuyo predicado lex-leader ordena las filas por completo.
// Devuelve los automorfismos a romper y la posición de cada variable en el orden global.
fn discover(graph: &Graph) -> (Vec<Permutation>, Vec<u32>) {
    let variables = graph.ids.len();
    let (base, count) = base_partition(graph);
    if count == 0 {
        return (Vec::new(), (0..variables as u32).collect());
    }
    // Celdas de variables como tramos de una lista ordenada por (color, variable).
    let mut members: Vec<u32> = (0..variables as u32)
        .filter(|&variable| !graph.occurrences(variable as usize).is_empty()).collect();
    members.sort_unstable_by_key(|&variable| (base[variable as usize], variable));
    let cell_of = |variable: u32| {
        let color = base[variable as usize];
        let start = members.partition_point(|&other| base[other as usize] < color);
        let end = members.partition_point(|&other| base[other as usize] <= color);
        &members[start..end]
    };
    let mut search = Search::new(graph, &base, count);
    let mut orbits = Orbits((0..variables as u32).collect());
    let mut moved = vec![false; variables];
    let mut breakers: Vec<Permutation> = Vec::new();
    let mut families: Vec<Vec<(u32, Permutation)>> = Vec::new();
    let mut order: Vec<u32> = Vec::with_capacity(variables);
    let mut placed = vec![false; variables];
    for origin in 0..variables as u32 {
        if search.exhausted() {
            break;
        }
        if moved[origin as usize] || graph.occurrences(origin as usize).is_empty() {
            continue;
        }
        let cell = cell_of(origin);
        if cell.len() < 2 {
            continue;
        }
        let (near, far): (Vec<u32>, Vec<u32>) = cell.iter().copied()
            .filter(|&target| target != origin)
            .partition(|&target| cooccur(graph, origin, target));
        let mut star: Vec<(u32, Permutation)> = Vec::new();
        let mut failures = 0;
        for (target, close) in near.into_iter().map(|target| (target, true))
            .chain(far.into_iter().map(|target| (target, false)))
        {
            // Una vecina se prueba siempre; una lejana, solo si aún no es alcanzable.
            if !close && orbits.find(origin) == orbits.find(target) {
                continue;
            }
            // Primero el emparejamiento por lotes; si no cuadra, el individual.
            let found = [true, false].into_iter().find_map(|batch| {
                search.find(origin as usize, target as usize, batch)
                    .filter(|permutation| is_automorphism(graph, permutation))
            });
            let Some(permutation) = found else {
                failures += 1;
                if failures == MAX_FAILURES || search.exhausted() {
                    break;
                }
                continue;
            };
            failures = 0;
            for &(variable, image) in &permutation {
                orbits.join(variable, image);
                moved[variable as usize] = true;
            }
            star.push((target, permutation));
        }
        // Familias de involuciones compatibles: cada una fija el destino de las demás.
        let mut local: Vec<Vec<(u32, Permutation)>> = Vec::new();
        for (target, permutation) in star {
            if !is_involution(&permutation) {
                breakers.push(permutation);
                continue;
            }
            let family = local.iter_mut().find(|family| family.iter().all(|(other, mate)| {
                apply(mate, target) == target && apply(&permutation, *other) == *other
            }));
            match family {
                Some(family) => family.push((target, permutation)),
                None => local.push(vec![(target, permutation)]),
            }
        }
        for family in local {
            if family.len() > 1 {
                place_rows(graph, &family, &mut order, &mut placed);
            }
            families.push(family);
        }
    }
    order.extend((0..variables as u32).filter(|&variable| !placed[variable as usize]));
    let mut rank = vec![0_u32; variables];
    for (position, &variable) in order.iter().enumerate() {
        rank[variable as usize] = position as u32;
    }
    for mut family in families {
        family.sort_unstable_by_key(|(target, _)| rank[*target as usize]);
        breakers.push(family[0].1.clone());
        for pair in family.windows(2) {
            let adjacent = conjugate(&pair[0].1, &pair[1].1);
            debug_assert!(is_automorphism(graph, &adjacent));
            breakers.push(adjacent);
        }
    }
    (breakers, rank)
}

// Predicado lex-leader de σ con las variables en el orden global: para cada variable
// movida u, (pares anteriores iguales) ⇒ u ≤ σ(u). Se emite con el antecedente reforzado
// «pares anteriores ambos falsos»: sigue siendo consecuencia del predicado y no necesita
// variables auxiliares. Si F ya contiene (¬u ∨ ¬σ(u)), la resolución elimina σ(u).
// `heads` recibe la variable u de cada cláusula emitida: son las únicas que el orden de
// una traza necesita comparar.
fn lex_leader(
    graph: &Graph, permutation: &Permutation, rank: &[u32],
    emitted: &mut Vec<Vec<u32>>, heads: &mut Vec<u32>,
) {
    let mut pairs = permutation.clone();
    pairs.sort_unstable_by_key(|&(variable, _)| rank[variable as usize]);
    let mut prefix: Vec<u32> = Vec::new();
    let mut conjuncts = 0;
    for &(variable, image) in &pairs {
        let (smaller, larger) = (variable << 1, image << 1);
        // Con u ya en el prefijo la cláusula contendría u y ¬u: es una tautología.
        if !prefix.contains(&smaller) {
            if conjuncts == MAX_CHAIN {
                break;
            }
            conjuncts += 1;
            heads.push(variable);
            let mut clause = prefix.clone();
            clause.push(smaller | 1);
            let mut exclusion = [smaller | 1, larger | 1];
            exclusion.sort_unstable();
            if !graph.contains(&exclusion) {
                clause.push(larger);
            }
            clause.sort_unstable();
            clause.dedup();
            emitted.push(clause);
        }
        for literal in [smaller, larger] {
            if !prefix.contains(&literal) {
                prefix.push(literal);
            }
        }
    }
}

pub fn break_symmetries(formula: &[Clause]) -> SymmetryBreaking {
    break_symmetries_with_proof(formula, None)
}

// Bytes de traza admitidos para el orden y las dominancias. Los pesos del orden son
// potencias de dos en decimal, así que el coste crece con el cuadrado de las variables
// ordenadas; lo que no cabe no se certifica y, por tanto, no se rompe.
const DOMINANCE_BUDGET: usize = 1 << 26;

// Número de generadores, en su orden, cuya certificación cabe en el presupuesto.
fn affordable(breakers: &[Permutation], heads: &[u32], ends: &[(usize, usize)], variables: usize) -> usize {
    let mut kept = breakers.len();
    let mut ordered = vec![false; variables];
    while kept > 0 {
        ordered.fill(false);
        let mut count = 0;
        for &variable in &heads[..ends[kept - 1].1] {
            count += usize::from(!std::mem::replace(&mut ordered[variable as usize], true));
        }
        // Dos términos por variable ordenada en la definición y por cada una que mueve un
        // generador, con pesos de hasta 0.302 dígitos por variable; el testigo es lineal.
        let mut terms = count;
        let mut witness = 0;
        for permutation in &breakers[..kept] {
            terms += permutation.iter().filter(|&&(variable, _)| ordered[variable as usize]).count();
            witness += permutation.len();
        }
        if 2 * terms * (count * 3 / 10 + 12) + 24 * witness <= DOMINANCE_BUDGET {
            break;
        }
        kept /= 2;
    }
    kept
}

// Con traza, cada automorfismo se justifica por dominancia antes de sus predicados. El
// orden de la prueba son las cabezas de las cadenas lex-leader, en su mismo orden global:
// bajo la negación de un predicado, las posiciones anteriores a su cabeza que σ mueve
// valen cero, la cabeza resta su peso y las posteriores no llegan a compensarlo, así que
// el predicado es RUP a partir de z ≤lex σ(z). Un formato sin regla de dominancia no puede
// certificar nada de esto: no se rompe nada.
pub fn break_symmetries_with_proof(
    formula: &[Clause], mut proof: Option<&mut (dyn ProofLog + '_)>,
) -> SymmetryBreaking {
    let unsupported = proof.as_ref().is_some_and(|log| !log.supports_dominance());
    if unsupported || formula.iter().any(Vec::is_empty) {
        return SymmetryBreaking { generators: Vec::new(), clauses: Vec::new() };
    }
    let graph = Graph::new(formula);
    let (mut breakers, rank) = discover(&graph);
    let id = |variable: u32| graph.ids[variable as usize];
    let signed = |literal: u32| {
        let variable = id(literal >> 1) as i32;
        if literal & 1 == 0 { variable } else { -variable }
    };
    let mut emitted = Vec::new();
    // Dónde acaban las cláusulas y las cabezas de cada generador.
    let mut heads = Vec::new();
    let mut ends = Vec::with_capacity(breakers.len());
    for permutation in &breakers {
        lex_leader(&graph, permutation, &rank, &mut emitted, &mut heads);
        ends.push((emitted.len(), heads.len()));
    }
    if let Some(log) = proof.as_deref_mut() {
        let kept = affordable(&breakers, &heads, &ends, graph.ids.len());
        breakers.truncate(kept);
        ends.truncate(kept);
        let (clauses, ordered) = ends.last().copied().unwrap_or((0, 0));
        emitted.truncate(clauses);
        heads.truncate(ordered);
        heads.sort_unstable_by_key(|&variable| rank[variable as usize]);
        heads.dedup();
        if !heads.is_empty() {
            log.order(&heads.iter().map(|&variable| id(variable) as i32).collect::<Vec<i32>>());
        }
        let mut line: Vec<i32> = Vec::new();
        let mut first = 0;
        for (permutation, &(end, _)) in breakers.iter().zip(&ends) {
            if end > first {
                let pairs: Vec<(i32, i32)> = permutation.iter()
                    .map(|&(variable, image)| (id(variable) as i32, id(image) as i32)).collect();
                log.dominance(&pairs);
                for clause in &emitted[first..end] {
                    line.clear();
                    line.extend(clause.iter().map(|&literal| signed(literal)));
                    log.add(&line);
                }
            }
            first = end;
        }
    }
    emitted.sort_unstable();
    emitted.dedup();
    emitted.retain(|clause| !graph.contains(clause));
    SymmetryBreaking {
        generators: breakers.iter().map(|permutation| {
            permutation.iter().map(|&(variable, image)| (id(variable), id(image))).collect()
        }).collect(),
        clauses: emitted.iter().map(|clause| clause.iter().map(|&literal| signed(literal)).collect()).collect(),
    }
}

}

pub mod cdcl {
use std::collections::{HashSet, VecDeque};
use super::proof::ProofLog;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
pub type Clause = Vec<i32>;
pub type Formula = Vec<Clause>;
#[derive(Clone, Copy, Debug, Default)]
pub struct SolveStats {
    pub decisions: usize,
    pub cube_decisions: usize,
    pub propagations: usize,
    pub conflicts: usize,
    pub learned_clauses: usize,
    pub backjumps: usize,
    pub nonchronological_backjumps: usize,
    pub restarts: usize,
    pub deleted_clauses: usize,
    pub database_reductions: usize,
    pub reignitions: usize,
    pub reignition_flips: usize,
    pub reignition_ms: f64,
    pub exported_clauses: usize,
    pub imported_clauses: usize,
    pub dropped_exports: usize,
    // Siempre 0 desde la Fase 12b: solo se importa en el nivel 0.
    pub import_backjumps: usize,
}
impl SolveStats {
    fn add_search(&mut self, other: Self) {
        self.decisions += other.decisions;
        self.cube_decisions += other.cube_decisions;
        self.propagations += other.propagations;
        self.conflicts += other.conflicts;
        self.learned_clauses += other.learned_clauses;
        self.backjumps += other.backjumps;
        self.nonchronological_backjumps += other.nonchronological_backjumps;
        self.restarts += other.restarts;
        self.deleted_clauses += other.deleted_clauses;
        self.database_reductions += other.database_reductions;
        self.reignitions += other.reignitions;
        self.reignition_flips += other.reignition_flips;
        self.reignition_ms += other.reignition_ms;
        self.exported_clauses += other.exported_clauses;
        self.imported_clauses += other.imported_clauses;
        self.dropped_exports += other.dropped_exports;
        self.import_backjumps += other.import_backjumps;
    }
}
#[derive(Clone, Copy, Debug)]
pub struct OracleConfig {
    // Tope de flips por variable en cada ejecución; 0 desactiva el oráculo.
    pub flips_per_variable: usize,
    pub seed: u64,
}
impl Default for OracleConfig {
    fn default() -> Self {
        Self { flips_per_variable: 16, seed: 0x243f_6a88_85a3_08d3 }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct ReignitionConfig {
    // Conflictos mínimos entre ejecuciones del oráculo; 0 lo desactiva.
    pub interval: usize,
    pub oracle: OracleConfig,
}
impl Default for ReignitionConfig {
    fn default() -> Self {
        Self { interval: 500, oracle: OracleConfig::default() }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct SwarmConfig {
    pub workers: usize,
    pub cubes_per_worker: usize,
    // Ventana de cláusulas retenidas en el anillo; 0 desactiva el intercambio.
    pub share_capacity: usize,
    // Conflictos mínimos entre sincronizaciones; se importa solo al reiniciar.
    pub share_interval: usize,
    pub reignition: ReignitionConfig,
    pub share_enabled: bool,
}
impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            workers: std::thread::available_parallelism().map_or(1, |count| count.get()),
            cubes_per_worker: 4,
            share_capacity: 8192,
            share_interval: 64,
            reignition: ReignitionConfig::default(),
            share_enabled: true,
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct SwarmStats {
    pub workers: usize,
    pub cubes_generated: usize,
    pub cubes_started: usize,
    pub cubes_unsat: usize,
    pub cubes_cancelled: usize,
    pub total: SolveStats,
    // Snapshot auditable: IDs originales; nunca contiene cláusulas vacías.
    pub shared_clauses: Vec<Vec<i32>>,
}
// Solo se exportan cláusulas glue; una importada entra con LBD 3 y debe ganarse la permanencia.
const EXPORT_LBD: u32 = 2;
const IMPORT_LBD: u32 = 3;
const EXPORT_BATCH: usize = 64;
const KNOWN_LIMIT: usize = 1 << 20;
// Lote inmutable de cláusulas codificadas como [n, literales...].
struct Batch {
    owner: u32,
    count: usize,
    words: Arc<[u32]>,
}
struct CloudLog {
    batches: VecDeque<Batch>,
    // Secuencia del próximo lote; el más antiguo retenido es `next - batches.len()`.
    next: u64,
    clauses: usize,
}
// Anillo de lotes sin tope de vida: `capacity` es la ventana de cláusulas retenidas.
pub struct CloudPool { log: Mutex<CloudLog>, capacity: usize }
impl CloudPool {
    pub fn new(capacity: usize) -> Self {
        Self { log: Mutex::new(CloudLog { batches: VecDeque::new(), next: 0, clauses: 0 }), capacity }
    }
    // El candado solo cubre un push de puntero y la expulsión de los lotes más antiguos.
    fn publish(&self, batch: Batch, wait: bool) -> bool {
        let guard = if wait { self.log.lock().ok() } else { self.log.try_lock().ok() };
        let Some(mut log) = guard else { return false; };
        log.clauses += batch.count;
        log.batches.push_back(batch);
        log.next += 1;
        while log.clauses > self.capacity {
            let Some(oldest) = log.batches.pop_front() else { break; };
            log.clauses -= oldest.count;
        }
        true
    }
    // Lotes ajenos desde `cursor`; un lector rezagado continúa en el más antiguo retenido.
    fn harvest(&self, owner: u32, cursor: &mut u64, batches: &mut Vec<Arc<[u32]>>) {
        let Ok(log) = self.log.try_lock() else { return; };
        let first = log.next - log.batches.len() as u64;
        let skip = cursor.saturating_sub(first) as usize;
        batches.extend(log.batches.iter().skip(skip)
            .filter(|batch| batch.owner != owner).map(|batch| Arc::clone(&batch.words)));
        *cursor = log.next;
    }
    fn snapshot(&self, variables: &[u32]) -> Vec<Vec<i32>> {
        let Ok(log) = self.log.lock() else { return Vec::new(); };
        log.batches.iter().flat_map(|batch| clauses_of(&batch.words)).map(|clause| {
            clause.iter().map(|&literal| {
                let variable = variables[(literal >> 1) as usize] as i32;
                if literal & 1 == 0 { variable } else { -variable }
            }).collect()
        }).collect()
    }
}
fn clauses_of(words: &[u32]) -> impl Iterator<Item = &[u32]> {
    let mut rest = words;
    std::iter::from_fn(move || {
        let (&size, tail) = rest.split_first()?;
        let (clause, tail) = tail.split_at(size as usize);
        rest = tail;
        Some(clause)
    })
}
// Huella independiente del orden de los literales: el filtro no necesita canonizar.
fn fingerprint(clause: &[u32]) -> u64 {
    clause.iter().fold(clause.len() as u64, |sum, &literal| {
        let mut mixed = u64::from(literal).wrapping_add(0x9e37_79b9_7f4a_7c15);
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        sum.wrapping_add(mixed ^ (mixed >> 31))
    })
}
// Contexto de intercambio de un worker: sobrevive entre cubos, así que nada se reimporta.
struct Sharing<'a> {
    pool: &'a CloudPool,
    owner: u32,
    interval: u64,
    cursor: u64,
    last_sync: u64,
    known: HashSet<u64>,
    outbox: Vec<u32>,
    pending: usize,
    inbox: Vec<u32>,
}
impl<'a> Sharing<'a> {
    fn new(pool: &'a CloudPool, owner: u32, interval: u64) -> Self {
        Self {
            pool, owner, interval, cursor: 0, last_sync: 0, known: HashSet::new(),
            outbox: Vec::new(), pending: 0, inbox: Vec::new(),
        }
    }
    fn remember(&mut self, clause: &[u32]) -> bool {
        // Olvidar el filtro solo permite algún duplicado; nunca afecta a la corrección.
        if self.known.len() >= KNOWN_LIMIT {
            self.known.clear();
        }
        self.known.insert(fingerprint(clause))
    }
    fn export(&mut self, clause: &[u32], stats: &mut SolveStats) {
        if !self.remember(clause) {
            return;
        }
        self.outbox.push(clause.len() as u32);
        self.outbox.extend_from_slice(clause);
        self.pending += 1;
        if self.pending >= EXPORT_BATCH.min(self.pool.capacity) {
            self.flush(stats, false);
        }
    }
    fn flush(&mut self, stats: &mut SolveStats, wait: bool) {
        if self.pending == 0 {
            return;
        }
        // Un lote nunca excede el anillo: se conservan las cláusulas más recientes.
        let count = self.pending.min(self.pool.capacity);
        let mut start = 0;
        for _ in count..self.pending {
            start += 1 + self.outbox[start] as usize;
        }
        let batch = Batch { owner: self.owner, count, words: Arc::from(&self.outbox[start..]) };
        if self.pool.publish(batch, wait) {
            stats.exported_clauses += count;
            stats.dropped_exports += self.pending - count;
            self.outbox.clear();
            self.pending = 0;
        }
    }
    // Publica lo pendiente y deja en `inbox` las cláusulas ajenas aún no vistas.
    fn exchange(&mut self, conflicts: u64, stats: &mut SolveStats) {
        self.flush(stats, false);
        let mut batches = Vec::new();
        self.pool.harvest(self.owner, &mut self.cursor, &mut batches);
        for words in batches {
            for clause in clauses_of(&words) {
                if self.remember(clause) {
                    self.inbox.push(clause.len() as u32);
                    self.inbox.extend_from_slice(clause);
                }
            }
        }
        self.last_sync = conflicts;
    }
}
enum SearchOutcome { Sat(Vec<i32>), Unsat, Cancelled }
struct SearchControl<'a, 'p> {
    cube: &'a [u32],
    cancel: Option<&'a AtomicBool>,
    sharing: Option<&'a mut Sharing<'p>>,
    // Traza DRAT de la búsqueda en serie; el enjambre no la emite.
    proof: Option<&'a mut dyn ProofLog>,
}
// Pesos enteros de probSAT, 2^16 * 2.06^(-roturas): cuanto más rompe un flip, menos probable.
const BREAK_WEIGHTS: [u64; 16] =
    [65536, 31814, 15443, 7497, 3639, 1767, 858, 416, 202, 98, 48, 23, 11, 5, 3, 1];
// Oráculo discreto: búsqueda local estocástica sobre bits, sin aritmética flotante.
#[derive(Clone)]
struct Walker {
    literals: Vec<u32>,
    offsets: Vec<u32>,
    occurrences: Vec<u32>,
    occurrence_offsets: Vec<u32>,
    values: Vec<bool>,
    fixed: Vec<bool>,
    // Literales verdaderos de cada cláusula y su XOR: con uno solo, el XOR es ese literal.
    satisfied: Vec<u32>,
    critical: Vec<u32>,
    breaks: Vec<u32>,
    unsat: Vec<u32>,
    slots: Vec<u32>,
    // Mejor asignación vista y variables volteadas desde ella dentro de la ráfaga en curso.
    best: usize,
    best_values: Vec<bool>,
    journal: Vec<u32>,
    synced: usize,
    rng: u64,
}
impl Walker {
    fn new(variables: usize, literals: Vec<u32>, offsets: Vec<u32>, seed: u64) -> Self {
        let clauses = offsets.len() - 1;
        let mut occurrence_offsets = vec![0_u32; 2 * variables + 1];
        for &literal in &literals {
            occurrence_offsets[literal as usize + 1] += 1;
        }
        for literal in 0..2 * variables {
            occurrence_offsets[literal + 1] += occurrence_offsets[literal];
        }
        let mut cursor = occurrence_offsets.clone();
        let mut occurrences = vec![0_u32; literals.len()];
        for clause in 0..clauses {
            for &literal in &literals[offsets[clause] as usize..offsets[clause + 1] as usize] {
                occurrences[cursor[literal as usize] as usize] = clause as u32;
                cursor[literal as usize] += 1;
            }
        }
        let mut walker = Self {
            literals, offsets, occurrences, occurrence_offsets,
            values: vec![false; variables],
            fixed: vec![false; variables],
            satisfied: vec![0; clauses],
            critical: vec![0; clauses],
            breaks: vec![0; variables],
            unsat: Vec::new(),
            slots: vec![NONE; clauses],
            best: 0,
            best_values: Vec::new(),
            journal: Vec::new(),
            synced: 0,
            rng: seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1,
        };
        // Asignación inicial aleatoria.
        for variable in 0..variables {
            walker.values[variable] = walker.next() >> 63 == 1;
        }
        for clause in 0..clauses {
            let (mut count, mut xor) = (0, 0);
            for position in walker.offsets[clause] as usize..walker.offsets[clause + 1] as usize {
                let literal = walker.literals[position];
                if walker.values[variable_of(literal)] == (literal & 1 == 0) {
                    count += 1;
                    xor ^= literal;
                }
            }
            walker.satisfied[clause] = count;
            walker.critical[clause] = xor;
            match count {
                0 => walker.violate(clause),
                1 => walker.breaks[variable_of(xor)] += 1,
                _ => {}
            }
        }
        walker.settle();
        walker
    }
    fn next(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn reseed(&mut self, salt: u64) {
        self.rng = (self.rng ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15)) | 1;
    }
    // La asignación actual pasa a ser la referencia: se usa tras fijar literales de la raíz.
    fn settle(&mut self) {
        self.best = self.unsat.len();
        self.best_values.clone_from(&self.values);
        self.journal.clear();
    }
    fn violate(&mut self, clause: usize) {
        self.slots[clause] = self.unsat.len() as u32;
        self.unsat.push(clause as u32);
    }
    fn repair(&mut self, clause: usize) {
        let slot = self.slots[clause];
        let last = self.unsat.pop().expect("A violated clause is listed");
        if last as usize != clause {
            self.unsat[slot as usize] = last;
            self.slots[last as usize] = slot;
        }
        self.slots[clause] = NONE;
    }
    // Coste proporcional a las ocurrencias de la variable, no al tamaño de la fórmula.
    fn flip(&mut self, variable: usize) {
        let gained = ((variable as u32) << 1) | u32::from(self.values[variable]);
        let lost = gained ^ 1;
        self.values[variable] = !self.values[variable];
        let range = self.occurrence_offsets[gained as usize]..self.occurrence_offsets[gained as usize + 1];
        for index in range {
            let clause = self.occurrences[index as usize] as usize;
            let before = self.satisfied[clause];
            self.satisfied[clause] = before + 1;
            match before {
                0 => {
                    self.repair(clause);
                    self.breaks[variable] += 1;
                }
                // El antiguo único verdadero deja de ser crítico.
                1 => self.breaks[variable_of(self.critical[clause])] -= 1,
                _ => {}
            }
            self.critical[clause] ^= gained;
        }
        let range = self.occurrence_offsets[lost as usize]..self.occurrence_offsets[lost as usize + 1];
        for index in range {
            let clause = self.occurrences[index as usize] as usize;
            let after = self.satisfied[clause] - 1;
            self.satisfied[clause] = after;
            self.critical[clause] ^= lost;
            match after {
                0 => {
                    self.violate(clause);
                    self.breaks[variable] -= 1;
                }
                // El verdadero que queda pasa a ser crítico.
                1 => self.breaks[variable_of(self.critical[clause])] += 1,
                _ => {}
            }
        }
    }
    fn fix(&mut self, literal: u32) {
        let variable = variable_of(literal);
        if self.values[variable] != (literal & 1 == 0) {
            self.flip(variable);
        }
        self.fixed[variable] = true;
    }
    #[inline]
    fn weight(&self, variable: usize) -> u64 {
        if self.fixed[variable] {
            0
        } else {
            BREAK_WEIGHTS[(self.breaks[variable] as usize).min(BREAK_WEIGHTS.len() - 1)]
        }
    }
    // Ráfaga probSAT: cláusula violada al azar y flip con probabilidad decreciente en roturas.
    // La caminata continúa entre ráfagas; `best_values` conserva la mejor asignación vista.
    fn walk(&mut self, flips: usize, cancel: Option<&AtomicBool>) -> usize {
        let mut done = 0;
        let mut improved = false;
        self.journal.clear();
        while done < flips && !self.unsat.is_empty() {
            if done & 4095 == 4095 && cancel.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                break;
            }
            let draw = self.next();
            let clause = self.unsat[(draw % self.unsat.len() as u64) as usize] as usize;
            let (start, end) = (self.offsets[clause] as usize, self.offsets[clause + 1] as usize);
            let total: u64 = (start..end).map(|position| self.weight(variable_of(self.literals[position]))).sum();
            // Todos sus literales son falsos en la raíz: el CDCL ya tiene ese conflicto.
            if total == 0 {
                break;
            }
            let mut pick = self.next() % total;
            for position in start..end {
                let variable = variable_of(self.literals[position]);
                let weight = self.weight(variable);
                if pick < weight {
                    self.flip(variable);
                    if improved {
                        self.journal.push(variable as u32);
                    }
                    break;
                }
                pick -= weight;
            }
            done += 1;
            if self.unsat.len() < self.best {
                self.best = self.unsat.len();
                self.journal.clear();
                improved = true;
            }
        }
        if improved {
            // La mejor asignación es la actual deshaciendo los flips posteriores a ella.
            self.best_values.clone_from(&self.values);
            for &variable in &self.journal {
                self.best_values[variable as usize] = !self.best_values[variable as usize];
            }
        }
        done
    }
}
const DECAY: f64 = 0.95;
const CLEANUP_INTERVAL: u64 = 64;
const MAX_ACTIVITY: f64 = 1e100;
// Arena: [tamaño, LBD | flags, literales...]. Unidades y binarias no viven en ella.
const HEADER: usize = 2;
const LEARNED: u32 = 1 << 31;
const GARBAGE: u32 = 1 << 30;
const LOCKED: u32 = 1 << 29;
const USED: u32 = 1 << 28;
const LBD_MASK: u32 = USED - 1;
// Bit alto de `Watcher::reference` y de una razón: el resto es el otro literal de una binaria.
const BINARY: u32 = 1 << 31;
const NO_REASON: u32 = u32::MAX;
const NONE: u32 = u32::MAX;
const RESTART_MARGIN: f64 = 1.1;
const RESTART_INTERVAL: u64 = 2;
const LUBY_UNIT: u64 = 1024;
const MODE_INTERVAL: u64 = 1000;
// Tras la ignición inicial el oráculo gasta como mucho un flip por cada ORACLE_EFFORT
// literales propagados: su coste queda acotado por el trabajo del CDCL, no por V.
const ORACLE_EFFORT: u64 = 8;
#[derive(Clone, Copy)]
struct Watcher {
    blocker: u32,
    reference: u32,
}
#[derive(Clone, Copy)]
enum Conflict {
    Binary(u32, u32),
    Long(u32),
}
// Media exponencial con corrección de sesgo: la primera muestra ya es representativa.
#[derive(Clone, Copy)]
struct Ema {
    value: f64,
    biased: f64,
    alpha: f64,
    correction: f64,
}
impl Ema {
    fn new(alpha: f64) -> Self {
        Self { value: 0.0, biased: 0.0, alpha, correction: 1.0 }
    }
    fn update(&mut self, sample: f64) {
        self.biased += self.alpha * (sample - self.biased);
        self.correction *= 1.0 - self.alpha;
        self.value = self.biased / (1.0 - self.correction);
    }
}
// Cola de prioridad indexada (EVSIDS): mayor puntuación primero; a igualdad, menor índice.
#[derive(Clone)]
struct Heap {
    order: Vec<u32>,
    position: Vec<u32>,
}
impl Heap {
    fn new(variables: usize) -> Self {
        Self { order: Vec::with_capacity(variables), position: vec![NONE; variables] }
    }
    #[inline]
    fn before(scores: &[f64], a: u32, b: u32) -> bool {
        let (x, y) = (scores[a as usize], scores[b as usize]);
        x > y || (x == y && a < b)
    }
    fn contains(&self, variable: u32) -> bool {
        self.position[variable as usize] != NONE
    }
    fn insert(&mut self, variable: u32, scores: &[f64]) {
        if self.contains(variable) {
            return;
        }
        self.order.push(variable);
        self.up(self.order.len() - 1, scores);
    }
    fn increase(&mut self, variable: u32, scores: &[f64]) {
        self.up(self.position[variable as usize] as usize, scores);
    }
    fn pop(&mut self, scores: &[f64]) -> Option<u32> {
        let top = *self.order.first()?;
        let last = self.order.pop().expect("Heap is not empty");
        self.position[top as usize] = NONE;
        if top != last {
            self.order[0] = last;
            self.down(0, scores);
        }
        Some(top)
    }
    fn up(&mut self, mut index: usize, scores: &[f64]) {
        let variable = self.order[index];
        while index > 0 {
            let parent = (index - 1) / 2;
            let other = self.order[parent];
            if !Self::before(scores, variable, other) {
                break;
            }
            self.order[index] = other;
            self.position[other as usize] = index as u32;
            index = parent;
        }
        self.order[index] = variable;
        self.position[variable as usize] = index as u32;
    }
    fn down(&mut self, mut index: usize, scores: &[f64]) {
        let variable = self.order[index];
        loop {
            let mut child = 2 * index + 1;
            if child >= self.order.len() {
                break;
            }
            if child + 1 < self.order.len()
                && Self::before(scores, self.order[child + 1], self.order[child])
            {
                child += 1;
            }
            let other = self.order[child];
            if !Self::before(scores, other, variable) {
                break;
            }
            self.order[index] = other;
            self.position[other as usize] = index as u32;
            index = child;
        }
        self.order[index] = variable;
        self.position[variable as usize] = index as u32;
    }
    fn heapify(&mut self, scores: &[f64]) {
        for index in (0..self.order.len() / 2).rev() {
            self.down(index, scores);
        }
    }
}
// VMTF: lista doblemente enlazada con marcas de tiempo; `last` es el frente.
#[derive(Clone)]
struct Queue {
    previous: Vec<u32>,
    next: Vec<u32>,
    stamps: Vec<u64>,
    last: u32,
    search: u32,
    stamp: u64,
}
impl Queue {
    fn new(variables: usize) -> Self {
        Self {
            previous: vec![NONE; variables],
            next: vec![NONE; variables],
            stamps: vec![0; variables],
            last: NONE,
            search: NONE,
            stamp: 0,
        }
    }
    fn push(&mut self, variable: u32) {
        self.previous[variable as usize] = self.last;
        self.next[variable as usize] = NONE;
        if self.last != NONE {
            self.next[self.last as usize] = variable;
        }
        self.last = variable;
        self.stamp += 1;
        self.stamps[variable as usize] = self.stamp;
    }
    fn bump(&mut self, variable: u32) {
        if self.last == variable {
            return;
        }
        let (previous, next) = (self.previous[variable as usize], self.next[variable as usize]);
        if previous != NONE {
            self.next[previous as usize] = next;
        }
        // `variable` no es el frente, así que `next` existe.
        self.previous[next as usize] = previous;
        self.push(variable);
    }
    // Invariante: toda variable con marca posterior a `search` está asignada.
    fn unassign(&mut self, variable: u32) {
        if self.search == NONE
            || self.stamps[variable as usize] > self.stamps[self.search as usize]
        {
            self.search = variable;
        }
    }
}
#[inline]
fn variable_of(literal: u32) -> usize {
    (literal >> 1) as usize
}
#[derive(Clone)]
struct State {
    arena: Vec<u32>,
    watches: Vec<Vec<Watcher>>,
    // Indexado por literal: 1 verdadero, -1 falso, 0 libre.
    values: Vec<i8>,
    levels: Vec<u32>,
    reasons: Vec<u32>,
    phases: Vec<u8>,
    trail: Vec<u32>,
    trail_limits: Vec<usize>,
    head: usize,
    scores: Vec<f64>,
    increment: f64,
    heap: Heap,
    queue: Queue,
    stable: bool,
    seen: Vec<bool>,
    analyzed: Vec<u32>,
    learned: Vec<u32>,
    stack: Vec<u32>,
    level_stamps: Vec<u64>,
    level_stamp: u64,
    fast_lbd: Ema,
    slow_lbd: Ema,
    conflicts: u64,
    next_restart: u64,
    luby: (u64, u64),
    mode_span: u64,
    next_mode: u64,
    cleanup_interval: u64,
    next_cleanup: u64,
    walker: Option<Walker>,
    next_reignition: u64,
    ignited: bool,
    propagated: u64,
    reignition_mark: u64,
    root_unsat: bool,
    stats: SolveStats,
}
impl State {
    fn new(variables: usize, stats: SolveStats) -> Self {
        Self {
            arena: Vec::new(),
            watches: vec![Vec::new(); 2 * variables],
            values: vec![0; 2 * variables],
            levels: vec![0; variables],
            reasons: vec![NO_REASON; variables],
            phases: vec![0; variables],
            trail: Vec::with_capacity(variables),
            trail_limits: Vec::new(),
            head: 0,
            scores: vec![0.0; variables],
            increment: 1.0,
            heap: Heap::new(variables),
            queue: Queue::new(variables),
            stable: false,
            seen: vec![false; variables],
            analyzed: Vec::new(),
            learned: Vec::new(),
            stack: Vec::new(),
            level_stamps: vec![0; variables + 1],
            level_stamp: 0,
            fast_lbd: Ema::new(0.03),
            slow_lbd: Ema::new(1e-5),
            conflicts: 0,
            next_restart: RESTART_INTERVAL,
            luby: (1, 1),
            mode_span: MODE_INTERVAL,
            next_mode: MODE_INTERVAL,
            cleanup_interval: CLEANUP_INTERVAL,
            next_cleanup: CLEANUP_INTERVAL,
            walker: None,
            next_reignition: 0,
            ignited: false,
            propagated: 0,
            reignition_mark: 0,
            root_unsat: false,
            stats,
        }
    }
    #[inline]
    fn value(&self, literal: u32) -> i8 {
        self.values[literal as usize]
    }
    #[inline]
    fn level(&self) -> usize {
        self.trail_limits.len()
    }
    #[inline]
    fn assign(&mut self, literal: u32, reason: u32) {
        let variable = variable_of(literal);
        debug_assert_eq!(self.values[literal as usize], 0);
        self.values[literal as usize] = 1;
        self.values[(literal ^ 1) as usize] = -1;
        self.levels[variable] = self.trail_limits.len() as u32;
        self.reasons[variable] = reason;
        // Phase saving de toda asignación, no solo de las decisiones.
        self.phases[variable] = (literal & 1) as u8;
        self.trail.push(literal);
        self.stats.propagations += usize::from(reason != NO_REASON);
    }
    fn assign_root(&mut self, literal: u32) {
        debug_assert_eq!(self.level(), 0);
        match self.value(literal) {
            0 => self.assign(literal, NO_REASON),
            1 => {}
            _ => self.root_unsat = true,
        }
    }
    // Binaria inline: el watcher lleva el otro literal y la propagación no toca la arena.
    fn watch_binary(&mut self, first: u32, second: u32) {
        self.watches[first as usize].push(Watcher { blocker: second, reference: BINARY | second });
        self.watches[second as usize].push(Watcher { blocker: first, reference: BINARY | first });
    }
    fn add_long(&mut self, literals: &[u32], learned: bool, lbd: u32) -> u32 {
        debug_assert!(literals.len() > 2);
        let reference = u32::try_from(self.arena.len()).ok()
            .filter(|reference| reference & BINARY == 0)
            .expect("Clause arena exceeds 2^31 words");
        self.arena.push(literals.len() as u32);
        self.arena.push(if learned { LEARNED | lbd.min(LBD_MASK) } else { 0 });
        self.arena.extend_from_slice(literals);
        // Los vigilados son siempre las posiciones 0 y 1; cada uno bloquea al otro.
        self.watches[literals[0] as usize].push(Watcher { blocker: literals[1], reference });
        self.watches[literals[1] as usize].push(Watcher { blocker: literals[0], reference });
        reference
    }
    fn add_original(&mut self, clause: &[u32]) {
        match *clause {
            [literal] => self.assign_root(literal),
            [first, second] => self.watch_binary(first, second),
            _ => {
                self.add_long(clause, false, 0);
            }
        }
    }
    // `head` recorre el propio trail hasta su cola (`trail.len()`): no hay cola de unidades.
    fn propagate(&mut self) -> Option<Conflict> {
        while self.head < self.trail.len() {
            let false_literal = self.trail[self.head] ^ 1;
            self.head += 1;
            self.propagated += 1;
            let mut watchers = std::mem::take(&mut self.watches[false_literal as usize]);
            let mut conflict = None;
            let (mut read, mut write) = (0, 0);
            while read < watchers.len() {
                let watcher = watchers[read];
                read += 1;
                let blocker_value = self.values[watcher.blocker as usize];
                if blocker_value > 0 {
                    watchers[write] = watcher;
                    write += 1;
                    continue;
                }
                if watcher.reference & BINARY != 0 {
                    watchers[write] = watcher;
                    write += 1;
                    if blocker_value < 0 {
                        conflict = Some(Conflict::Binary(false_literal, watcher.blocker));
                        break;
                    }
                    self.assign(watcher.blocker, BINARY | false_literal);
                    continue;
                }
                let start = watcher.reference as usize + HEADER;
                // El literal falso pasa a la posición 1; la 0 queda para el implicado.
                if self.arena[start] == false_literal {
                    self.arena.swap(start, start + 1);
                }
                let first = self.arena[start];
                let first_value = self.values[first as usize];
                let moved = Watcher { blocker: first, reference: watcher.reference };
                if first_value > 0 {
                    watchers[write] = moved;
                    write += 1;
                    continue;
                }
                let end = start + self.arena[start - HEADER] as usize;
                let mut replacement = None;
                for position in start + 2..end {
                    if self.values[self.arena[position] as usize] >= 0 {
                        replacement = Some(position);
                        break;
                    }
                }
                if let Some(position) = replacement {
                    let literal = self.arena[position];
                    self.arena[start + 1] = literal;
                    self.arena[position] = false_literal;
                    // `literal` no es falso, luego nunca es la lista que se está recorriendo.
                    self.watches[literal as usize].push(moved);
                    continue;
                }
                watchers[write] = moved;
                write += 1;
                if first_value < 0 {
                    conflict = Some(Conflict::Long(watcher.reference));
                    break;
                }
                self.assign(first, watcher.reference);
            }
            // Compactación in situ: los watchers movidos a otra lista no se copian.
            let remaining = watchers.len() - read;
            watchers.copy_within(read.., write);
            watchers.truncate(write + remaining);
            self.watches[false_literal as usize] = watchers;
            if conflict.is_some() {
                return conflict;
            }
        }
        None
    }
    // En la raíz un literal verdadero satisface la cláusula y uno falso desaparece:
    // lo que queda está libre, así que adjuntarla nunca propaga ni fuerza un backjump.
    fn import(&mut self, clause: &[u32]) {
        debug_assert_eq!(self.level(), 0);
        let mut literals = std::mem::take(&mut self.learned);
        literals.clear();
        let satisfied = clause.iter().any(|&literal| match self.value(literal) {
            1 => true,
            0 => {
                literals.push(literal);
                false
            }
            _ => false,
        });
        if !satisfied {
            self.stats.imported_clauses += 1;
            match *literals {
                [] => self.root_unsat = true,
                [literal] => self.assign_root(literal),
                [first, second] => self.watch_binary(first, second),
                _ => {
                    self.add_long(&literals, true, IMPORT_LBD);
                }
            }
        }
        self.learned = literals;
    }
    fn absorb(&mut self, inbox: &mut Vec<u32>) {
        for clause in clauses_of(inbox) {
            self.import(clause);
        }
        inbox.clear();
    }
    #[inline]
    fn mark(&mut self, literal: u32, pending: &mut usize) {
        let variable = variable_of(literal);
        if self.seen[variable] || self.levels[variable] == 0 {
            return;
        }
        self.seen[variable] = true;
        self.analyzed.push(variable as u32);
        if self.levels[variable] as usize == self.trail_limits.len() {
            *pending += 1;
        } else {
            self.learned.push(literal);
        }
    }
    fn mark_clause(&mut self, reference: u32, skip: usize, pending: &mut usize) {
        let start = reference as usize + HEADER;
        let end = start + self.arena[reference as usize] as usize;
        // Una cláusula que participa en un análisis sobrevive a la siguiente reducción.
        self.arena[reference as usize + 1] |= USED;
        for position in start + skip..end {
            let literal = self.arena[position];
            self.mark(literal, pending);
        }
    }
    fn expand(&mut self, literal: u32, abstract_levels: u32) -> bool {
        let variable = variable_of(literal);
        if self.seen[variable] || self.levels[variable] == 0 {
            return true;
        }
        if self.reasons[variable] == NO_REASON
            || (1_u32 << (self.levels[variable] & 31)) & abstract_levels == 0
        {
            return false;
        }
        self.seen[variable] = true;
        self.analyzed.push(variable as u32);
        self.stack.push(literal);
        true
    }
    // Un literal sobra si su antecedente se deriva, recursivamente, de literales ya presentes.
    fn redundant(&mut self, literal: u32, abstract_levels: u32) -> bool {
        let top = self.analyzed.len();
        self.stack.clear();
        self.stack.push(literal);
        while let Some(literal) = self.stack.pop() {
            let reason = self.reasons[variable_of(literal)];
            let implied = if reason & BINARY != 0 {
                self.expand(reason ^ BINARY, abstract_levels)
            } else {
                let start = reason as usize + HEADER;
                let end = start + self.arena[reason as usize] as usize;
                (start + 1..end).all(|position| {
                    let other = self.arena[position];
                    self.expand(other, abstract_levels)
                })
            };
            if !implied {
                for index in top..self.analyzed.len() {
                    self.seen[self.analyzed[index] as usize] = false;
                }
                self.analyzed.truncate(top);
                return false;
            }
        }
        true
    }
    // 1UIP minimizado en `self.learned`; devuelve (nivel de backjump, LBD).
    fn analyze(&mut self, conflict: Conflict) -> (usize, u32) {
        debug_assert!(self.level() > 0);
        self.learned.clear();
        self.learned.push(0);
        let mut pending = 0;
        match conflict {
            Conflict::Binary(first, second) => {
                self.mark(first, &mut pending);
                self.mark(second, &mut pending);
            }
            Conflict::Long(reference) => self.mark_clause(reference, 0, &mut pending),
        }
        let mut index = self.trail.len();
        let uip = loop {
            debug_assert!(pending > 0, "Conflict has a current-level pivot");
            let literal = loop {
                index -= 1;
                let literal = self.trail[index];
                if self.seen[variable_of(literal)] {
                    break literal;
                }
            };
            pending -= 1;
            if pending == 0 {
                break literal;
            }
            let reason = self.reasons[variable_of(literal)];
            debug_assert_ne!(reason, NO_REASON, "A non-UIP pivot has an antecedent");
            if reason & BINARY != 0 {
                self.mark(reason ^ BINARY, &mut pending);
            } else {
                // El literal implicado ocupa la posición 0 de su antecedente.
                self.mark_clause(reason, 1, &mut pending);
            }
        };
        self.learned[0] = uip ^ 1;
        let bumped = self.analyzed.len();
        let abstract_levels = self.learned[1..].iter().fold(0_u32, |levels, &literal| {
            levels | 1 << (self.levels[variable_of(literal)] & 31)
        });
        let mut kept = 1;
        for index in 1..self.learned.len() {
            let literal = self.learned[index];
            if self.reasons[variable_of(literal)] == NO_REASON
                || !self.redundant(literal, abstract_levels)
            {
                self.learned[kept] = literal;
                kept += 1;
            }
        }
        self.learned.truncate(kept);
        let mut backjump_level = 0;
        if self.learned.len() > 1 {
            let mut highest = 1;
            for index in 2..self.learned.len() {
                if self.levels[variable_of(self.learned[index])]
                    > self.levels[variable_of(self.learned[highest])]
                {
                    highest = index;
                }
            }
            self.learned.swap(1, highest);
            backjump_level = self.levels[variable_of(self.learned[1])] as usize;
        }
        // LBD: niveles de decisión distintos entre los literales aprendidos.
        self.level_stamp += 1;
        let mut lbd = 0;
        for &literal in &self.learned {
            let level = self.levels[variable_of(literal)] as usize;
            if self.level_stamps[level] != self.level_stamp {
                self.level_stamps[level] = self.level_stamp;
                lbd += 1;
            }
        }
        // Se actualizan TODAS las variables vistas en la derivación, no solo las del conflicto.
        let mut analyzed = std::mem::take(&mut self.analyzed);
        if self.stable {
            for &variable in &analyzed[..bumped] {
                self.bump_score(variable);
            }
            self.increment /= DECAY;
            if self.increment > MAX_ACTIVITY {
                self.rescale();
            }
        } else {
            // Al frente conservando el orden relativo que ya tenían.
            analyzed[..bumped].sort_unstable_by_key(|&variable| self.queue.stamps[variable as usize]);
            for &variable in &analyzed[..bumped] {
                self.queue.bump(variable);
            }
        }
        for &variable in &analyzed {
            self.seen[variable as usize] = false;
        }
        analyzed.clear();
        self.analyzed = analyzed;
        (backjump_level, lbd)
    }
    // EVSIDS: crece el incremento en vez de decaer todas las puntuaciones.
    fn bump_score(&mut self, variable: u32) {
        self.scores[variable as usize] += self.increment;
        if self.scores[variable as usize] > MAX_ACTIVITY {
            self.rescale();
        }
        if self.heap.contains(variable) {
            self.heap.increase(variable, &self.scores);
        }
    }
    fn rescale(&mut self) {
        for score in &mut self.scores {
            *score /= MAX_ACTIVITY;
        }
        self.increment /= MAX_ACTIVITY;
        // El redondeo puede crear empates: se restaura el orden completo.
        self.heap.heapify(&self.scores);
    }
    fn learn(&mut self, lbd: u32) {
        let learned = std::mem::take(&mut self.learned);
        let first = learned[0];
        match learned.len() {
            1 => self.assign_root(first),
            2 => {
                self.watch_binary(first, learned[1]);
                self.assign(first, BINARY | learned[1]);
            }
            _ => {
                let reference = self.add_long(&learned, true, lbd);
                self.assign(first, reference);
            }
        }
        self.learned = learned;
        self.stats.learned_clauses += 1;
        self.conflicts += 1;
        self.fast_lbd.update(f64::from(lbd));
        self.slow_lbd.update(f64::from(lbd));
    }
    fn backjump(&mut self, level: usize) {
        if self.level() <= level {
            return;
        }
        let mark = self.trail_limits[level];
        for index in (mark..self.trail.len()).rev() {
            let literal = self.trail[index];
            self.values[literal as usize] = 0;
            self.values[(literal ^ 1) as usize] = 0;
            self.queue.unassign(literal >> 1);
            if self.stable {
                self.heap.insert(literal >> 1, &self.scores);
            }
        }
        self.trail.truncate(mark);
        self.trail_limits.truncate(level);
        self.head = mark;
    }
    fn restart_due(&self) -> bool {
        self.conflicts >= self.next_mode
            || (self.conflicts >= self.next_restart
                && (self.stable || self.fast_lbd.value > RESTART_MARGIN * self.slow_lbd.value))
    }
    // Modo enfocado: VMTF y restarts tipo Glucose. Modo estable: EVSIDS y restarts Luby.
    fn restart(&mut self, base: usize) {
        self.backjump(base);
        self.stats.restarts += 1;
        if self.conflicts >= self.next_mode {
            self.stable = !self.stable;
            self.mode_span = self.mode_span.saturating_mul(2);
            self.next_mode = self.conflicts.saturating_add(self.mode_span);
            if self.stable {
                for variable in 0..self.levels.len() as u32 {
                    if self.values[(variable << 1) as usize] == 0 {
                        self.heap.insert(variable, &self.scores);
                    }
                }
            }
        }
        self.next_restart = self.conflicts + if self.stable {
            let (u, v) = self.luby;
            self.luby = if u & u.wrapping_neg() == v { (u + 1, 1) } else { (u, 2 * v) };
            LUBY_UNIT * v
        } else {
            RESTART_INTERVAL
        };
    }
    fn decide(&mut self) -> Option<u32> {
        let variable = if self.stable {
            loop {
                let variable = self.heap.pop(&self.scores)?;
                if self.values[(variable << 1) as usize] == 0 {
                    break variable;
                }
            }
        } else {
            let mut variable = self.queue.search;
            while variable != NONE && self.values[(variable << 1) as usize] != 0 {
                variable = self.queue.previous[variable as usize];
            }
            self.queue.search = variable;
            if variable == NONE {
                return None;
            }
            variable
        };
        Some((variable << 1) | u32::from(self.phases[variable as usize]))
    }
    // El oráculo solo lee las unidades de la raíz y solo escribe fases:
    // nunca toca valores, antecedentes, watches ni el trail.
    fn reignite(&mut self, configuration: ReignitionConfig, cancel: Option<&AtomicBool>) -> bool {
        let Some(mut walker) = self.walker.take() else { return false; };
        let started = Instant::now();
        let root = self.trail_limits.first().copied().unwrap_or(self.trail.len());
        if walker.synced < root {
            for &literal in &self.trail[walker.synced..root] {
                walker.fix(literal);
            }
            walker.synced = root;
            walker.settle();
        }
        let mut budget = configuration.oracle.flips_per_variable.saturating_mul(self.phases.len());
        if self.ignited {
            let earned = (self.propagated - self.reignition_mark) / ORACLE_EFFORT;
            budget = budget.min(usize::try_from(earned).unwrap_or(usize::MAX));
        }
        self.reignition_mark = self.propagated;
        let flips = walker.walk(budget, cancel);
        // Transferencia: la mejor asignación del oráculo sobrescribe las polaridades guardadas.
        for (phase, &value) in self.phases.iter_mut().zip(&walker.best_values) {
            *phase = u8::from(!value);
        }
        let solved = walker.best == 0;
        self.walker = Some(walker);
        self.ignited = true;
        self.next_reignition = self.conflicts.saturating_add(configuration.interval as u64);
        self.stats.reignitions += 1;
        self.stats.reignition_flips += flips;
        self.stats.reignition_ms += started.elapsed().as_secs_f64() * 1000.0;
        solved
    }
    fn oracle_model(&self, variables: &[u32]) -> Vec<i32> {
        let walker = self.walker.as_ref().expect("Only the oracle reports its own model");
        signed_model(variables, walker.best_values.iter().copied())
    }
    // Candidatas: aprendidas con LBD > 2 que no son razón ni se usaron desde la última reducción.
    fn reduce_database(&mut self, mut proof: Option<(&mut (dyn ProofLog + '_), &[u32])>) {
        debug_assert_eq!(self.head, self.trail.len());
        // Intervalo aritmético: la base retenida crece con la raíz del número de conflictos.
        self.cleanup_interval += CLEANUP_INTERVAL;
        self.next_cleanup = self.conflicts + self.cleanup_interval;
        // Las razones vigentes son largas sin el bit BINARY (NO_REASON lo tiene).
        for &literal in &self.trail {
            let reason = self.reasons[variable_of(literal)];
            if reason & BINARY == 0 {
                self.arena[reason as usize + 1] |= LOCKED;
            }
        }
        let mut candidates = Vec::new();
        let mut reference = 0;
        while reference < self.arena.len() {
            let meta = self.arena[reference + 1];
            if meta & (LEARNED | LOCKED | USED) == LEARNED && meta & LBD_MASK > 2 {
                candidates.push(reference);
            }
            self.arena[reference + 1] = meta & !(LOCKED | USED);
            reference += HEADER + self.arena[reference] as usize;
        }
        if candidates.is_empty() {
            return;
        }
        // Se sacrifican primero el mayor LBD y, a igualdad, la cláusula más larga.
        candidates.sort_unstable_by(|&a, &b| {
            (self.arena[b + 1] & LBD_MASK).cmp(&(self.arena[a + 1] & LBD_MASK))
                .then_with(|| self.arena[b].cmp(&self.arena[a]))
                .then_with(|| a.cmp(&b))
        });
        let remove_count = (candidates.len() / 2).max(1);
        let mut line: Vec<i32> = Vec::new();
        for &reference in &candidates[..remove_count] {
            self.arena[reference + 1] |= GARBAGE;
            // Una razón vigente nunca es candidata: lo borrado no sostiene ninguna unidad.
            if let Some((log, variables)) = proof.as_mut() {
                let start = reference + HEADER;
                external(variables, &self.arena[start..start + self.arena[reference] as usize], &mut line);
                log.delete(&line);
            }
        }
        // La arena vieja guarda la nueva referencia de cada superviviente en su palabra de flags.
        let mut old = std::mem::take(&mut self.arena);
        let mut arena = Vec::with_capacity(old.len());
        let mut reference = 0;
        while reference < old.len() {
            let next = reference + HEADER + old[reference] as usize;
            if old[reference + 1] & GARBAGE == 0 {
                let moved = arena.len() as u32;
                arena.extend_from_slice(&old[reference..next]);
                old[reference + 1] = moved;
            }
            reference = next;
        }
        for &literal in &self.trail {
            let variable = variable_of(literal);
            let reason = self.reasons[variable];
            if reason & BINARY == 0 {
                self.reasons[variable] = old[reason as usize + 1];
            }
        }
        for watchers in &mut self.watches {
            watchers.retain(|watcher| watcher.reference & BINARY != 0);
        }
        let mut reference = 0;
        while reference < arena.len() {
            let (first, second) = (arena[reference + HEADER], arena[reference + HEADER + 1]);
            self.watches[first as usize].push(Watcher { blocker: second, reference: reference as u32 });
            self.watches[second as usize].push(Watcher { blocker: first, reference: reference as u32 });
            reference += HEADER + arena[reference] as usize;
        }
        self.arena = arena;
        self.stats.deleted_clauses += remove_count;
        self.stats.database_reductions += 1;
    }
}
fn active(config: Option<ReignitionConfig>) -> Option<ReignitionConfig> {
    config.filter(|configuration| {
        configuration.interval != 0 && configuration.oracle.flips_per_variable != 0
    })
}
fn prepare_formula(formula: &[Vec<i32>], oracle: Option<OracleConfig>) -> Option<(State, Vec<u32>)> {
    if formula.iter().any(Vec::is_empty) {
        return None;
    }
    let mut variables: Vec<u32> = formula.iter().flatten().map(|&literal| {
        assert!(literal != 0 && literal != i32::MIN, "Invalid SAT literal");
        literal.unsigned_abs()
    }).collect();
    variables.sort_unstable();
    variables.dedup();
    // Literales y referencias comparten 31 bits; NO_REASON queda fuera de ambos rangos.
    assert!(variables.len() < 1 << 30, "Too many variables");
    let mut state = State::new(variables.len(), SolveStats::default());
    let mut packed: Vec<u32> = Vec::new();
    // Copia plana para el oráculo; las unidades le llegan después por el trail raíz.
    let mut walk_literals = Vec::new();
    let mut walk_offsets = vec![0_u32];
    for clause in formula {
        packed.clear();
        packed.extend(clause.iter().map(|&literal| {
            let variable = variables.binary_search(&literal.unsigned_abs())
                .expect("Variable was indexed") as u32;
            (variable << 1) | u32::from(literal < 0)
        }));
        packed.sort_unstable();
        packed.dedup();
        // Tras ordenar, x y ¬x quedan adyacentes: la tautología no aporta nada.
        if packed.windows(2).any(|pair| pair[0] ^ 1 == pair[1]) {
            continue;
        }
        for &literal in &packed {
            state.scores[variable_of(literal)] += 1.0;
        }
        state.add_original(&packed);
        if oracle.is_some() && packed.len() > 1 {
            walk_literals.extend_from_slice(&packed);
            walk_offsets.push(u32::try_from(walk_literals.len())
                .expect("Oracle arena exceeds 2^32 literals"));
        }
    }
    if state.root_unsat {
        state.stats.conflicts = 1;
    } else if let Some(oracle) = oracle {
        state.walker = Some(Walker::new(variables.len(), walk_literals, walk_offsets, oracle.seed));
    }
    // Orden VMTF inicial: al frente la mayor puntuación y, a igualdad, el menor índice.
    let mut order: Vec<u32> = (0..variables.len() as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        state.scores[a as usize].total_cmp(&state.scores[b as usize]).then_with(|| b.cmp(&a))
    });
    for variable in order {
        state.queue.push(variable);
    }
    state.queue.search = state.queue.last;
    Some((state, variables))
}
// Literales internos a identificadores DIMACS, para las líneas de la traza.
fn external(variables: &[u32], clause: &[u32], line: &mut Vec<i32>) {
    line.clear();
    line.extend(clause.iter().map(|&literal| {
        let id = variables[(literal >> 1) as usize] as i32;
        if literal & 1 == 0 { id } else { -id }
    }));
}
pub fn solve_cdcl(formula: &[Vec<i32>]) -> (Option<Vec<i32>>, SolveStats) {
    solve_cdcl_inner(formula, None, None)
}
// Igual que `solve_cdcl_with_reignition`, registrando cada cláusula aprendida o borrada.
// Toda cláusula aprendida es RUP; un resultado UNSAT termina con la cláusula vacía.
pub fn solve_cdcl_with_proof(
    formula: &[Vec<i32>],
    config: ReignitionConfig,
    proof: &mut dyn ProofLog,
) -> (Option<Vec<i32>>, SolveStats) {
    solve_cdcl_inner(formula, Some(config), Some(proof))
}
pub fn solve_cdcl_with_reignition(
    formula: &[Vec<i32>],
    config: ReignitionConfig,
) -> (Option<Vec<i32>>, SolveStats) {
    solve_cdcl_inner(formula, Some(config), None)
}
fn solve_cdcl_inner(
    formula: &[Vec<i32>],
    config: Option<ReignitionConfig>,
    proof: Option<&mut dyn ProofLog>,
) -> (Option<Vec<i32>>, SolveStats) {
    let feedback = active(config);
    let prepared = prepare_formula(formula, feedback.map(|configuration| configuration.oracle));
    let Some((mut state, variables)) = prepared else {
        return (None, SolveStats { conflicts: 1, ..SolveStats::default() });
    };
    let control = SearchControl { cube: &[], cancel: None, sharing: None, proof };
    let (outcome, stats) = run_state(&mut state, &variables, feedback, control);
    match outcome {
        SearchOutcome::Sat(model) => (Some(model), stats),
        SearchOutcome::Unsat => (None, stats),
        SearchOutcome::Cancelled => unreachable!("Serial execution cannot be cancelled"),
    }
}
fn signed_model(variables: &[u32], values: impl Iterator<Item = bool>) -> Vec<i32> {
    variables.iter().zip(values).map(|(&id, value)| {
        if value { id as i32 } else { -(id as i32) }
    }).collect()
}
fn extract_model(state: &State, variables: &[u32]) -> Vec<i32> {
    signed_model(variables, state.values.iter().step_by(2).map(|&value| value == 1))
}
fn run_state(
    state: &mut State,
    variables: &[u32],
    feedback: Option<ReignitionConfig>,
    mut control: SearchControl<'_, '_>,
) -> (SearchOutcome, SolveStats) {
    let outcome = search(state, variables, feedback, &mut control);
    if let Some(link) = control.sharing {
        // Lo aprendido en este cubo se publica entero antes de soltar el estado.
        link.flush(&mut state.stats, true);
    }
    (outcome, state.stats)
}
fn search(
    state: &mut State,
    variables: &[u32],
    feedback: Option<ReignitionConfig>,
    control: &mut SearchControl<'_, '_>,
) -> SearchOutcome {
    // La cancelación se consulta una vez por conflicto o decisión, nunca dentro del BCP.
    let cancel = control.cancel;
    let cancelled = || cancel.is_some_and(|flag| flag.load(Ordering::Acquire));
    if cancelled() {
        return SearchOutcome::Cancelled;
    }
    // El certificado raiz evita reprocesar una cola contradictoria y cubos posteriores.
    if state.root_unsat {
        // Dos unidades opuestas de la fórmula: la cláusula vacía es RUP.
        if let Some(proof) = control.proof.as_deref_mut() {
            proof.add(&[]);
        }
        return SearchOutcome::Unsat;
    }
    let mut line: Vec<i32> = Vec::new();
    if let Some(link) = control.sharing.as_deref_mut() {
        // Un cubo empieza en el nivel 0: mismo punto de sincronización que un restart.
        if state.level() == 0 {
            link.exchange(state.conflicts, &mut state.stats);
            state.absorb(&mut link.inbox);
        }
    }
    loop {
        if cancelled() {
            return SearchOutcome::Cancelled;
        }
        if state.root_unsat {
            // Una importación contradice la raíz: consecuencia global, UNSAT.
            state.stats.conflicts += 1;
            return SearchOutcome::Unsat;
        }
        if let Some(conflict) = state.propagate() {
            state.stats.conflicts += 1;
            let level = state.level();
            if level == 0 {
                // No se publica la cláusula vacía ni se confunde cancelación con UNSAT.
                state.root_unsat = true;
                if let Some(proof) = control.proof.as_deref_mut() {
                    proof.add(&[]);
                }
                return SearchOutcome::Unsat;
            }
            let (backjump_level, lbd) = state.analyze(conflict);
            if let Some(proof) = control.proof.as_deref_mut() {
                external(variables, &state.learned, &mut line);
                proof.add(&line);
            }
            if lbd <= EXPORT_LBD {
                if let Some(link) = control.sharing.as_deref_mut() {
                    // Los cubos son decisiones: toda cláusula aprendida es consecuencia global.
                    link.export(&state.learned, &mut state.stats);
                }
            }
            state.stats.backjumps += 1;
            state.stats.nonchronological_backjumps += usize::from(level > backjump_level + 1);
            state.backjump(backjump_level);
            state.learn(lbd);
            continue;
        }
        if state.conflicts >= state.next_cleanup {
            state.reduce_database(control.proof.as_deref_mut().map(|proof| (proof, variables)));
        }
        if let Some(configuration) = feedback {
            // Ignición inicial: primer fixpoint de la búsqueda, todavía en el nivel 0.
            if !state.ignited && state.reignite(configuration, cancel) {
                return SearchOutcome::Sat(state.oracle_model(variables));
            }
        }
        if state.level() < control.cube.len() {
            let literal = control.cube[state.level()];
            let value = state.value(literal);
            if value == -1 { return SearchOutcome::Unsat; }
            // Un nivel por supuesto; niveles vacíos solo se crean en un fixpoint.
            state.trail_limits.push(state.trail.len());
            if value == 0 {
                state.stats.decisions += 1;
                state.stats.cube_decisions += 1;
                state.assign(literal, NO_REASON);
            }
            // Cubos como DECISIONES: sus literales permanecen en las cláusulas 1UIP.
            continue;
        }
        if state.restart_due() {
            // El restart conserva los supuestos del cubo salvo que haya algo que importar:
            // importar exige el nivel 0 y allí nunca fuerza un backjump.
            let mut base = control.cube.len();
            if let Some(link) = control.sharing.as_deref_mut() {
                if state.conflicts - link.last_sync >= link.interval {
                    link.exchange(state.conflicts, &mut state.stats);
                    if !link.inbox.is_empty() {
                        base = 0;
                    }
                }
            }
            state.restart(base);
            if let Some(link) = control.sharing.as_deref_mut() {
                if !link.inbox.is_empty() {
                    state.absorb(&mut link.inbox);
                }
            }
            if let Some(configuration) = feedback {
                // Re-ignición: solo aquí, con el trail recién limpiado por el restart.
                if state.conflicts >= state.next_reignition && !state.root_unsat
                    && state.reignite(configuration, cancel)
                {
                    return SearchOutcome::Sat(state.oracle_model(variables));
                }
            }
            continue;
        }
        let Some(literal) = state.decide() else {
            return SearchOutcome::Sat(extract_model(state, variables));
        };
        state.trail_limits.push(state.trail.len());
        state.stats.decisions += 1;
        state.assign(literal, NO_REASON);
    }
}
fn make_cubes(base: &State, config: SwarmConfig) -> Vec<Vec<u32>> {
    let target = config.workers.max(1).saturating_mul(config.cubes_per_worker.max(1)).clamp(1, 1024);
    let mut candidates: Vec<usize> = (0..base.levels.len())
        .filter(|&variable| base.values[variable << 1] == 0).collect();
    candidates.sort_unstable_by(|&a, &b| {
        base.scores[b].total_cmp(&base.scores[a]).then_with(|| a.cmp(&b))
    });
    let mut depth = 0;
    while depth < candidates.len().min(10) && (1 << depth) < target { depth += 1; }
    candidates.truncate(depth);
    // Todas las combinaciones: las fases ordenan los cubos y nunca descartan ninguno.
    (0..1usize << depth).map(|mask| candidates.iter().enumerate().map(|(index, &variable)| {
        ((variable as u32) << 1) | (u32::from(base.phases[variable]) ^ ((mask >> index) & 1) as u32)
    }).collect()).collect()
}
#[derive(Default)]
struct WorkerReport {
    started: usize,
    unsat: usize,
    cancelled: usize,
    model: Option<Vec<i32>>,
    total: SolveStats,
}
fn swarm_worker(
    base: &State,
    variables: &[u32],
    cubes: &[Vec<u32>],
    next: &AtomicUsize,
    cancel: &AtomicBool,
    pool: &CloudPool,
    config: SwarmConfig,
    owner: u32,
) -> WorkerReport {
    let mut report = WorkerReport::default();
    let mut state = base.clone();
    // Cada worker continúa la caminata del estado base con su propio azar.
    if let Some(walker) = state.walker.as_mut() {
        walker.reseed(u64::from(owner) + 1);
    }
    let feedback = active(Some(config.reignition));
    // El contexto de intercambio vive lo que el worker, no lo que un cubo.
    let mut link = (config.share_enabled && config.share_capacity != 0)
        .then(|| Sharing::new(pool, owner, config.share_interval.max(1) as u64));
    loop {
        if cancel.load(Ordering::Acquire) { break; }
        let index = next.fetch_add(1, Ordering::Relaxed);
        let Some(cube) = cubes.get(index) else { break; };
        report.started += 1;
        if !state.root_unsat {
            // Solo se llega al siguiente cubo tras CubeUNSAT en un fixpoint.
            debug_assert_eq!(state.head, state.trail.len());
            state.backjump(0);
        }
        // Clausulas, unidades raiz, fases, actividades y oraculo sobreviven al cubo.
        state.stats = SolveStats::default();
        let control = SearchControl { cube, cancel: Some(cancel), sharing: link.as_mut(), proof: None };
        let (outcome, stats) = run_state(&mut state, variables, feedback, control);
        report.total.add_search(stats);
        match outcome {
            SearchOutcome::Sat(model) => {
                if cancel.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                    report.model = Some(model);
                } else { report.cancelled += 1; }
                break;
            }
            SearchOutcome::Unsat => report.unsat += 1,
            SearchOutcome::Cancelled => { report.cancelled += 1; break; }
        }
    }
    report
}
pub fn solve_swarm(formula: &[Vec<i32>], config: SwarmConfig) -> (Option<Vec<i32>>, SwarmStats) {
    let mut stats = SwarmStats::default();
    let feedback = active(Some(config.reignition));
    let prepared = prepare_formula(formula, feedback.map(|configuration| configuration.oracle));
    let Some((mut base, variables)) = prepared else {
        stats.total.conflicts = 1;
        return (None, stats);
    };
    if !base.root_unsat && base.propagate().is_some() {
        base.root_unsat = true;
        base.stats.conflicts += 1;
    }
    if base.root_unsat {
        stats.total = base.stats;
        return (None, stats);
    }
    if base.trail.len() == variables.len() {
        stats.total = base.stats;
        return (Some(extract_model(&base, &variables)), stats);
    }
    if let Some(configuration) = feedback {
        // Ignición única antes de cubicar: las fases del oráculo ordenan los cubos.
        if base.reignite(configuration, None) {
            stats.total = base.stats;
            return (Some(base.oracle_model(&variables)), stats);
        }
    }
    stats.total = base.stats;
    let cubes = make_cubes(&base, config);
    stats.cubes_generated = cubes.len();
    let workers = config.workers.max(1).min(cubes.len());
    let next = AtomicUsize::new(0);
    let cancel = AtomicBool::new(false);
    let pool = CloudPool::new(config.share_capacity);
    let (spawned, reports) = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for owner in 0..workers {
            if cancel.load(Ordering::Acquire) { break; }
            let (base, variables, cubes, next, cancel, pool) =
                (&base, variables.as_slice(), cubes.as_slice(), &next, &cancel, &pool);
            let worker = move || swarm_worker(base, variables, cubes,
                next, cancel, pool, config, owner as u32);
            match std::thread::Builder::new().name(format!("swarm-{owner}"))
                .spawn_scoped(scope, worker) {
                Ok(handle) => handles.push(handle),
                Err(_) => break,
            }
        }
        if handles.is_empty() {
            return (0, vec![swarm_worker(&base, &variables, &cubes,
                &next, &cancel, &pool, config, 0)]);
        }
        let spawned = handles.len();
        let reports = handles.into_iter().map(|handle| handle.join().expect("SAT worker panicked"))
            .collect::<Vec<_>>();
        (spawned, reports)
    });
    stats.workers = spawned;
    let mut winner = None;
    for report in reports {
        stats.cubes_started += report.started;
        stats.cubes_unsat += report.unsat;
        stats.cubes_cancelled += report.cancelled;
        stats.total.add_search(report.total);
        if report.model.is_some() { winner = report.model; }
    }
    stats.shared_clauses = pool.snapshot(&variables);
    if winner.is_none() {
        assert_eq!(stats.cubes_unsat, stats.cubes_generated, "Incomplete cube coverage cannot prove UNSAT");
    }
    (winner, stats)
}

#[cfg(test)]
mod deterministic_import_tests {
    use super::*;

    #[test]
    fn published_consequence_is_imported_and_promoted_to_root() {
        // (x1 OR x2) AND (x1 OR !x2) entails x1 without a syntactic unit.
        let formula = vec![vec![1, 2], vec![1, -2]];
        let (mut state, variables) = prepare_formula(&formula, None).unwrap();
        let positive_x1 = 0;

        let pool = CloudPool::new(8);
        let mut peer = Sharing::new(&pool, 1, 1);
        let mut peer_stats = SolveStats::default();
        peer.export(&[positive_x1], &mut peer_stats);
        peer.flush(&mut peer_stats, true);
        assert_eq!(peer_stats.exported_clauses, 1);
        let snapshot = pool.snapshot(&variables);
        assert_eq!(snapshot, vec![vec![1]]);
        for bits in 0..4 {
            let literal_true = |literal: i32| {
                (bits & (1 << (literal.unsigned_abs() - 1)) != 0) == (literal > 0)
            };
            if formula.iter().all(|clause| clause.iter().copied().any(literal_true)) {
                assert!(snapshot[0].iter().copied().any(literal_true));
            }
        }

        let mut link = Sharing::new(&pool, 0, 1);
        let control = SearchControl { cube: &[], cancel: None, sharing: Some(&mut link), proof: None };
        let (outcome, stats) = run_state(&mut state, &variables, None, control);
        let model = match outcome {
            SearchOutcome::Sat(model) => model,
            _ => panic!("Importing a valid consequence must preserve SAT"),
        };
        assert!(model.contains(&1));
        assert_eq!(stats.imported_clauses, 1);
        assert_eq!(stats.import_backjumps, 0);
        assert_eq!(state.value(positive_x1), 1);
        assert_eq!(state.levels[0], 0);
        // La unidad importada vive en el trail raíz: no necesita cláusula antecedente.
        assert_eq!(state.reasons[0], NO_REASON);

        // Siguiente cubo del mismo worker: cursor y filtro sobreviven, nada se reimporta.
        state.backjump(0);
        state.stats = SolveStats::default();
        let control = SearchControl { cube: &[], cancel: None, sharing: Some(&mut link), proof: None };
        let (_, stats) = run_state(&mut state, &variables, None, control);
        assert_eq!(stats.imported_clauses, 0);
        // Un lote nunca vuelve a su autor.
        peer.exchange(0, &mut peer_stats);
        assert!(peer.inbox.is_empty());
    }

    #[test]
    fn ring_keeps_the_newest_clauses_and_lagging_readers_skip_ahead() {
        let pool = CloudPool::new(2);
        let mut writer = Sharing::new(&pool, 0, 1);
        let mut stats = SolveStats::default();
        // Dos cláusulas llenan el anillo y salen como un lote; la tercera lo expulsa entero.
        for literal in [0, 2, 4] {
            writer.export(&[literal, literal + 11], &mut stats);
        }
        writer.flush(&mut stats, true);
        assert_eq!(stats.exported_clauses, 3);
        let variables: Vec<u32> = (1..=10).collect();
        assert_eq!(pool.snapshot(&variables), vec![vec![3, -8]]);

        let mut reader = Sharing::new(&pool, 1, 1);
        reader.exchange(0, &mut stats);
        let seen: Vec<Vec<u32>> = clauses_of(&reader.inbox).map(<[u32]>::to_vec).collect();
        assert_eq!(seen, vec![vec![4, 15]]);
        // El filtro reconoce la misma cláusula en cualquier orden de literales.
        assert!(!reader.remember(&[15, 4]));
    }

    fn violations(walker: &Walker, values: &[bool]) -> usize {
        (0..walker.offsets.len() - 1).filter(|&clause| {
            let range = walker.offsets[clause] as usize..walker.offsets[clause + 1] as usize;
            !walker.literals[range].iter().any(|&literal| values[variable_of(literal)] == (literal & 1 == 0))
        }).count()
    }

    #[test]
    fn walker_counters_survive_fixing_and_walking() {
        let variables = 12_usize;
        let mut seed = 0x9e37_79b9_u64;
        let mut literals = Vec::new();
        let mut offsets = vec![0_u32];
        for _ in 0..60 {
            let mut clause: Vec<u32> = (0..3).map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed % (2 * variables as u64)) as u32
            }).collect();
            clause.sort_unstable();
            clause.dedup();
            if clause.len() < 2 || clause.windows(2).any(|pair| pair[0] ^ 1 == pair[1]) {
                continue;
            }
            literals.extend_from_slice(&clause);
            offsets.push(literals.len() as u32);
        }
        let mut walker = Walker::new(variables, literals, offsets, 7);
        // x3 verdadero y x6 falso quedan fijados como unidades de la raíz.
        walker.fix(4);
        walker.fix(11);
        walker.settle();
        let start = walker.best;
        assert!(walker.walk(400, None) <= 400);
        assert!(walker.best <= start);
        assert!(walker.values[2] && !walker.values[5]);
        assert!(walker.best_values[2] && !walker.best_values[5]);
        // Recuento desde cero: verdaderos, críticos, roturas y lista de violadas.
        let mut breaks = vec![0_u32; variables];
        let mut violated = Vec::new();
        for clause in 0..walker.offsets.len() - 1 {
            let range = walker.offsets[clause] as usize..walker.offsets[clause + 1] as usize;
            let truths: Vec<u32> = walker.literals[range].iter().copied()
                .filter(|&literal| walker.values[variable_of(literal)] == (literal & 1 == 0)).collect();
            assert_eq!(walker.satisfied[clause] as usize, truths.len());
            assert_eq!(walker.critical[clause], truths.iter().fold(0, |xor, &literal| xor ^ literal));
            match truths[..] {
                [] => violated.push(clause as u32),
                [only] => breaks[variable_of(only)] += 1,
                _ => {}
            }
        }
        assert_eq!(walker.breaks, breaks);
        let mut listed = walker.unsat.clone();
        listed.sort_unstable();
        assert_eq!(listed, violated);
        // La asignación guardada es exactamente tan buena como dice `best`.
        assert_eq!(violations(&walker, &walker.best_values), walker.best);
    }

    #[test]
    fn oracle_model_ends_the_search_before_any_decision() {
        let formula: Vec<Vec<i32>> = (1..=30).step_by(3).map(|v| vec![v, v + 1, v + 2]).collect();
        let config = ReignitionConfig {
            interval: 1,
            oracle: OracleConfig { flips_per_variable: 8, seed: 5 },
        };
        let (model, stats) = solve_cdcl_with_reignition(&formula, config);
        let model = model.expect("Disjoint clauses are satisfiable");
        assert!(formula.iter().all(|clause| clause.iter().any(|literal| model.contains(literal))));
        assert_eq!((stats.decisions, stats.conflicts, stats.reignitions), (0, 0, 1));
        assert!(stats.reignition_flips <= 8 * 30);
    }
}

}

pub use cdcl::{OracleConfig, SolveStats};
pub type Clause = Vec<i32>;
pub type Formula = Vec<Clause>;

#[derive(Clone, Copy, Debug)]
pub struct HybridConfig {
    pub oracle: OracleConfig,
    // Cero desactiva el oráculo sin cambiar el motor exacto.
    pub reignition_interval: usize,
    // Ruptura estática de simetrías antes del oráculo y del CDCL.
    pub symmetry: bool,
    // Extracción de paridad y eliminación gaussiana antes del oráculo y del CDCL.
    pub parity: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        let reignition = cdcl::ReignitionConfig::default();
        Self {
            oracle: reignition.oracle, reignition_interval: reignition.interval,
            symmetry: true, parity: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SymmetryStats {
    pub generators: usize,
    pub clauses: usize,
    pub ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ParityStats {
    pub equations: usize,
    pub rank: usize,
    pub units: usize,
    pub equivalences: usize,
    pub clauses: usize,
    pub auxiliaries: usize,
    pub refuted: bool,
    pub ms: f64,
}

#[derive(Clone, Debug)]
pub struct HybridReport {
    pub model: Option<Vec<i32>>,
    pub symmetry: SymmetryStats,
    pub parity: ParityStats,
    pub cdcl: SolveStats,
    pub cdcl_ms: f64,
}

#[derive(Clone, Debug)]
pub struct ParallelHybridReport {
    pub model: Option<Vec<i32>>,
    pub symmetry: SymmetryStats,
    pub parity: ParityStats,
    pub swarm: cdcl::SwarmStats,
    pub conquer_ms: f64,
}

// Capas de inprocessing. La fórmula que ven el oráculo y el CDCL lleva las consecuencias
// lineales y los predicados de ruptura; el modelo se sigue verificando contra la original.
// Ambas capas leen la fórmula original: la paridad solo añade consecuencias suyas, así que
// no altera las simetrías ni la validez de sus predicados.
fn preprocess<'f>(
    formula: &'f [Clause], config: HybridConfig, mut proof: Option<&mut (dyn proof::ProofLog + '_)>,
) -> (std::borrow::Cow<'f, [Clause]>, SymmetryStats, ParityStats) {
    use std::borrow::Cow;
    use std::time::Instant;
    let mut added: Vec<Clause> = Vec::new();
    let mut symmetry_stats = SymmetryStats::default();
    let mut break_symmetry = |added: &mut Vec<Clause>, proof: Option<&mut (dyn proof::ProofLog + '_)>| {
        let start = Instant::now();
        let breaking = symmetry::break_symmetries_with_proof(formula, proof);
        symmetry_stats = SymmetryStats {
            generators: breaking.generators.len(),
            clauses: breaking.clauses.len(),
            ms: start.elapsed().as_secs_f64() * 1000.0,
        };
        added.extend(breaking.clauses);
    };
    // Con una traza que admite dominancia la simetría se certifica primero, sobre la
    // fórmula intacta: lo que derive después la paridad no tiene por qué ser simétrico.
    // Con una traza que no la admite, no se rompe ninguna simetría.
    let certified = proof.is_some();
    if config.symmetry && certified {
        break_symmetry(&mut added, proof.as_deref_mut());
    }
    let mut parity_stats = ParityStats::default();
    if config.parity {
        let start = Instant::now();
        let reduction = parity::reduce_parity_with_proof(formula, proof.as_deref_mut());
        parity_stats = ParityStats {
            equations: reduction.equations,
            rank: reduction.rank,
            units: reduction.units,
            equivalences: reduction.equivalences,
            clauses: reduction.clauses.len(),
            auxiliaries: reduction.auxiliaries,
            refuted: reduction.refuted,
            ms: start.elapsed().as_secs_f64() * 1000.0,
        };
        added.extend(reduction.clauses);
    }
    // Sin traza, con 0 = 1 ya demostrado no queda nada que romper.
    if config.symmetry && !certified && !parity_stats.refuted {
        break_symmetry(&mut added, None);
    }
    if added.is_empty() {
        return (Cow::Borrowed(formula), symmetry_stats, parity_stats);
    }
    let mut extended = formula.to_vec();
    extended.extend(added);
    (Cow::Owned(extended), symmetry_stats, parity_stats)
}

// Las variables auxiliares de la paridad quedan por encima de todo identificador original.
fn restrict(model: Option<Vec<i32>>, formula: &[Clause], parity: &ParityStats) -> Option<Vec<i32>> {
    if parity.auxiliaries == 0 {
        return model;
    }
    let last = formula.iter().flatten().map(|literal| literal.unsigned_abs()).max().unwrap_or(0);
    model.map(|mut model| {
        model.retain(|literal| literal.unsigned_abs() <= last);
        model
    })
}

pub fn solve_hybrid_parallel(
    formula: &[Clause], config: HybridConfig, mut swarm: cdcl::SwarmConfig,
) -> ParallelHybridReport {
    use std::time::Instant;
    swarm.reignition = cdcl::ReignitionConfig {
        interval: config.reignition_interval, oracle: config.oracle,
    };
    let (extended, symmetry, parity) = preprocess(formula, config, None);
    let start = Instant::now();
    let (model, swarm) = cdcl::solve_swarm(&extended, swarm);
    let conquer_ms = start.elapsed().as_secs_f64() * 1000.0;
    let model = restrict(model, formula, &parity);
    if let Some(model) = &model { assert!(verify_model(formula, model)); }
    ParallelHybridReport { model, symmetry, parity, swarm, conquer_ms }
}

pub fn verify_model(formula: &[Clause], model: &[i32]) -> bool {
    use std::collections::BTreeMap;
    let mut values = BTreeMap::new();
    for &literal in model {
        if literal == 0 || literal == i32::MIN { return false; }
        if let Some(previous) = values.insert(literal.unsigned_abs(), literal > 0) {
            if previous != (literal > 0) { return false; }
        }
    }
    formula.iter().all(|clause| clause.iter().any(|&literal| {
        values.get(&literal.unsigned_abs()) == Some(&(literal > 0))
    }))
}

pub fn solve_hybrid(formula: &[Clause], config: HybridConfig) -> HybridReport {
    use std::time::Instant;
    let (extended, symmetry, parity) = preprocess(formula, config, None);
    let start = Instant::now();
    // El oráculo vive dentro del CDCL y solo escribe fases; su modelo se verifica aquí.
    let feedback = cdcl::ReignitionConfig {
        interval: config.reignition_interval, oracle: config.oracle,
    };
    let (model, cdcl) = cdcl::solve_cdcl_with_reignition(&extended, feedback);
    let cdcl_ms = start.elapsed().as_secs_f64() * 1000.0;
    let model = restrict(model, formula, &parity);
    if let Some(model) = &model {
        assert!(verify_model(formula, model), "Model must satisfy original clauses");
    }
    HybridReport { model, symmetry, parity, cdcl, cdcl_ms }
}

// Como `solve_hybrid`, dejando en `proof` una traza de todo lo que se añade a la fórmula:
// la dominancia de cada simetría, las derivaciones de la paridad y cada cláusula aprendida
// o borrada por el CDCL. La capa de simetría solo participa si el formato tiene regla de
// dominancia (`PbProof`); con DRAT no rompe nada. Un resultado UNSAT termina con la
// cláusula vacía; un modelo no necesita traza.
pub fn solve_hybrid_with_proof(
    formula: &[Clause], config: HybridConfig, proof: &mut dyn proof::ProofLog,
) -> HybridReport {
    use std::time::Instant;
    let (extended, symmetry, parity) = preprocess(formula, config, Some(&mut *proof));
    let start = Instant::now();
    let feedback = cdcl::ReignitionConfig {
        interval: config.reignition_interval, oracle: config.oracle,
    };
    let (model, cdcl) = if parity.refuted {
        // La derivación de 0 = 1 ya cerró la traza con la cláusula vacía.
        (None, SolveStats { conflicts: 1, ..SolveStats::default() })
    } else {
        cdcl::solve_cdcl_with_proof(&extended, feedback, proof)
    };
    if formula.iter().any(Vec::is_empty) {
        proof.input_refuted();
    }
    let cdcl_ms = start.elapsed().as_secs_f64() * 1000.0;
    let model = restrict(model, formula, &parity);
    if let Some(model) = &model {
        assert!(verify_model(formula, model), "Model must satisfy original clauses");
    }
    HybridReport { model, symmetry, parity, cdcl, cdcl_ms }
}

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
    println!("{name},serial,{n},{},{serial_status},{serial_ms:.6},{:.6},{},{},{},{},{:.6},1,0,0,0,0,0,0,0,0,0,1.000000,0,0.0000,{},{},{:.6}",
        formula.len(), serial.cdcl_ms, serial.cdcl.decisions,
        serial.cdcl.conflicts, serial.cdcl.reignitions, serial.cdcl.reignition_flips,
        serial.cdcl.reignition_ms, serial.symmetry.generators, serial.symmetry.clauses,
        serial.symmetry.ms);
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
        println!("{name},{mode},{n},{},{status},{elapsed_ms:.6},{:.6},{},{},{},{},{:.6},{},{},{},{},{},{},{},{},{},{},{speedup:.6},{},{reduction},{},{},{:.6}",
            formula.len(), report.conquer_ms,
            total.decisions, total.conflicts, total.reignitions, total.reignition_flips,
            total.reignition_ms, stats.workers, stats.cubes_generated, stats.cubes_started,
            stats.cubes_unsat, stats.cubes_cancelled,
            total.exported_clauses, total.imported_clauses, total.dropped_exports,
            total.import_backjumps, stats.shared_clauses.len(),
            no_share_decisions.map_or_else(|| "NA".to_owned(), |value| value.to_string()),
            report.symmetry.generators, report.symmetry.clauses, report.symmetry.ms);
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
    println!("case,mode,n,clauses,status,elapsed_ms,conquer_ms,decisions,conflicts,reignitions,reignition_flips,reignition_ms,workers,cubes_generated,cubes_started,cubes_unsat,cubes_cancelled,exports,imports,sharing_drops,import_backjumps,pool_clauses,speedup_vs_serial,private_decisions,decision_reduction_vs_private_pct,symmetry_generators,symmetry_clauses,symmetry_ms");
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
