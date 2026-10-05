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
