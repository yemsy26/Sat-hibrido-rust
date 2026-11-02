// Inprocessing: equivalencias del grafo de implicaciones y cierre por congruencia de puertas.
//
// Corrección. Una puerta AND, XOR o ITE se reconoce solo cuando la fórmula contiene todas las
// cláusulas de su definición. Dos puertas iguales con entradas equivalentes tienen salidas
// equivalentes en todo modelo, igual que dos literales en la misma componente fuerte del grafo
// de implicaciones. Sustituir cada clase por un representante conserva la satisfacibilidad y
// el modelo original se recupera copiando el valor del representante.
//
// Certificación. Cada equivalencia entra en la traza como dos cláusulas binarias y cada valor
// fijo como una unidad, siempre antes de usarse. Con ellas la propagación unitaria sobre la
// fórmula original simula la de la fórmula ya sustituida, así que una AND es RUP directa. Una
// XOR o una ITE no propagan sin saber sus entradas: se parte por casos sobre ellas con lemas
// RUP que se borran al terminar. Las cláusulas sustituidas son RUP por lo mismo; después se
// borran las equivalencias y las originales, que el CDCL ya no usa y frenarían al verificador.
use super::proof::ProofLog;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

pub type Clause = Vec<i32>;

// La capa es opcional: cualquier tope solo la debilita, nunca compromete la corrección.
const MAX_AND_INPUTS: usize = 64;
const MAX_XOR_VARIABLES: usize = 5;
const MAX_SPLITS: usize = 6;
const MAX_ROUNDS: usize = 16;
const MAX_ID: u32 = 1 << 28;
const EXTRACTION_WORK: u64 = 1 << 31;
const PROPAGATION_WORK: u64 = 1 << 27;

// Literales internos: 2 * variable + signo. La variable 0 es la constante verdadera.
const TRUE: u32 = 0;
const FALSE: u32 = 1;

// Imagen de una variable fijada; cualquier otra imagen es un literal externo.
const IMAGE_TRUE: i32 = i32::MAX;
const IMAGE_FALSE: i32 = i32::MAX - 1;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind { And, Xor, Ite }

// out = AND(inputs), out = XOR(inputs) o out = ITE(inputs[0], inputs[1], inputs[2]).
struct Gate {
    kind: Kind,
    out: u32,
    inputs: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct Congruence {
    pub ands: usize,
    pub xors: usize,
    pub ites: usize,
    pub equivalences: usize,
    pub units: usize,
    pub refuted: bool,
    // Por identificador externo: 0 si la variable sigue en la fórmula.
    image: Vec<i32>,
    // Binarias que entraron en la traza, para borrarlas tras sustituir.
    lemmas: Vec<[i32; 2]>,
}

struct Closure<'a> {
    table: Vec<u32>,
    ids: Vec<u32>,
    parent: Vec<u32>,
    gates: Vec<Gate>,
    occurs: Vec<Vec<u32>>,
    known: HashMap<Vec<u32>, (u32, u32)>,
    queue: Vec<u32>,
    log: Option<&'a mut dyn ProofLog>,
    lemmas: Vec<[i32; 2]>,
    merges: usize,
    refuted: bool,
}

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

// Deriva `clause` partiendo por casos: cada hoja es RUP y cada nodo se sigue de sus dos hijos,
// que se borran enseguida para no cargar al verificador.
fn derive(log: &mut dyn ProofLog, clause: &mut Vec<i32>, splits: &[i32]) {
    if let Some((&split, rest)) = splits.split_first() {
        for literal in [split, -split] {
            clause.push(literal);
            derive(log, clause, rest);
            clause.pop();
        }
    }
    log.add(clause);
    if let Some(&split) = splits.first() {
        for literal in [split, -split] {
            clause.push(literal);
            log.delete(clause);
            clause.pop();
        }
    }
}

impl Closure<'_> {
    fn find(&mut self, mut literal: u32) -> u32 {
        loop {
            let up = self.parent[literal as usize];
            if up == literal {
                return literal;
            }
            let top = self.parent[up as usize];
            self.parent[literal as usize] = top;
            self.parent[literal as usize ^ 1] = top ^ 1;
            literal = top;
        }
    }
    fn internal(&self, literal: i32) -> u32 {
        (self.table[literal.unsigned_abs() as usize] << 1) | u32::from(literal < 0)
    }
    fn external(&self, literal: u32) -> i32 {
        let id = self.ids[(literal >> 1) as usize] as i32;
        if literal & 1 == 0 { id } else { -id }
    }
    // Literales de la cláusula bajo las clases actuales; falso si ya no restringe nada.
    fn view(&mut self, clause: &[i32], out: &mut Vec<u32>) -> bool {
        out.clear();
        for &literal in clause {
            let internal = self.internal(literal);
            match self.find(internal) {
                TRUE => return false,
                FALSE => {}
                root => out.push(root),
            }
        }
        out.sort_unstable();
        out.dedup();
        !out.windows(2).any(|pair| pair[0] ^ 1 == pair[1])
    }
    // Entradas sobre las que hay que partir para que la definición de la puerta propague.
    fn case_inputs(&self, gate: u32) -> &[u32] {
        let gate = &self.gates[gate as usize];
        match gate.kind {
            Kind::And => &[],
            Kind::Ite => &gate.inputs[..1],
            Kind::Xor => &gate.inputs[..gate.inputs.len() - 1],
        }
    }
    fn splits(&mut self, sources: [Option<u32>; 2], targets: [u32; 2]) -> Vec<i32> {
        let mut roots: Vec<u32> = Vec::new();
        for source in sources.into_iter().flatten() {
            for position in 0..self.case_inputs(source).len() {
                let input = self.case_inputs(source)[position];
                roots.push(self.find(input) & !1);
            }
        }
        roots.sort_unstable();
        roots.dedup();
        roots.retain(|&root| root != TRUE && root != targets[0] & !1 && root != targets[1] & !1);
        roots.iter().map(|&root| self.external(root)).collect()
    }
    // a ≡ b, sabiendo que lo implican las puertas `sources` o el grafo de implicaciones.
    fn merge(&mut self, a: u32, b: u32, sources: [Option<u32>; 2]) {
        let (a, b) = if a >> 1 == 0 { (b, a) } else { (a, b) };
        let (first, second) = (self.find(a), self.find(b));
        if first == second || self.refuted {
            return;
        }
        let splits = self.splits(sources, [first, second]);
        if splits.len() > MAX_SPLITS {
            return;
        }
        let contradiction = first == second ^ 1;
        let (left, right) = (self.external(a), self.external(b));
        if let Some(log) = self.log.as_deref_mut() {
            if b >> 1 == 0 {
                derive(log, &mut vec![if b == TRUE { left } else { -left }], &splits);
            } else if a == b ^ 1 {
                derive(log, &mut vec![right], &splits);
                derive(log, &mut vec![left], &splits);
            } else {
                derive(log, &mut vec![-left, right], &splits);
                derive(log, &mut vec![left, -right], &splits);
                self.lemmas.extend([[-left, right], [left, -right]]);
                if contradiction {
                    log.add(&[left]);
                }
            }
            if contradiction {
                log.add(&[]);
            }
        }
        if contradiction {
            self.refuted = true;
            return;
        }
        self.merges += 1;
        let (lost, kept) = if first >> 1 == 0
            || (second >> 1 != 0 && self.class_gates(first) >= self.class_gates(second)) {
            (second, first)
        } else {
            (first, second)
        };
        self.parent[lost as usize] = kept;
        self.parent[lost as usize ^ 1] = kept ^ 1;
        if !self.occurs.is_empty() {
            let moved = std::mem::take(&mut self.occurs[(lost >> 1) as usize]);
            self.queue.extend_from_slice(&moved);
            if kept >> 1 != 0 {
                self.occurs[(kept >> 1) as usize].extend(moved);
            }
        }
    }
    fn class_gates(&self, root: u32) -> usize {
        self.occurs.get((root >> 1) as usize).map_or(0, Vec::len)
    }
    fn matched(&mut self, key: Vec<u32>, out: u32, gate: u32) {
        match self.known.get(&key) {
            Some(&(other, source)) => self.merge(out, other, [Some(gate), Some(source)]),
            None => {
                self.known.insert(key, (out, gate));
            }
        }
    }
    fn normal_and(&mut self, out: u32, mut inputs: Vec<u32>, gate: u32) {
        inputs.sort_unstable();
        inputs.dedup();
        if inputs.first() == Some(&FALSE) || inputs.get(1) == Some(&FALSE)
            || inputs.windows(2).any(|pair| pair[0] ^ 1 == pair[1]) {
            return self.merge(out, FALSE, [Some(gate), None]);
        }
        inputs.retain(|&input| input != TRUE);
        match inputs.len() {
            0 => self.merge(out, TRUE, [Some(gate), None]),
            1 => self.merge(out, inputs[0], [Some(gate), None]),
            _ => {
                inputs.push(Kind::And as u32);
                self.matched(inputs, out, gate);
            }
        }
    }
    // `inputs` ya son raíces: los pares se cancelan y cada signo o constante va a la salida.
    fn normal_xor(&mut self, mut out: u32, inputs: &[u32], gate: u32) {
        let mut rest: Vec<u32> = Vec::with_capacity(inputs.len() + 1);
        for &input in inputs {
            if input >> 1 == 0 {
                out ^= u32::from(input == TRUE);
            } else {
                out ^= input & 1;
                rest.push(input & !1);
            }
        }
        rest.sort_unstable();
        let mut kept: Vec<u32> = Vec::with_capacity(rest.len() + 1);
        for input in rest {
            if kept.last() == Some(&input) { kept.pop(); } else { kept.push(input); }
        }
        match kept.len() {
            0 => self.merge(out, FALSE, [Some(gate), None]),
            1 => self.merge(out, kept[0], [Some(gate), None]),
            _ => {
                kept.push(Kind::Xor as u32);
                self.matched(kept, out, gate);
            }
        }
    }
    fn normal_ite(&mut self, mut out: u32, [mut c, mut t, mut e]: [u32; 3], gate: u32) {
        if c >> 1 == 0 {
            return self.merge(out, if c == TRUE { t } else { e }, [Some(gate), None]);
        }
        if c & 1 == 1 {
            c ^= 1;
            std::mem::swap(&mut t, &mut e);
        }
        if t == e {
            return self.merge(out, t, [Some(gate), None]);
        }
        if t == e ^ 1 {
            return self.normal_xor(out, &[c, e], gate);
        }
        if t == c || t == TRUE {
            return self.normal_and(out ^ 1, vec![c ^ 1, e ^ 1], gate);
        }
        if t == c ^ 1 || t == FALSE {
            return self.normal_and(out, vec![c ^ 1, e], gate);
        }
        if e == c || e == FALSE {
            return self.normal_and(out, vec![c, t], gate);
        }
        if e == c ^ 1 || e == TRUE {
            return self.normal_and(out ^ 1, vec![c, t ^ 1], gate);
        }
        if t & 1 == 1 {
            t ^= 1;
            e ^= 1;
            out ^= 1;
        }
        self.matched(vec![c, t, e, Kind::Ite as u32], out, gate);
    }
    fn normalize(&mut self, gate: u32) {
        let mut inputs = std::mem::take(&mut self.gates[gate as usize].inputs);
        let original = inputs.clone();
        for input in &mut inputs {
            *input = self.find(*input);
        }
        self.gates[gate as usize].inputs = original;
        let (kind, out) = (self.gates[gate as usize].kind, self.gates[gate as usize].out);
        match kind {
            Kind::And => self.normal_and(out, inputs, gate),
            Kind::Xor => self.normal_xor(out, &inputs, gate),
            Kind::Ite => self.normal_ite(out, [inputs[0], inputs[1], inputs[2]], gate),
        }
    }
    fn close_gates(&mut self) {
        self.occurs = vec![Vec::new(); self.parent.len() / 2];
        for gate in 0..self.gates.len() {
            let mut roots: Vec<u32> = Vec::with_capacity(self.gates[gate].inputs.len());
            for position in 0..self.gates[gate].inputs.len() {
                let input = self.gates[gate].inputs[position];
                roots.push(self.find(input) >> 1);
            }
            roots.sort_unstable();
            roots.dedup();
            for root in roots {
                if root != 0 {
                    self.occurs[root as usize].push(gate as u32);
                }
            }
            self.queue.push(gate as u32);
        }
        while let Some(gate) = self.queue.pop() {
            if self.refuted {
                self.queue.clear();
                break;
            }
            self.normalize(gate);
        }
        self.gates.clear();
        self.known.clear();
        self.occurs = Vec::new();
    }
    // Fija lo que ya es unidad y devuelve las binarias como pares ordenados (literal, compañero).
    fn binaries(&mut self, parts: &[&[Clause]]) -> Vec<(u32, u32)> {
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        let mut viewed = Vec::new();
        let before = self.merges;
        for clause in parts.iter().copied().flatten() {
            // Sin fusiones ninguna cláusula larga se ha acortado.
            if (self.merges != 0 || clause.len() <= 2) && self.view(clause, &mut viewed) {
                match viewed[..] {
                    // Cada unidad se fija al verla: un circuito en orden se propaga en una pasada.
                    [unit] => self.merge(unit, TRUE, [None, None]),
                    [first, second] => {
                        pairs.push((first, second));
                        pairs.push((second, first));
                    }
                    _ => {}
                }
            }
        }
        // Con fusiones hay otra pasada y estos pares ya no sirven: ordenarlos solo gastaría tiempo.
        if self.merges != before {
            return Vec::new();
        }
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }
    // Componentes fuertes del grafo de implicaciones, con Tarjan iterativo.
    fn components(&mut self, pairs: &[(u32, u32)]) {
        let count = self.parent.len();
        let mut start = vec![0_u32; count + 1];
        for &(literal, _) in pairs {
            start[literal as usize + 1] += 1;
        }
        for literal in 0..count {
            start[literal + 1] += start[literal];
        }
        // Un literal implica a los compañeros de su negación.
        let edges = |literal: u32| {
            let source = literal as usize ^ 1;
            start[source] as usize..start[source + 1] as usize
        };
        const DONE: u32 = u32::MAX;
        let mut index = vec![0_u32; count];
        let mut low = vec![0_u32; count];
        let mut stack: Vec<u32> = Vec::new();
        let mut calls: Vec<(u32, usize)> = Vec::new();
        let mut clock = 0;
        let mut equal: Vec<(u32, u32)> = Vec::new();
        for root in 2..count as u32 {
            if index[root as usize] != 0 || edges(root).is_empty() {
                continue;
            }
            clock += 1;
            index[root as usize] = clock;
            low[root as usize] = clock;
            stack.push(root);
            calls.push((root, edges(root).start));
            while let Some(&mut (literal, ref mut next)) = calls.last_mut() {
                if *next < edges(literal).end {
                    let successor = pairs[*next].1;
                    *next += 1;
                    let seen = index[successor as usize];
                    if seen == 0 {
                        clock += 1;
                        index[successor as usize] = clock;
                        low[successor as usize] = clock;
                        stack.push(successor);
                        calls.push((successor, edges(successor).start));
                    } else if seen != DONE {
                        low[literal as usize] = low[literal as usize].min(seen);
                    }
                    continue;
                }
                calls.pop();
                let reach = low[literal as usize];
                if let Some(&(caller, _)) = calls.last() {
                    low[caller as usize] = low[caller as usize].min(reach);
                }
                if reach == index[literal as usize] {
                    let first = stack.len() - stack.iter().rev().position(|&member| member == literal).expect("Literal is on the stack") - 1;
                    for &member in &stack[first..] {
                        index[member as usize] = DONE;
                        if member != literal {
                            equal.push((member, literal));
                        }
                    }
                    stack.truncate(first);
                }
            }
        }
        for (member, literal) in equal {
            self.merge(member, literal, [None, None]);
        }
    }
    // Puertas AND: la cláusula larga (x ∨ ¬a ∨ ¬b ...) y las binarias (¬x ∨ a), (¬x ∨ b)...
    fn extract_ands(&mut self, parts: &[&[Clause]], pairs: &[(u32, u32)], work: &mut u64) -> usize {
        if pairs.is_empty() {
            return 0;
        }
        let mut found = 0;
        let mut viewed = Vec::new();
        let degree = |literal: u32| {
            pairs.partition_point(|&(first, _)| first < literal + 1)
                - pairs.partition_point(|&(first, _)| first < literal)
        };
        for clause in parts.iter().copied().flatten() {
            if clause.len() < 3 || clause.len() > MAX_AND_INPUTS + 1 || *work == 0 {
                continue;
            }
            if !self.view(clause, &mut viewed) || viewed.len() < 3 {
                continue;
            }
            for &out in &viewed {
                *work = work.saturating_sub(viewed.len() as u64);
                if degree(out ^ 1) + 1 < viewed.len() {
                    continue;
                }
                let complete = viewed.iter().all(|&other| {
                    other == out || pairs.binary_search(&(out ^ 1, other ^ 1)).is_ok()
                });
                if complete {
                    let inputs = viewed.iter().filter(|&&other| other != out).map(|&other| other ^ 1).collect();
                    self.gates.push(Gate { kind: Kind::And, out, inputs });
                    found += 1;
                }
            }
        }
        found
    }
    // Huella del soporte de la cláusula, si puede ser parte de una XOR o una ITE. Sin fusiones
    // basta con los identificadores: no toca las tablas, que en fórmulas enormes fallan caché.
    fn support_print(&mut self, clause: &[i32], viewed: &mut Vec<u32>) -> Option<u64> {
        if !(3..=2 * MAX_XOR_VARIABLES).contains(&clause.len()) {
            return None;
        }
        if self.merges == 0 {
            viewed.clear();
            viewed.extend(clause.iter().map(|literal| (literal.unsigned_abs() << 1) | u32::from(*literal < 0)));
            viewed.sort_unstable();
            viewed.dedup();
            if viewed.windows(2).any(|pair| pair[0] ^ 1 == pair[1]) {
                return None;
            }
        } else if !self.view(clause, viewed) {
            return None;
        }
        (3..=MAX_XOR_VARIABLES).contains(&viewed.len()).then(|| {
            viewed.iter().fold(viewed.len() as u64, |print, &literal| mix(print ^ u64::from(literal >> 1)))
        })
    }
    // XOR e ITE necesitan varias cláusulas sobre el mismo soporte: las que están solas, que en
    // 3-SAT aleatorio son casi todas, se descartan por su huella sin llegar a agruparlas.
    fn extract_supports(&mut self, parts: &[&[Clause]]) -> (usize, usize) {
        let mut viewed = Vec::new();
        let mut prints: Vec<u64> = Vec::new();
        for clause in parts.iter().copied().flatten() {
            prints.extend(self.support_print(clause, &mut viewed));
        }
        prints.sort_unstable();
        let mut repeated: Vec<u64> = prints.windows(2).filter(|pair| pair[0] == pair[1]).map(|pair| pair[0]).collect();
        drop(prints);
        repeated.dedup();
        if repeated.is_empty() {
            return (0, 0);
        }
        // Soporte en orden creciente y patrón de negaciones por bit.
        let mut groups: HashMap<Vec<u32>, u32> = HashMap::new();
        for clause in parts.iter().copied().flatten() {
            let candidate = self.support_print(clause, &mut viewed).is_some_and(|print| repeated.binary_search(&print).is_ok());
            if candidate && self.view(clause, &mut viewed) && (3..=MAX_XOR_VARIABLES).contains(&viewed.len()) {
                let pattern = viewed.iter().enumerate()
                    .fold(0_usize, |pattern, (bit, &literal)| pattern | ((literal & 1) as usize) << bit);
                let support: Vec<u32> = viewed.iter().map(|&literal| literal & !1).collect();
                *groups.entry(support).or_insert(0) |= 1 << pattern;
            }
        }
        let mut xors = 0;
        // Bajo c: x ≡ t. Clave (x, c), con x positivo.
        let mut halves: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for (support, mask) in &groups {
            let arity = support.len();
            let half = 1_u32 << (arity - 1);
            for negations in 0..2_u32 {
                let present = (0..1_u32 << arity)
                    .filter(|&pattern| pattern.count_ones() & 1 == negations && mask >> pattern & 1 == 1)
                    .count() as u32;
                if present != half {
                    continue;
                }
                // Negaciones pares: el XOR del soporte vale 1; impares: 0.
                xors += 1;
                for (position, &variable) in support.iter().enumerate() {
                    let mut inputs = support.clone();
                    inputs.remove(position);
                    let out = variable ^ u32::from(negations == 0);
                    self.gates.push(Gate { kind: Kind::Xor, out, inputs });
                }
            }
            if arity != 3 {
                continue;
            }
            // Dos cláusulas (a ∨ b ∨ d), (a ∨ ¬b ∨ ¬d): bajo ¬a, b ≡ ¬d.
            for first in (0..8_u32).filter(|&pattern| mask >> pattern & 1 == 1) {
                for shared in 0..3 {
                    let second = first ^ 0b111 ^ (1 << shared);
                    if second <= first || mask >> second & 1 == 0 {
                        continue;
                    }
                    let literal = |position: usize| support[position] | (first >> position & 1);
                    let (b, d) = ((shared + 1) % 3, (shared + 2) % 3);
                    let condition = literal(shared) ^ 1;
                    let (b, d) = (literal(b), literal(d));
                    halves.entry((b & !1, condition)).or_default().push(d ^ 1 ^ (b & 1));
                    halves.entry((d & !1, condition)).or_default().push(b ^ 1 ^ (d & 1));
                }
            }
        }
        let mut ites = 0;
        for (&(out, condition), thens) in &halves {
            if condition & 1 == 1 {
                continue;
            }
            let Some(elses) = halves.get(&(out, condition ^ 1)) else { continue };
            for &then in thens {
                for &otherwise in elses {
                    if then != otherwise ^ 1 {
                        self.gates.push(Gate { kind: Kind::Ite, out, inputs: vec![condition, then, otherwise] });
                        ites += 1;
                    }
                }
            }
        }
        (xors, ites)
    }
}

// Calcula las equivalencias de `parts`, sin tocarlas. `rewrite` aplica el resultado.
pub fn close(
    parts: &[&[Clause]], proof: Option<&mut (dyn ProofLog + '_)>, stop: Option<&AtomicBool>,
) -> Congruence {
    let mut result = Congruence::default();
    let last = parts.iter().copied().flatten().flatten().map(|literal| literal.unsigned_abs()).max().unwrap_or(0);
    if last == 0 || last > MAX_ID || parts.iter().copied().flatten().any(Vec::is_empty) {
        return result;
    }
    let mut table = vec![0_u32; last as usize + 1];
    for literal in parts.iter().copied().flatten().flatten() {
        table[literal.unsigned_abs() as usize] = 1;
    }
    let mut ids = vec![0_u32];
    for id in 1..=last {
        if table[id as usize] != 0 {
            table[id as usize] = ids.len() as u32;
            ids.push(id);
        }
    }
    let literals = 2 * ids.len() as u32;
    let mut closure = Closure {
        table, ids, parent: (0..literals).collect(), gates: Vec::new(), occurs: Vec::new(),
        known: HashMap::new(), queue: Vec::new(), log: proof.map(|log| log as &mut dyn ProofLog),
        lemmas: Vec::new(), merges: 0, refuted: false,
    };
    let stopped = || stop.is_some_and(|flag| flag.load(Ordering::Acquire));
    let mut work = EXTRACTION_WORK;
    let size = parts.iter().copied().flatten().map(|clause| clause.len() as u64).sum::<u64>();
    let mut propagation = PROPAGATION_WORK;
    let mut rounds = 0;
    // Una cadena de constantes avanza un paso por pasada. No gasta rondas, que son para las
    // puertas: si las gastara, un circuito grande nunca llegaría a extraerlas.
    while rounds < MAX_ROUNDS {
        if closure.refuted || stopped() {
            break;
        }
        let before = closure.merges;
        let pairs = closure.binaries(parts);
        propagation = propagation.saturating_sub(size);
        if closure.merges == before && !pairs.is_empty() {
            closure.components(&pairs);
        }
        if closure.refuted {
            break;
        }
        if closure.merges != before {
            if propagation == 0 {
                break;
            }
            continue;
        }
        rounds += 1;
        let ands = closure.extract_ands(parts, &pairs, &mut work);
        drop(pairs);
        let (xors, ites) = closure.extract_supports(parts);
        result.ands = result.ands.max(ands);
        result.xors = result.xors.max(xors);
        result.ites = result.ites.max(ites);
        if closure.gates.is_empty() {
            break;
        }
        closure.close_gates();
        if closure.merges == before {
            break;
        }
    }
    result.refuted = closure.refuted;
    if closure.merges == 0 || closure.refuted {
        return result;
    }
    result.lemmas = std::mem::take(&mut closure.lemmas);
    // El representante de cada clase es su variable de identificador menor.
    result.image = vec![0; last as usize + 1];
    let mut chosen = vec![u32::MAX; closure.ids.len()];
    for variable in 1..closure.ids.len() as u32 {
        let root = closure.find(variable << 1);
        let id = closure.ids[variable as usize] as usize;
        if root >> 1 == 0 {
            result.image[id] = if root == TRUE { IMAGE_TRUE } else { IMAGE_FALSE };
            result.units += 1;
        } else if chosen[(root >> 1) as usize] == u32::MAX {
            chosen[(root >> 1) as usize] = (variable << 1) | (root & 1);
        } else {
            let image = chosen[(root >> 1) as usize] ^ (root & 1);
            result.image[id] = closure.external(image);
            result.equivalences += 1;
        }
    }
    result
}

impl Congruence {
    pub fn is_identity(&self) -> bool {
        self.image.is_empty()
    }
    // La fórmula sobre los representantes, sin repetir cláusulas. Una cláusula vacía la refuta.
    pub fn rewrite(&self, clauses: &[Clause], mut proof: Option<&mut (dyn ProofLog + '_)>) -> Vec<Clause> {
        let mut rewritten: Vec<Clause> = Vec::with_capacity(clauses.len());
        let mut changed: Vec<bool> = Vec::with_capacity(clauses.len());
        let mut buffer: Vec<i32> = Vec::new();
        'clauses: for clause in clauses {
            buffer.clear();
            let mut different = false;
            for &literal in clause {
                let id = literal.unsigned_abs() as usize;
                let image = self.image.get(id).copied().unwrap_or(0);
                match image {
                    0 => buffer.push(literal),
                    IMAGE_TRUE | IMAGE_FALSE => {
                        different = true;
                        if (image == IMAGE_TRUE) == (literal > 0) {
                            continue 'clauses;
                        }
                    }
                    image => {
                        different = true;
                        buffer.push(if literal > 0 { image } else { -image });
                    }
                }
            }
            if buffer.is_empty() {
                // Todos sus literales están fijados a falso: la vacía es RUP y cierra la traza.
                if let Some(log) = proof.as_deref_mut() {
                    log.add(&[]);
                }
                return vec![Vec::new()];
            }
            if different {
                buffer.sort_unstable_by_key(|literal| (literal.unsigned_abs(), *literal));
                buffer.dedup();
                if buffer.windows(2).any(|pair| pair[0] == -pair[1]) {
                    continue;
                }
            }
            rewritten.push(buffer.clone());
            changed.push(different);
        }
        // Mismas cláusulas por huella; la igualdad se comprueba con los literales ordenados.
        let sorted = |clause: &Clause| {
            let mut copy = clause.clone();
            copy.sort_unstable();
            copy
        };
        let mut prints: Vec<(u64, u32)> = rewritten.iter().enumerate().map(|(index, clause)| {
            (sorted(clause).iter().fold(clause.len() as u64, |print, &literal| mix(print ^ u64::from(literal as u32))), index as u32)
        }).collect();
        prints.sort_unstable();
        let mut repeated = vec![false; rewritten.len()];
        for run in prints.chunk_by(|a, b| a.0 == b.0) {
            let mut kept: Vec<(Clause, u32)> = Vec::new();
            for &(_, index) in run {
                let copy = sorted(&rewritten[index as usize]);
                match kept.iter_mut().find(|(other, _)| *other == copy) {
                    // Se conserva la original si la hay: esa ya está en la traza.
                    Some((_, first)) => {
                        let drop = if changed[*first as usize] && !changed[index as usize] {
                            std::mem::replace(first, index)
                        } else {
                            index
                        };
                        repeated[drop as usize] = true;
                    }
                    None => kept.push((copy, index)),
                }
            }
        }
        let mut kept: Vec<Clause> = Vec::with_capacity(rewritten.len());
        let image = &self.image;
        for (index, clause) in rewritten.into_iter().enumerate() {
            if repeated[index] {
                continue;
            }
            if changed[index] {
                if let Some(log) = proof.as_deref_mut() {
                    log.add(&clause);
                }
            }
            kept.push(clause);
        }
        if let Some(log) = proof {
            // Cada cláusula una vez; con un literal repetido el verificador la guarda de otra forma.
            let mut unused: Vec<Clause> = clauses.iter().filter(|clause| clause.iter().any(|literal| {
                image.get(literal.unsigned_abs() as usize).is_some_and(|&target| target != 0)
            })).map(sorted).chain(self.lemmas.iter().map(|lemma| sorted(&lemma.to_vec()))).filter(|clause| {
                let mut variables: Vec<u32> = clause.iter().map(|literal| literal.unsigned_abs()).collect();
                variables.sort_unstable();
                variables.windows(2).all(|pair| pair[0] != pair[1])
            }).collect();
            unused.sort_unstable();
            unused.dedup();
            log.release_core();
            for clause in &unused {
                log.delete(clause);
            }
        }
        kept
    }
    // El modelo de la fórmula sustituida se extiende a las variables de `formula`. Una variable
    // que se quedó sin cláusulas no restringe nada y puede valer cualquier cosa.
    pub fn extend(&self, model: Vec<i32>, formula: &[Clause]) -> Vec<i32> {
        if self.is_identity() {
            return model;
        }
        let mut values = vec![false; self.image.len()];
        for &literal in &model {
            values[literal.unsigned_abs() as usize] = literal > 0;
        }
        let mut seen = vec![false; self.image.len()];
        for literal in formula.iter().flatten() {
            seen[literal.unsigned_abs() as usize] = true;
        }
        (1..self.image.len()).filter(|&id| seen[id]).map(|id| {
            let value = match self.image[id] {
                0 => values[id],
                IMAGE_TRUE => true,
                IMAGE_FALSE => false,
                image => values[image.unsigned_abs() as usize] == (image > 0),
            };
            if value { id as i32 } else { -(id as i32) }
        }).collect()
    }
}
