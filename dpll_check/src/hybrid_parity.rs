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
