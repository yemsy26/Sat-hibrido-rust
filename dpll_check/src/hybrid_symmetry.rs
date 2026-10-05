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
