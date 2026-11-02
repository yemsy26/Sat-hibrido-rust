//! Referencia CPU del DFS con diario de `ramsey43_native.cu`, generalizada a (s, t, n).
//!
//! Reproduce `ramsey43_search_journal_body` con un solo universo (prefix_bits=0): misma
//! primera arista libre lexicográfica, mismo primer color por hash, mismo diario de frames
//! {arista, color, alternativa consumida, forzada} y los mismos contadores. Con
//! `lookahead`, el oráculo de colores permitidos combina el K_s/K_t nuevo con las cotas de
//! grado R(s-1,t)-1 (rojo) y R(s,t-1)-1 (azul); con s=t=5, n=43 son las 24/24 del kernel.
//! Las heurísticas sólo eligen la siguiente arista o el primer color: toda poda pasa por el
//! oráculo exacto, así que UNSAT depende sólo de él y del agotamiento del diario.
use std::ops::RangeInclusive;
use std::time::{Duration, Instant};

pub const MAX_VERTICES: usize = 64;
pub const RED: u32 = 0;
pub const BLUE: u32 = 1;

/// R(s,t) exactos ya demostrados (Radziszowski, "Small Ramsey Numbers", DS1).
pub fn known_ramsey(s: u32, t: u32) -> Option<u32> {
    match (s.min(t), s.max(t)) {
        (1, _) => Some(1),
        (2, t) => Some(t),
        (3, 3) => Some(6),
        (3, 4) => Some(9),
        (3, 5) => Some(14),
        (3, 6) => Some(18),
        (3, 7) => Some(23),
        (3, 8) => Some(28),
        (3, 9) => Some(36),
        (4, 4) => Some(18),
        (4, 5) => Some(25),
        _ => None,
    }
}

/// K_n bicoloreado sin K_s rojo ni K_t azul.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Problem {
    pub s: u32,
    pub t: u32,
    pub n: usize,
}

impl Problem {
    pub fn new(s: u32, t: u32, n: usize) -> Result<Self, String> {
        if s < 2 || t < 2 || !(1..=MAX_VERTICES).contains(&n) {
            return Err(format!("se requiere s,t >= 2 y 1 <= n <= {MAX_VERTICES}"));
        }
        Ok(Self { s, t, n })
    }
    pub fn edges(&self) -> usize {
        self.n * (self.n - 1) / 2
    }
    fn clique(&self, colour: u32) -> u32 {
        if colour == RED {
            self.s
        } else {
            self.t
        }
    }
    /// N_rojo(v) no contiene K_{s-1} rojo ni K_t azul: grado rojo <= R(s-1,t)-1.
    pub fn red_cap(&self) -> u32 {
        self.cap(known_ramsey(self.s - 1, self.t))
    }
    /// N_azul(v) no contiene K_s rojo ni K_{t-1} azul: grado azul <= R(s,t-1)-1.
    pub fn blue_cap(&self) -> u32 {
        self.cap(known_ramsey(self.s, self.t - 1))
    }
    fn cap(&self, ramsey: Option<u32>) -> u32 {
        let free = self.n as u32 - 1;
        ramsey.map_or(free, |r| (r - 1).min(free))
    }
    fn cap_of(&self, colour: u32) -> u32 {
        if colour == RED {
            self.red_cap()
        } else {
            self.blue_cap()
        }
    }
    /// Grados rojos del vértice 0 que cubren todas las soluciones, como `canonical_star`
    /// para K43 (18..=21): cotas de grado y, si s=t, el intercambio de colores permite
    /// suponer grado rojo <= (n-1)/2. Rango vacío: UNSAT sin búsqueda.
    pub fn star_degrees(&self) -> RangeInclusive<u32> {
        let free = self.n as u32 - 1;
        let low = free.saturating_sub(self.blue_cap());
        let mut high = self.red_cap();
        if self.s == self.t {
            high = high.min(free / 2);
        }
        low..=high
    }
    /// Vértice 0 rojo hacia 1..=degree y azul hacia el resto; renombrar vértices basta.
    pub fn canonical_star(&self, degree: u32) -> Colouring {
        let mut graph = Colouring::empty(self.n);
        for b in 1..self.n {
            graph.set(0, b, Some(if b as u32 <= degree { RED } else { BLUE }));
        }
        graph
    }
}

fn after(vertex: usize) -> u64 {
    if vertex >= 63 {
        0
    } else {
        u64::MAX << (vertex + 1)
    }
}

/// ¿Hay un K_size dentro de `candidates` en el grafo `rows`?
fn has_clique(rows: &[u64; MAX_VERTICES], candidates: u64, size: u32) -> bool {
    if size == 0 {
        return true;
    }
    let mut rest = candidates;
    while rest.count_ones() >= size {
        let vertex = rest.trailing_zeros() as usize;
        rest &= rest - 1;
        if has_clique(rows, rest & rows[vertex], size - 1) {
            return true;
        }
    }
    false
}

/// Filas `[rojo, azul][vértice]`; una arista desconocida no pertenece a ningún color.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Colouring {
    n: usize,
    rows: [[u64; MAX_VERTICES]; 2],
}

impl Colouring {
    pub fn empty(n: usize) -> Self {
        assert!(n <= MAX_VERTICES, "n > {MAX_VERTICES}");
        Self {
            n,
            rows: [[0; MAX_VERTICES]; 2],
        }
    }
    pub fn n(&self) -> usize {
        self.n
    }
    pub fn row(&self, colour: u32, vertex: usize) -> u64 {
        self.rows[colour as usize][vertex]
    }
    pub fn colour(&self, a: usize, b: usize) -> Option<u32> {
        [RED, BLUE]
            .into_iter()
            .find(|&colour| self.row(colour, a) >> b & 1 != 0)
    }
    /// `None` borra la arista, como colour=2 en `ramsey43_set_edge`.
    pub fn set(&mut self, a: usize, b: usize, colour: Option<u32>) {
        assert!(a != b && a < self.n && b < self.n, "arista ({a},{b}) fuera de K{}", self.n);
        for c in [RED, BLUE] {
            let on = u64::from(colour == Some(c));
            let rows = &mut self.rows[c as usize];
            rows[a] = rows[a] & !(1 << b) | on << b;
            rows[b] = rows[b] & !(1 << a) | on << a;
        }
    }
    fn vertices(&self) -> u64 {
        if self.n == 64 {
            u64::MAX
        } else {
            (1 << self.n) - 1
        }
    }
    fn unknown(&self, vertex: usize) -> u64 {
        self.vertices() & !(1 << vertex) & !(self.rows[0][vertex] | self.rows[1][vertex])
    }
    pub fn free_edges(&self) -> usize {
        (0..self.n)
            .map(|v| (self.unknown(v) & after(v)).count_ones() as usize)
            .sum()
    }
    /// Aristas libres en orden (0,1), (0,2), ..., (n-2,n-1).
    fn free_iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.n).flat_map(move |a| {
            let mut rest = self.unknown(a) & after(a);
            std::iter::from_fn(move || {
                (rest != 0).then(|| {
                    let b = rest.trailing_zeros() as usize;
                    rest &= rest - 1;
                    (a, b)
                })
            })
        })
    }
    /// `ramsey43_pick_free_edge`.
    pub fn first_free_edge(&self) -> Option<(usize, usize)> {
        self.free_iter().next()
    }
    pub fn has_clique(&self, colour: u32, size: u32) -> bool {
        has_clique(&self.rows[colour as usize], self.vertices(), size)
    }
    pub fn degree(&self, colour: u32, vertex: usize) -> u32 {
        self.row(colour, vertex).count_ones()
    }
}

/// Índice lexicográfico 0-based; para n=43 es el `ramsey43_edge(id, 0, 1)` del kernel.
pub fn edge_id(n: usize, a: usize, b: usize) -> usize {
    a * (2 * n - a - 1) / 2 + b - a - 1
}

fn mix(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// `ramsey43_first_colour`.
pub fn first_colour(seed: u32, universe: u32, edge: u32) -> u32 {
    mix(seed ^ universe.wrapping_mul(0x9e37_79b9) ^ edge.wrapping_mul(0x85eb_ca6b)) & 1
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Heuristic {
    /// Primera arista libre lexicográfica (`ramsey43_pick_free_edge`).
    Lex,
    /// MRV sobre todas las aristas libres: conflicto si alguna no admite color, forzada si
    /// admite uno; si no, la de más aristas asignadas en sus extremos y después más vecinos
    /// comunes del mismo color. Requiere lookahead.
    FailFirst,
    /// `ramsey43_search_neuro_smart`: cada `interval` decisiones libres consulta la red sobre
    /// las 32 primeras aristas libres lexicográficas. Requiere lookahead.
    Neural { interval: u32 },
    /// Control de `Neural`: misma cadencia y mismos 32 candidatos, pero arista y color
    /// uniformes (semilla de `Config`). Requiere lookahead.
    Random { interval: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub heuristic: Heuristic,
    pub lookahead: bool,
    pub seed: u32,
    pub universe: u32,
    /// Transiciones máximas; agotarlas da `Unknown`, nunca UNSAT.
    pub budget: u64,
}

impl Config {
    pub fn new(heuristic: Heuristic, lookahead: bool, budget: u64) -> Self {
        Self {
            heuristic,
            lookahead,
            seed: 0,
            universe: 0,
            budget,
        }
    }
}

/// Mismos significados que los contadores del kernel; las consultas son propias de la CPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub transitions: u64,
    pub prunes: u64,
    pub decisions: u64,
    pub forced: u64,
    pub checks: u64,
    pub conflicts: u64,
    pub policy_queries: u64,
    pub policy_fallbacks: u64,
}

impl Counters {
    fn add(&mut self, other: &Self) {
        self.transitions += other.transitions;
        self.prunes += other.prunes;
        self.decisions += other.decisions;
        self.forced += other.forced;
        self.checks += other.checks;
        self.conflicts += other.conflicts;
        self.policy_queries += other.policy_queries;
        self.policy_fallbacks += other.policy_fallbacks;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Sat(Colouring),
    Unsat,
    Unknown,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub outcome: Outcome,
    pub counters: Counters,
    pub elapsed: Duration,
}

struct Frame {
    a: usize,
    b: usize,
    colour: u32,
    alternate_consumed: bool,
    forced: bool,
}

struct Search<'a> {
    problem: Problem,
    config: Config,
    policy: Option<&'a Policy>,
    graph: Colouring,
    counters: Counters,
}

/// `ramsey43_allowed_colours` para (a,b) libre en un padre sin K monocromático:
/// bit0 rojo, bit1 azul, 0 = conflicto. Cotas de grado y K_s/K_t nuevo que contenga (a,b).
pub fn allowed_colours(problem: &Problem, graph: &Colouring, a: usize, b: usize) -> u32 {
    let mut allowed = 0;
    for colour in [RED, BLUE] {
        let (ra, rb) = (graph.row(colour, a), graph.row(colour, b));
        let cap = problem.cap_of(colour);
        let blocked = ra.count_ones() >= cap
            || rb.count_ones() >= cap
            || new_clique(problem, graph, a, b, colour);
        if !blocked {
            allowed |= 1 << colour;
        }
    }
    allowed
}

/// El padre no tiene K: uno nuevo contiene (a,b) y un K_{size-2} en N(a)∩N(b).
fn new_clique(problem: &Problem, graph: &Colouring, a: usize, b: usize, colour: u32) -> bool {
    let rows = &graph.rows[colour as usize];
    has_clique(rows, rows[a] & rows[b], problem.clique(colour) - 2)
}

impl Search<'_> {
    fn allowed_colours(&self, a: usize, b: usize) -> u32 {
        allowed_colours(&self.problem, &self.graph, a, b)
    }
    fn new_clique(&self, a: usize, b: usize, colour: u32) -> bool {
        new_clique(&self.problem, &self.graph, a, b, colour)
    }
    fn degrees_viable(&self) -> bool {
        (0..self.problem.n).all(|v| {
            self.graph.degree(RED, v) <= self.problem.red_cap()
                && self.graph.degree(BLUE, v) <= self.problem.blue_cap()
        })
    }
    /// Primera arista libre con 0 o 1 colores permitidos; si no hay, la de más aristas
    /// asignadas en sus extremos, después más vecinos comunes del mismo color.
    fn fail_first_pick(&mut self) -> Option<(usize, usize, u32)> {
        let mut best: Option<((u32, u32), (usize, usize))> = None;
        let free: Vec<_> = self.graph.free_iter().collect();
        for (a, b) in free {
            self.counters.checks += 1;
            let allowed = self.allowed_colours(a, b);
            if allowed != 3 {
                return Some((a, b, allowed));
            }
            let g = &self.graph;
            let common = (g.row(RED, a) & g.row(RED, b)).count_ones()
                + (g.row(BLUE, a) & g.row(BLUE, b)).count_ones();
            let assigned = (g.row(RED, a) | g.row(BLUE, a)).count_ones()
                + (g.row(RED, b) | g.row(BLUE, b)).count_ones();
            if best.map_or(true, |(key, _)| (assigned, common) > key) {
                best = Some(((assigned, common), (a, b)));
            }
        }
        best.map(|(_, (a, b))| (a, b, 3))
    }

    fn run(mut self) -> Result<Report, String> {
        let start = Instant::now();
        let finish = |outcome, counters| Report {
            outcome,
            counters,
            elapsed: start.elapsed(),
        };
        let (n, la) = (self.problem.n, self.config.lookahead);
        if self.graph.n != n {
            return Err(format!("base K{} para un problema K{n}", self.graph.n));
        }
        if self.config.heuristic != Heuristic::Lex && !la {
            return Err("fail-first y las políticas requieren lookahead".into());
        }
        let interval = match self.config.heuristic {
            Heuristic::Neural { interval } => {
                if self.policy.is_none() || interval == 0 {
                    return Err("la política neuronal requiere modelo e intervalo >= 1".into());
                }
                interval
            }
            Heuristic::Random { interval } => {
                if interval == 0 {
                    return Err("el control aleatorio requiere intervalo >= 1".into());
                }
                interval
            }
            _ => 0,
        };
        let neural = interval != 0;
        let base_bad = self.graph.has_clique(RED, self.problem.s)
            || self.graph.has_clique(BLUE, self.problem.t);
        let invalid_degrees = la && !self.degrees_viable();
        if base_bad || invalid_degrees {
            self.counters.prunes += 1;
            if invalid_degrees {
                self.counters.conflicts += 1;
            }
            return Ok(finish(Outcome::Unsat, self.counters));
        }

        let mut journal: Vec<Frame> = Vec::with_capacity(self.graph.free_edges());
        let mut backtrack = false;
        // Primera consulta en la primera elección libre, como el kernel.
        let mut since_policy = interval;
        let mut ready: Option<Option<(usize, usize, u32)>> = None;
        let mut random = u64::from(self.config.seed) << 1 | 1;
        loop {
            if self.counters.transitions >= self.config.budget {
                return Ok(finish(Outcome::Unknown, self.counters));
            }
            self.counters.transitions += 1;
            if backtrack {
                let Some(frame) = journal.pop() else {
                    return Ok(finish(Outcome::Unsat, self.counters));
                };
                let (a, b) = (frame.a, frame.b);
                debug_assert_eq!(self.graph.colour(a, b), Some(frame.colour));
                self.graph.set(a, b, None);
                if frame.forced || frame.alternate_consumed {
                    continue;
                }
                // El color complementario procede del frame; el padre está libre de K.
                let colour = frame.colour ^ 1;
                let rejected = if la {
                    self.counters.checks += 1;
                    self.allowed_colours(a, b) & (1 << colour) == 0
                } else {
                    self.new_clique(a, b, colour)
                };
                if rejected {
                    self.counters.prunes += 1;
                    if la {
                        self.counters.conflicts += 1;
                    }
                    continue;
                }
                if la {
                    self.counters.decisions += 1;
                }
                if neural && since_policy < interval {
                    since_policy += 1;
                }
                self.graph.set(a, b, Some(colour));
                journal.push(Frame {
                    a,
                    b,
                    colour,
                    alternate_consumed: true,
                    forced: false,
                });
                backtrack = false;
                continue;
            }

            let picked = match self.config.heuristic {
                Heuristic::FailFirst => self.fail_first_pick().map(|(a, b, allowed)| {
                    let edge = edge_id(n, a, b) as u32;
                    (a, b, first_colour(self.config.seed, self.config.universe, edge), Some(allowed))
                }),
                Heuristic::Lex | Heuristic::Neural { .. } | Heuristic::Random { .. } => {
                    self.graph.first_free_edge().map(|(mut a, mut b)| {
                        let mut predicted = None;
                        // READY se consume una sola vez; se revalida que la arista siga libre.
                        if let Some(choice) = ready.take() {
                            since_policy = 0;
                            if let Some((ca, cb, colour)) = choice {
                                if self.graph.colour(ca, cb).is_none() {
                                    (a, b, predicted) = (ca, cb, Some(colour));
                                }
                            }
                        }
                        let edge = edge_id(n, a, b) as u32;
                        let colour = predicted.unwrap_or_else(|| {
                            first_colour(self.config.seed, self.config.universe, edge)
                        });
                        let allowed = la.then(|| {
                            self.counters.checks += 1;
                            self.allowed_colours(a, b)
                        });
                        (a, b, colour, allowed)
                    })
                }
            };
            let Some((a, b, mut colour, allowed)) = picked else {
                // Sin aristas libres: comprobación completa e independiente del diario.
                let valid = !self.graph.has_clique(RED, self.problem.s)
                    && !self.graph.has_clique(BLUE, self.problem.t);
                if !valid {
                    return Err("estado completo con un K monocromático".into());
                }
                return Ok(finish(Outcome::Sat(self.graph), self.counters));
            };
            let (mut alternate_consumed, mut forced) = (false, false);
            match allowed {
                Some(0) => {
                    self.counters.prunes += 1;
                    self.counters.conflicts += 1;
                    backtrack = true;
                    continue;
                }
                Some(allowed) => {
                    forced = allowed != 3;
                    if neural && !forced && since_policy >= interval {
                        // El kernel suspende aquí sin asignar; la transición ya contó.
                        self.counters.policy_queries += 1;
                        let depth = journal.len();
                        let choice = if let Heuristic::Random { .. } = self.config.heuristic {
                            random ^= random << 13;
                            random ^= random >> 7;
                            random ^= random << 17;
                            let candidates: Vec<_> = self.graph.free_iter().take(CANDIDATES).collect();
                            let (a, b) = candidates[(random >> 1) as usize % candidates.len()];
                            Some((a, b, (random & 1) as u32))
                        } else {
                            self.policy.and_then(|p| p.choose(&self.problem, &self.graph, depth))
                        };
                        if choice.is_none() {
                            self.counters.policy_fallbacks += 1;
                        }
                        ready = Some(choice);
                        continue;
                    }
                    if forced {
                        colour = if allowed == 1 { RED } else { BLUE };
                        alternate_consumed = true;
                        self.counters.forced += 1;
                    } else {
                        self.counters.decisions += 1;
                        if neural && since_policy < interval {
                            since_policy += 1;
                        }
                    }
                }
                None => {
                    if self.new_clique(a, b, colour) {
                        // Rechazar antes de escribir; el segundo color se prueba exactamente.
                        self.counters.prunes += 1;
                        colour ^= 1;
                        alternate_consumed = true;
                        if self.new_clique(a, b, colour) {
                            self.counters.prunes += 1;
                            backtrack = true;
                            continue;
                        }
                    }
                }
            }
            self.graph.set(a, b, Some(colour));
            journal.push(Frame {
                a,
                b,
                colour,
                alternate_consumed,
                forced,
            });
        }
    }
}

/// DFS completo desde `base`. `Unsat` refuta sólo las extensiones de `base`.
pub fn solve(
    problem: Problem,
    base: &Colouring,
    config: Config,
    policy: Option<&Policy>,
) -> Result<Report, String> {
    Search {
        problem,
        config,
        policy,
        graph: base.clone(),
        counters: Counters::default(),
    }
    .run()
}

/// Una ejecución por clase de `star_degrees`, como `--canonical-stars` en K43. SAT en
/// cuanto una clase lo da; UNSAT sólo si todas lo dan (sin clases: UNSAT por cotas).
pub fn solve_stars(
    problem: Problem,
    config: Config,
    policy: Option<&Policy>,
) -> Result<Report, String> {
    let mut counters = Counters::default();
    let mut elapsed = Duration::ZERO;
    let mut outcome = Outcome::Unsat;
    for degree in problem.star_degrees() {
        let report = solve(problem, &problem.canonical_star(degree), config, policy)?;
        counters.add(&report.counters);
        elapsed += report.elapsed;
        match report.outcome {
            Outcome::Sat(model) => {
                outcome = Outcome::Sat(model);
                break;
            }
            Outcome::Unknown => outcome = Outcome::Unknown,
            Outcome::Unsat => {}
        }
    }
    Ok(Report {
        outcome,
        counters,
        elapsed,
    })
}

/// Estimación de Knuth del árbol de `solve`: nodos = estados con una arista asignada más la
/// raíz. Agotar un árbol de N nodos cuesta exactamente 2N transiciones sin política.
#[derive(Clone, Copy, Debug)]
pub struct Estimate {
    pub nodes: f64,
    pub std_error: f64,
    pub probes: u64,
}

/// Knuth (1975): cada sonda baja por un hijo uniforme y suma el producto de ramificaciones
/// de cada nivel; la media es insesgada. Sólo para selección que depende únicamente del
/// estado (lex, fail-first); la política neuronal depende de su contador de decisiones.
pub fn knuth_estimate(
    problem: Problem,
    base: &Colouring,
    heuristic: Heuristic,
    lookahead: bool,
    probes: u64,
    seed: u64,
) -> Result<Estimate, String> {
    match heuristic {
        Heuristic::Neural { .. } | Heuristic::Random { .. } => {
            return Err("las políticas dependen de la historia".into())
        }
        Heuristic::FailFirst if !lookahead => return Err("fail-first requiere lookahead".into()),
        _ => {}
    }
    let mut search = Search {
        problem,
        config: Config::new(heuristic, lookahead, 0),
        policy: None,
        graph: base.clone(),
        counters: Counters::default(),
    };
    let viable = !search.graph.has_clique(RED, problem.s)
        && !search.graph.has_clique(BLUE, problem.t)
        && (!lookahead || search.degrees_viable());
    let mut state = seed | 1;
    let (mut sum, mut sum_sq) = (0.0, 0.0);
    for _ in 0..probes {
        search.graph = base.clone();
        let (mut weight, mut total) = (1.0_f64, if viable { 1.0 } else { 0.0 });
        while viable {
            let picked = match heuristic {
                Heuristic::FailFirst => search.fail_first_pick(),
                _ => search.graph.first_free_edge().map(|(a, b)| {
                    let allowed = if lookahead {
                        search.allowed_colours(a, b)
                    } else {
                        [RED, BLUE]
                            .into_iter()
                            .filter(|&c| !search.new_clique(a, b, c))
                            .fold(0, |mask, c| mask | 1 << c)
                    };
                    (a, b, allowed)
                }),
            };
            let Some((a, b, allowed)) = picked else { break };
            let children = allowed.count_ones();
            if children == 0 {
                break;
            }
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let colour = if children == 2 { (state & 1) as u32 } else { allowed.trailing_zeros() };
            weight *= f64::from(children);
            total += weight;
            search.graph.set(a, b, Some(colour));
        }
        sum += total;
        sum_sq += total * total;
    }
    let mean = sum / probes as f64;
    let variance = (sum_sq / probes as f64 - mean * mean).max(0.0);
    Ok(Estimate {
        nodes: mean,
        std_error: (variance / probes as f64).sqrt(),
        probes,
    })
}

/// Grafo de Paley de orden primo q ≡ 1 (mod 4): rojo si a-b es residuo cuadrático.
pub fn paley(q: usize) -> Colouring {
    assert!(q <= MAX_VERTICES && q % 4 == 1, "q primo ≡ 1 (mod 4)");
    let residues: Vec<bool> = (0..q).map(|x| (1..q).any(|y| y * y % q == x)).collect();
    let mut graph = Colouring::empty(q);
    for a in 0..q {
        for b in a + 1..q {
            graph.set(a, b, Some(if residues[b - a] { RED } else { BLUE }));
        }
    }
    graph
}

/// MLP 32→32→32→2 de `neuro_ramsey_policy.onnx` evaluado en f32. Salida softplus =
/// coste {rojo, azul}; el kernel elige el mínimo.
pub struct Policy {
    layers: [(Vec<f32>, Vec<f32>); 3],
}

const FEATURES: usize = 32;
const CANDIDATES: usize = 32;

fn read_varint(bytes: &[u8], at: &mut usize) -> Result<u64, String> {
    let mut value = 0_u64;
    for shift in (0..64).step_by(7) {
        let byte = *bytes.get(*at).ok_or("protobuf truncado")?;
        *at += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            return Ok(value);
        }
    }
    Err("varint demasiado largo".into())
}

enum Field<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
    Fixed,
}

fn fields(bytes: &[u8]) -> Result<Vec<(u64, Field<'_>)>, String> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let key = read_varint(bytes, &mut at)?;
        let value = match key & 7 {
            0 => Field::Varint(read_varint(bytes, &mut at)?),
            1 | 5 => {
                at += if key & 7 == 1 { 8 } else { 4 };
                Field::Fixed
            }
            2 => {
                let len = read_varint(bytes, &mut at)? as usize;
                let slice = bytes.get(at..at + len).ok_or("protobuf truncado")?;
                at += len;
                Field::Bytes(slice)
            }
            wire => return Err(format!("tipo protobuf {wire} no soportado")),
        };
        out.push((key >> 3, value));
    }
    Ok(out)
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

impl Policy {
    /// Lector mínimo de ONNX: exige Gemm(transB=1),Relu,Gemm,Relu,Gemm,Softplus y los
    /// inicializadores 0/2/4.{weight,bias} con las formas de la red exportada.
    pub fn from_onnx(bytes: &[u8]) -> Result<Self, String> {
        let graph = fields(bytes)?
            .into_iter()
            .find_map(|(f, v)| match (f, v) {
                (7, Field::Bytes(g)) => Some(g),
                _ => None,
            })
            .ok_or("ONNX sin grafo")?;
        let mut ops = Vec::new();
        let mut tensors = std::collections::HashMap::new();
        for (field, value) in fields(graph)? {
            let Field::Bytes(body) = value else { continue };
            match field {
                1 => {
                    let mut op = String::new();
                    let mut trans_b = None;
                    for (f, v) in fields(body)? {
                        match (f, v) {
                            (4, Field::Bytes(s)) => op = String::from_utf8_lossy(s).into(),
                            (5, Field::Bytes(attr)) => {
                                let attr = fields(attr)?;
                                let named = attr.iter().any(|(f, v)| {
                                    matches!((f, v), (1, Field::Bytes(b"transB")))
                                });
                                if named {
                                    trans_b = attr.iter().find_map(|(f, v)| match (f, v) {
                                        (3, Field::Varint(i)) => Some(*i),
                                        _ => None,
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                    if op == "Gemm" && trans_b != Some(1) {
                        return Err("Gemm sin transB=1".into());
                    }
                    ops.push(op);
                }
                5 => {
                    let (mut name, mut dims, mut data) = (String::new(), Vec::new(), Vec::new());
                    for (f, v) in fields(body)? {
                        match (f, v) {
                            (1, Field::Varint(d)) => dims.push(d),
                            (1, Field::Bytes(packed)) => {
                                let mut at = 0;
                                while at < packed.len() {
                                    dims.push(read_varint(packed, &mut at)?);
                                }
                            }
                            (2, Field::Varint(t)) if t != 1 => {
                                return Err("inicializador no FLOAT".into())
                            }
                            (4 | 9, Field::Bytes(raw)) => data = floats(raw),
                            (8, Field::Bytes(s)) => name = String::from_utf8_lossy(s).into(),
                            _ => {}
                        }
                    }
                    tensors.insert(name, (dims, data));
                }
                _ => {}
            }
        }
        if ops != ["Gemm", "Relu", "Gemm", "Relu", "Gemm", "Softplus"] {
            return Err(format!("topología inesperada: {ops:?}"));
        }
        let mut take = |name: &str, dims: &[u64]| -> Result<Vec<f32>, String> {
            let (shape, data) = tensors.remove(name).ok_or(format!("falta {name}"))?;
            let len = dims.iter().product::<u64>() as usize;
            if shape != dims || data.len() != len {
                return Err(format!("{name}: forma {shape:?}, se esperaba {dims:?}"));
            }
            Ok(data)
        };
        let width = FEATURES as u64;
        Ok(Self {
            layers: [
                (take("0.weight", &[width, width])?, take("0.bias", &[width])?),
                (take("2.weight", &[width, width])?, take("2.bias", &[width])?),
                (take("4.weight", &[2, width])?, take("4.bias", &[2])?),
            ],
        })
    }

    /// Costes {rojo, azul} de una fila de features.
    pub fn costs(&self, features: &[f32; FEATURES]) -> [f32; 2] {
        let mut x = features.to_vec();
        for (index, (weight, bias)) in self.layers.iter().enumerate() {
            x = bias
                .iter()
                .zip(weight.chunks_exact(x.len()))
                .map(|(b, row)| {
                    let y = b + row.iter().zip(&x).map(|(w, v)| w * v).sum::<f32>();
                    if index < 2 {
                        y.max(0.0)
                    } else {
                        y.max(0.0) + (-y.abs()).exp().ln_1p()
                    }
                })
                .collect();
        }
        [x[0], x[1]]
    }

    /// `ramsey43_pack_features` + MLP + `ramsey43_apply_scores`: (arista, color) de menor
    /// coste entre las 32 primeras aristas libres; empates por arista y después rojo.
    /// `None` = fallback (coste no finito). Normaliza por |E| y n-1 en lugar de 903 y 42.
    pub fn choose(&self, problem: &Problem, graph: &Colouring, depth: usize) -> Option<(usize, usize, u32)> {
        let n = problem.n;
        let edges = problem.edges() as f32;
        let span = (n - 1) as f32;
        let degrees = |colour| (0..n).map(move |v| graph.degree(colour, v));
        let red_edges = degrees(RED).sum::<u32>() / 2;
        let blue_edges = degrees(BLUE).sum::<u32>() / 2;
        let unknown_edges = problem.edges() as u32 - red_edges - blue_edges;
        let globals = [
            red_edges as f32 / edges,
            blue_edges as f32 / edges,
            unknown_edges as f32 / edges,
            depth as f32 / edges,
            degrees(RED).min()? as f32 / span,
            degrees(RED).max()? as f32 / span,
            degrees(BLUE).min()? as f32 / span,
            degrees(BLUE).max()? as f32 / span,
        ];
        let mut best: Option<(f32, (usize, usize, u32))> = None;
        for (a, b) in graph.free_iter().take(CANDIDATES) {
            let (ra, rb) = (graph.row(RED, a), graph.row(RED, b));
            let (ba, bb) = (graph.row(BLUE, a), graph.row(BLUE, b));
            let (ua, ub) = (graph.unknown(a), graph.unknown(b));
            let pop = |x: u64| x.count_ones() as f32;
            let pair = |x: u64, y: u64| [pop(x).min(pop(y)) / span, pop(x).max(pop(y)) / span];
            let mut x = [0.0_f32; FEATURES];
            x[..8].copy_from_slice(&globals);
            x[8..10].copy_from_slice(&pair(ra, rb));
            x[10..12].copy_from_slice(&pair(ba, bb));
            x[12..14].copy_from_slice(&pair(ua, ub));
            x[14] = pop(ra & rb) / span;
            x[15] = pop(ba & bb) / span;
            x[16] = pop(ua & ub) / span;
            x[17] = (pop(ra & bb) + pop(ba & rb)) / (2.0 * span);
            x[18] = (pop(ra & ub) + pop(ua & rb)) / (2.0 * span);
            x[19] = (pop(ba & ub) + pop(ua & bb)) / (2.0 * span);
            // El kernel entrega binary16: features normales, redondeo al par más cercano.
            for v in &mut x {
                let bits = v.to_bits();
                *v = f32::from_bits((bits + 0xfff + (bits >> 13 & 1)) & !0x1fff);
            }
            let [red, blue] = self.costs(&x);
            if !red.is_finite() || !blue.is_finite() {
                return None;
            }
            let (score, colour) = if blue < red { (blue, BLUE) } else { (red, RED) };
            if best.map_or(true, |(s, _)| score < s) {
                best = Some((score, (a, b, colour)));
            }
        }
        best.map(|(_, choice)| choice)
    }
}
