//! Solver nativo de Planos de Corte Pseudo-Booleanos (estilo RoundingSat) con
//! certificación directa VeriPB 3.0.2 (`pol` + `d` + `s` + `soli` + `conclusion BOUNDS`).
//!
//! Opera directamente sobre restricciones lineales $\sum a_i \ell_i \ge d$ en `u128` sin
//! bit-blasting a CNF: resuelve instancias exponenciales para resolución pura (ej. Palomar
//! sin simetría, mochilas con pesos grandes, sumas exactas) y certifica directamente contra
//! el `.opb` original con `veripb --opb`.

use crate::opb::{self, Problem};
use std::io::{self, BufWriter, Write};
use std::sync::atomic::{AtomicBool, Ordering};

const NO_REASON: usize = usize::MAX;
/// Límite seguro de coeficiente durante análisis para impedir desbordamiento en `u128`.
const COEFF_SOFT_CAP: u128 = 1_u128 << 58;
const INIT_CLEANUP_CONFLICTS: u64 = 2000;

#[derive(Clone, Debug)]
enum PolOp {
    Id(u64),
    Lit(u32),
    Add,
    Mul(u128),
    Div(u128),
    Sat,
}

#[derive(Clone, Debug)]
struct PbConstraint {
    /// Términos `(coeficiente > 0, lit)` ordenados descendentemente por coeficiente.
    terms: Vec<(u128, u32)>,
    degree: u128,
    slack: i128,
    proof_id: u64,
    learned: bool,
    lbd: u32,
    used: bool,
    deleted: bool,
}

/// Estadísticas del solver de planos de corte.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpStats {
    pub conflicts: u64,
    pub decisions: u64,
    pub propagations: u64,
    pub restarts: u64,
    pub learned: u64,
    pub deleted: u64,
    pub proof_lines: u64,
}

#[derive(Clone, Debug)]
pub enum CpOutcome {
    Sat(Vec<bool>),
    Unsat,
    Optimum { model: Vec<bool>, cost: i128 },
    Stopped { best: Option<(Vec<bool>, i128)> },
}

/// Escritor de trazas VeriPB 3.0.2 nativas para OPB (`pol`, `soli`, `del id`, `conclusion`).
pub struct OpbProofWriter<W: Write> {
    out: BufWriter<W>,
    next_id: u64,
    lines: u64,
    error: Option<io::Error>,
}

impl<W: Write> OpbProofWriter<W> {
    pub fn new(sink: W, input_constraints: usize) -> Self {
        let mut writer = Self {
            out: BufWriter::with_capacity(1 << 16, sink),
            next_id: input_constraints as u64 + 1,
            lines: 0,
            error: None,
        };
        writer.write_str("pseudo-Boolean proof version 3.0\nf ");
        writer.write_str(&input_constraints.to_string());
        writer.write_str(";\n");
        writer
    }

    fn write_str(&mut self, s: &str) {
        if self.error.is_none() {
            if let Err(e) = self.out.write_all(s.as_bytes()) {
                self.error = Some(e);
            }
        }
    }

    fn emit_pol(&mut self, ops: &[PolOp]) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.lines += 1;
        self.write_str("pol");
        for op in ops {
            self.write_str(" ");
            match *op {
                PolOp::Id(cid) => self.write_str(&cid.to_string()),
                PolOp::Lit(lit) => {
                    if lit & 1 == 1 {
                        self.write_str("~x");
                    } else {
                        self.write_str("x");
                    }
                    self.write_str(&((lit >> 1) + 1).to_string());
                }
                PolOp::Add => self.write_str("+"),
                PolOp::Mul(m) => {
                    self.write_str(&m.to_string());
                    self.write_str(" *");
                }
                PolOp::Div(d) => {
                    self.write_str(&d.to_string());
                    self.write_str(" d");
                }
                PolOp::Sat => self.write_str("s"),
            }
        }
        // VeriPB exige al menos un operador si solo hay un ID en la pila.
        if ops.len() == 1 {
            self.write_str(" 1 *");
        }
        self.write_str(";\n");
        id
    }

    fn emit_soli(&mut self, model: &[bool], vars: usize) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.lines += 1;
        self.write_str("soli");
        for v in 1..=vars {
            if model.get(v).copied().unwrap_or(false) {
                self.write_str(" x");
            } else {
                self.write_str(" ~x");
            }
            self.write_str(&v.to_string());
        }
        self.write_str(";\n");
        id
    }

    fn emit_delete_ids(&mut self, ids: &[u64]) {
        for chunk in ids.chunks(64) {
            self.lines += 1;
            self.write_str("del id");
            for &id in chunk {
                self.write_str(" ");
                self.write_str(&id.to_string());
            }
            self.write_str(";\n");
        }
    }

    pub fn lines(&self) -> u64 {
        self.lines
    }

    pub fn finish(mut self, has_objective: bool, optimum: Option<i128>, refuted: bool) -> io::Result<W> {
        self.write_str("output NONE;\n");
        if refuted {
            match (has_objective, optimum) {
                (false, _) => self.write_str("conclusion UNSAT : -1;\n"),
                (true, None) => self.write_str("conclusion BOUNDS INF : -1 INF;\n"),
                (true, Some(cost)) => {
                    self.write_str(&format!("conclusion BOUNDS {cost} : -1 {cost};\n"));
                }
            }
        } else {
            self.write_str("conclusion NONE;\n");
        }
        self.write_str("end pseudo-Boolean proof;\n");
        if let Some(err) = self.error {
            return Err(err);
        }
        self.out.flush()?;
        self.out.into_inner().map_err(|e| e.into_error())
    }
}

/// Normaliza `sum a_i * lit_i >= degree` a términos con coeficientes positivos `u128` sobre
/// literales distintos (`lit = ((var - 1) << 1) | neg`) y grado `u128`.
/// Devuelve `(terms, degree, max_coeff_exceeded_degree)`.
fn normalize_raw(terms: &[(i64, i32)], degree: i128, vars: usize) -> (Vec<(u128, u32)>, u128, bool) {
    // Acumular peso neto sobre la variable positiva x_v: a*(~x_v) = a - a*x_v.
    let mut net: Vec<(usize, i128)> = Vec::with_capacity(terms.len());
    let mut rhs = degree;
    for &(coeff, lit) in terms {
        if coeff == 0 {
            continue;
        }
        let v = lit.unsigned_abs() as usize - 1;
        debug_assert!(v < vars);
        let c = i128::from(coeff);
        if lit > 0 {
            net.push((v, c));
        } else {
            net.push((v, -c));
            rhs -= c;
        }
    }
    net.sort_unstable_by_key(|&(v, _)| v);
    let mut merged: Vec<(usize, i128)> = Vec::with_capacity(net.len());
    for (v, c) in net {
        if let Some(last) = merged.last_mut() {
            if last.0 == v {
                last.1 += c;
                continue;
            }
        }
        merged.push((v, c));
    }
    let mut out: Vec<(u128, u32)> = Vec::with_capacity(merged.len());
    for (v, c) in merged {
        if c == 0 {
            continue;
        }
        if c > 0 {
            out.push((c as u128, (v as u32) << 1));
        } else {
            let mag = c.unsigned_abs();
            rhs += c.abs(); // -mag * x_v = -mag + mag * ~x_v  =>  rhs += mag
            out.push((mag, ((v as u32) << 1) | 1));
        }
    }
    if rhs <= 0 {
        return (Vec::new(), 0, false);
    }
    let deg = rhs as u128;
    let mut needs_sat = false;
    for (w, _) in &mut out {
        if *w > deg {
            *w = deg;
            needs_sat = true;
        }
    }
    out.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    (out, deg, needs_sat)
}

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn ceil_div(a: u128, b: u128) -> u128 {
    debug_assert!(b > 0);
    if a == 0 { 0 } else { (a - 1) / b + 1 }
}

/// Restricción de trabajo densa usada durante el análisis de conflictos por Planos de Corte.
struct WorkingConstraint {
    /// Peso con signo por variable: `> 0` => `+w * x_v`, `< 0` => `|w| * ~x_v`, `0` => ausente.
    coeffs: Vec<i128>,
    active: Vec<usize>,
    degree: u128,
    ops: Vec<PolOp>,
    trace: bool,
}

impl WorkingConstraint {
    fn new(vars: usize, trace: bool) -> Self {
        Self {
            coeffs: vec![0; vars],
            active: Vec::new(),
            degree: 0,
            ops: Vec::new(),
            trace,
        }
    }

    fn clear(&mut self) {
        for &v in &self.active {
            self.coeffs[v] = 0;
        }
        self.active.clear();
        self.degree = 0;
        self.ops.clear();
    }

    fn load(&mut self, c: &PbConstraint) {
        self.clear();
        self.degree = c.degree;
        for &(w, lit) in &c.terms {
            let v = (lit >> 1) as usize;
            self.coeffs[v] = if lit & 1 == 0 { w as i128 } else { -(w as i128) };
            self.active.push(v);
        }
        if self.trace {
            self.ops.push(PolOp::Id(c.proof_id));
        }
    }

    #[inline]
    fn lit_of(&self, v: usize) -> Option<(u128, u32)> {
        let c = self.coeffs[v];
        if c == 0 {
            None
        } else if c > 0 {
            Some((c as u128, (v as u32) << 1))
        } else {
            Some((c.unsigned_abs(), ((v as u32) << 1) | 1))
        }
    }

    fn max_coeff(&self) -> u128 {
        self.active.iter().map(|&v| self.coeffs[v].unsigned_abs()).max().unwrap_or(0)
    }

    /// Calcula el slack bajo la asignación parcial `trail[0..limit]` (todo lo demás es libre).
    fn slack_at(&self, vals: &[i8], assign_pos: &[usize], limit: usize) -> i128 {
        let mut lhs: i128 = 0;
        for &v in &self.active {
            if let Some((w, lit)) = self.lit_of(v) {
                let falsified = vals[lit as usize] == -1 && assign_pos[v] < limit;
                if !falsified {
                    lhs += w as i128;
                }
            }
        }
        lhs - self.degree as i128
    }

    /// Debilita el literal en variable `v` en `amount` (`0 < amount <= |coeffs[v]|`).
    fn weaken_var(&mut self, v: usize, amount: u128) {
        let c = self.coeffs[v];
        debug_assert!(c != 0 && amount > 0 && amount <= c.unsigned_abs());
        let lit = if c > 0 { (v as u32) << 1 } else { ((v as u32) << 1) | 1 };
        let new_mag = c.unsigned_abs() - amount;
        self.coeffs[v] = if new_mag == 0 {
            0
        } else if c > 0 {
            new_mag as i128
        } else {
            -(new_mag as i128)
        };
        self.degree = self.degree.saturating_sub(amount);
        if self.trace {
            self.ops.push(PolOp::Lit(lit ^ 1));
            if amount > 1 {
                self.ops.push(PolOp::Mul(amount));
            }
            self.ops.push(PolOp::Add);
        }
    }

    fn compact_active(&mut self) {
        self.active.retain(|&v| self.coeffs[v] != 0);
    }

    fn saturate(&mut self) {
        let mut changed = false;
        let deg = self.degree;
        for &v in &self.active {
            let c = self.coeffs[v];
            if c.unsigned_abs() > deg {
                self.coeffs[v] = if c > 0 { deg as i128 } else { -(deg as i128) };
                changed = true;
            }
        }
        if changed && self.trace {
            self.ops.push(PolOp::Sat);
        }
    }

    fn divide(&mut self, div: u128) {
        if div <= 1 {
            return;
        }
        for &v in &self.active {
            let c = self.coeffs[v];
            if c != 0 {
                let mag = ceil_div(c.unsigned_abs(), div);
                self.coeffs[v] = if c > 0 { mag as i128 } else { -(mag as i128) };
            }
        }
        self.degree = ceil_div(self.degree, div);
        if self.trace {
            self.ops.push(PolOp::Div(div));
        }
    }

    /// Reduce coeficientes grandes dividiendo por `div`, debilitando antes los literales
    /// no falsificados módulo `div` para garantizar que la contradicción bajo `trail[0..limit]` se conserva.
    fn shrink_preserving_conflict(&mut self, div: u128, vals: &[i8], assign_pos: &[usize], limit: usize) {
        if div <= 1 {
            return;
        }
        self.compact_active();
        for idx in 0..self.active.len() {
            let v = self.active[idx];
            if let Some((w, lit)) = self.lit_of(v) {
                let falsified = vals[lit as usize] == -1 && assign_pos[v] < limit;
                if !falsified {
                    let rem = w % div;
                    if rem != 0 {
                        self.weaken_var(v, rem);
                    }
                }
            }
        }
        self.compact_active();
        self.divide(div);
        self.saturate();
    }

    /// Combina `self = self + mult * R'` donde `R'` es la razón `reason` debilitada y dividida por `div`.
    fn resolve_with_reason(
        &mut self,
        reason: &PbConstraint,
        pivot_lit: u32,
        pivot_coeff_in_w: u128,
        pivot_coeff_in_r: u128,
        vals: &[i8],
        assign_pos: &[usize],
        limit: usize,
    ) {
        let b = pivot_coeff_in_r;
        // 1) Construir R' debilitando literales no falsificados antes de `limit` (salvo `pivot_lit`)
        //    módulo `b`, luego dividiendo por `b` y saturando.
        let mut r_terms: Vec<(u128, u32)> = Vec::with_capacity(reason.terms.len());
        let mut r_deg = reason.degree;
        let mut r_weaken: Vec<(u32, u128)> = Vec::new();
        for &(w, lit) in &reason.terms {
            if lit == pivot_lit {
                r_terms.push((w, lit));
                continue;
            }
            let v = (lit >> 1) as usize;
            let falsified = vals[lit as usize] == -1 && assign_pos[v] < limit;
            if !falsified && b > 1 {
                let rem = w % b;
                if rem != 0 {
                    r_weaken.push((lit ^ 1, rem));
                    r_deg = r_deg.saturating_sub(rem);
                    let remain = w - rem;
                    if remain > 0 {
                        r_terms.push((remain, lit));
                    }
                    continue;
                }
            }
            r_terms.push((w, lit));
        }
        if b > 1 {
            for (w, _) in &mut r_terms {
                *w = ceil_div(*w, b);
            }
            r_deg = ceil_div(r_deg, b);
        }
        // Protección anti-overflow en `u128` antes de sumar `a * R'`.
        let max_r = r_terms.iter().map(|&(w, _)| w).max().unwrap_or(1).max(r_deg);
        let mut a = pivot_coeff_in_w;
        if a.saturating_mul(max_r) > (1_u128 << 110) || self.max_coeff() > COEFF_SOFT_CAP {
            let div = (a >> 48).max(self.max_coeff() >> 52).max(2);
            self.shrink_preserving_conflict(div, vals, assign_pos, limit);
            let pv = (pivot_lit >> 1) as usize;
            a = self.coeffs[pv].unsigned_abs();
            if a == 0 {
                return;
            }
        }
        // 2) Emitir `R'` en `ops` y multiplicar por `a`.
        if self.trace {
            self.ops.push(PolOp::Id(reason.proof_id));
            for (wlit, amt) in r_weaken {
                self.ops.push(PolOp::Lit(wlit));
                if amt > 1 {
                    self.ops.push(PolOp::Mul(amt));
                }
                self.ops.push(PolOp::Add);
            }
            if b > 1 {
                self.ops.push(PolOp::Div(b));
            }
            if a > 1 {
                self.ops.push(PolOp::Mul(a));
            }
            self.ops.push(PolOp::Add);
        }
        // 3) Sumar `a * R'` en `self` con cancelación exacta `k*x + k*~x = k`.
        let mut new_deg = self.degree as i128 + (a * r_deg) as i128;
        for (w_r, lit) in r_terms {
            let add_mag = a * w_r;
            let v = (lit >> 1) as usize;
            let cur = self.coeffs[v];
            if cur == 0 {
                self.coeffs[v] = if lit & 1 == 0 { add_mag as i128 } else { -(add_mag as i128) };
                self.active.push(v);
            } else {
                let cur_neg = cur < 0;
                let r_neg = lit & 1 == 1;
                let cur_mag = cur.unsigned_abs();
                if cur_neg == r_neg {
                    let sum = cur_mag + add_mag;
                    self.coeffs[v] = if cur_neg { -(sum as i128) } else { sum as i128 };
                } else {
                    let cancel = cur_mag.min(add_mag);
                    new_deg -= cancel as i128;
                    if cur_mag > add_mag {
                        let rem = cur_mag - add_mag;
                        self.coeffs[v] = if cur_neg { -(rem as i128) } else { rem as i128 };
                    } else if add_mag > cur_mag {
                        let rem = add_mag - cur_mag;
                        self.coeffs[v] = if r_neg { -(rem as i128) } else { rem as i128 };
                    } else {
                        self.coeffs[v] = 0;
                    }
                }
            }
        }
        self.degree = new_deg.max(0) as u128;
        self.compact_active();
        self.saturate();
    }

    /// Debilita todos los literales restantes cuando `sum(coeffs) < degree` para obtener `0 >= k`.
    fn weaken_all_to_contradiction(&mut self) {
        self.compact_active();
        for idx in 0..self.active.len() {
            let v = self.active[idx];
            let mag = self.coeffs[v].unsigned_abs();
            if mag > 0 {
                self.weaken_var(v, mag);
            }
        }
        self.compact_active();
    }

    /// Simplifica por GCD de coeficientes (fortalece `degree = ceil(degree / g)`).
    fn simplify_gcd(&mut self) {
        self.compact_active();
        if self.active.is_empty() {
            return;
        }
        let mut g = 0_u128;
        for &v in &self.active {
            g = gcd_u128(g, self.coeffs[v].unsigned_abs());
            if g == 1 {
                return;
            }
        }
        if g > 1 {
            self.divide(g);
            self.saturate();
        }
    }

    fn export_terms(&self) -> Vec<(u128, u32)> {
        let mut terms: Vec<(u128, u32)> = self.active.iter()
            .filter_map(|&v| self.lit_of(v))
            .collect();
        terms.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        terms
    }
}

struct CpSolver {
    vars: usize,
    constraints: Vec<PbConstraint>,
    /// Lista de ocurrencias por literal: `(cid, coeff)` donde aparece ese literal.
    occ: Vec<Vec<(usize, u128)>>,
    vals: Vec<i8>,
    levels: Vec<usize>,
    reasons: Vec<usize>,
    assign_pos: Vec<usize>,
    phases: Vec<u8>,
    scores: Vec<f64>,
    var_inc: f64,
    trail: Vec<u32>,
    trail_limits: Vec<usize>,
    head: usize,
    root_unsat: bool,
    stats: CpStats,
    luby: (u64, u64),
    next_restart: u64,
    next_cleanup: u64,
    cleanup_step: u64,
    learned_count: usize,
    work: WorkingConstraint,
}

impl CpSolver {
    fn new(vars: usize, trace: bool) -> Self {
        Self {
            vars,
            constraints: Vec::new(),
            occ: vec![Vec::new(); 2 * vars],
            vals: vec![0; 2 * vars],
            levels: vec![0; vars],
            reasons: vec![NO_REASON; vars],
            assign_pos: vec![usize::MAX; vars],
            phases: vec![1; vars], // Por defecto falso (favorece minimización de objetivos positivos).
            scores: vec![0.0; vars],
            var_inc: 1.0,
            trail: Vec::with_capacity(vars),
            trail_limits: Vec::new(),
            head: 0,
            root_unsat: false,
            stats: CpStats::default(),
            luby: (1, 1),
            next_restart: 100,
            next_cleanup: INIT_CLEANUP_CONFLICTS,
            cleanup_step: INIT_CLEANUP_CONFLICTS,
            learned_count: 0,
            work: WorkingConstraint::new(vars, trace),
        }
    }

    #[inline]
    fn level(&self) -> usize {
        self.trail_limits.len()
    }

    fn assign(&mut self, lit: u32, reason: usize) {
        let v = (lit >> 1) as usize;
        debug_assert_eq!(self.vals[lit as usize], 0);
        self.vals[lit as usize] = 1;
        self.vals[(lit ^ 1) as usize] = -1;
        self.levels[v] = self.level();
        self.reasons[v] = reason;
        self.assign_pos[v] = self.trail.len();
        self.phases[v] = (lit & 1) as u8;
        self.trail.push(lit);
    }

    fn backjump(&mut self, target_level: usize) {
        if target_level >= self.level() {
            return;
        }
        let keep = self.trail_limits[target_level];
        // Restaurar slacks de literales que ya habían sido propagados (`keep..self.head`).
        let propagated_end = self.head.min(self.trail.len());
        for idx in keep..propagated_end {
            let neg_lit = self.trail[idx] ^ 1;
            for &(cid, w) in &self.occ[neg_lit as usize] {
                if !self.constraints[cid].deleted {
                    self.constraints[cid].slack += w as i128;
                }
            }
        }
        for idx in keep..self.trail.len() {
            let lit = self.trail[idx];
            let v = (lit >> 1) as usize;
            self.vals[lit as usize] = 0;
            self.vals[(lit ^ 1) as usize] = 0;
            self.reasons[v] = NO_REASON;
            self.assign_pos[v] = usize::MAX;
        }
        self.trail.truncate(keep);
        self.trail_limits.truncate(target_level);
        self.head = keep;
    }

    fn bump_var(&mut self, v: usize) {
        self.scores[v] += self.var_inc;
        if self.scores[v] > 1e100 {
            for s in &mut self.scores {
                *s *= 1e-100;
            }
            self.var_inc *= 1e-100;
        }
    }

    fn decay_scores(&mut self) {
        self.var_inc /= 0.95;
    }

    /// Añade una restricción PB ya normalizada y saturada al solver.
    /// Devuelve `Some(cid)` si entra inmediatamente en conflicto.
    fn add_constraint(
        &mut self,
        terms: Vec<(u128, u32)>,
        degree: u128,
        proof_id: u64,
        learned: bool,
        lbd: u32,
    ) -> Option<usize> {
        if degree == 0 {
            return None;
        }
        // Calcular slack bajo los literales YA propagados (`0..self.head`).
        let mut lhs: i128 = 0;
        for &(w, lit) in &terms {
            let v = (lit >> 1) as usize;
            let falsified_propagated = self.vals[lit as usize] == -1 && self.assign_pos[v] < self.head;
            if !falsified_propagated {
                lhs += w as i128;
            }
        }
        let slack = lhs - degree as i128;
        let cid = self.constraints.len();
        for &(w, lit) in &terms {
            self.occ[lit as usize].push((cid, w));
        }
        self.constraints.push(PbConstraint {
            terms,
            degree,
            slack,
            proof_id,
            learned,
            lbd,
            used: true,
            deleted: false,
        });
        if learned {
            self.learned_count += 1;
            self.stats.learned += 1;
        }
        if slack < 0 {
            return Some(cid);
        }
        // Comprobar si la nueva restricción ya dispara propagaciones en el estado actual.
        let max_c = self.constraints[cid].terms.first().map_or(0, |&(w, _)| w);
        if (max_c as i128) > slack {
            let len = self.constraints[cid].terms.len();
            for i in 0..len {
                let (w, lit) = self.constraints[cid].terms[i];
                if (w as i128) <= slack {
                    break;
                }
                if self.vals[lit as usize] == 0 {
                    self.assign(lit, cid);
                }
            }
        }
        None
    }

    fn propagate(&mut self) -> Option<usize> {
        while self.head < self.trail.len() {
            let lit = self.trail[self.head];
            self.head += 1;
            self.stats.propagations += 1;
            let falsified = lit ^ 1;
            let occ_len = self.occ[falsified as usize].len();
            let mut conflict: Option<usize> = None;
            for idx in 0..occ_len {
                let (cid, w) = self.occ[falsified as usize][idx];
                if self.constraints[cid].deleted {
                    continue;
                }
                self.constraints[cid].slack -= w as i128;
                let slack = self.constraints[cid].slack;
                if slack < 0 {
                    if conflict.is_none() {
                        conflict = Some(cid);
                    }
                    // Seguimos actualizando el resto de ocurrencias de `falsified` para mantener
                    // el invariante exacto de `slack` al hacer `backjump`.
                    continue;
                }
                if conflict.is_some() {
                    continue;
                }
                let max_c = self.constraints[cid].terms.first().map_or(0, |&(c, _)| c);
                if (max_c as i128) > slack {
                    let tlen = self.constraints[cid].terms.len();
                    for ti in 0..tlen {
                        let (cw, clit) = self.constraints[cid].terms[ti];
                        if (cw as i128) <= slack {
                            break;
                        }
                        if self.vals[clit as usize] == 0 {
                            self.assign(clit, cid);
                        }
                    }
                }
            }
            if conflict.is_some() {
                return conflict;
            }
        }
        None
    }

    /// Analiza un conflicto mediante planos de corte (RoundingSat).
    /// Devuelve `None` si derivó contradicción global (`0 >= 1` UNSAT), o `Some(backjump_level)`.
    fn analyze_and_learn<W: Write>(
        &mut self,
        conflict_cid: usize,
        mut proof: Option<&mut OpbProofWriter<W>>,
    ) -> Option<usize> {
        self.work.load(&self.constraints[conflict_cid]);
        self.constraints[conflict_cid].used = true;
        for idx in 0..self.work.active.len() {
            let v = self.work.active[idx];
            self.bump_var(v);
        }
        let cur_lvl = self.level();
        // Caso especial: ya es contradictorio sin usar asignaciones (`sum_coeffs < degree`).
        let sum_all: u128 = self.work.active.iter().map(|&v| self.work.coeffs[v].unsigned_abs()).sum();
        if sum_all < self.work.degree {
            self.work.weaken_all_to_contradiction();
            if let Some(log) = proof.as_deref_mut() {
                log.emit_pol(&self.work.ops);
            }
            self.root_unsat = true;
            return None;
        }
        let mut p = self.trail.len();
        loop {
            // Comprobar si `work` es asertiva para `cur_lvl > 0` antes de resolver `trail[p - 1]`.
            if cur_lvl > 0 {
                if let Some(bt_level) = self.check_asserting(cur_lvl, p) {
                    // Simplificar por GCD y aprender restricción PB.
                    self.work.simplify_gcd();
                    let terms = self.work.export_terms();
                    let deg = self.work.degree;
                    let total_w: u128 = terms.iter().map(|&(w, _)| w).sum();
                    if total_w < deg {
                        self.work.weaken_all_to_contradiction();
                        if let Some(log) = proof.as_deref_mut() {
                            log.emit_pol(&self.work.ops);
                        }
                        self.root_unsat = true;
                        return None;
                    }
                    let lbd = self.compute_lbd(&terms);
                    let pid = if let Some(log) = proof.as_deref_mut() {
                        log.emit_pol(&self.work.ops)
                    } else {
                        0
                    };
                    self.decay_scores();
                    self.backjump(bt_level);
                    // Al añadir tras `backjump(bt_level)`, dispara propagación o conflicto inmediato.
                    if let Some(next_confl) = self.add_constraint(terms, deg, pid, true, lbd) {
                        // Conflicto inmediato en `bt_level`: continuar análisis en el nivel inferior.
                        if bt_level == 0 {
                            return self.analyze_and_learn(next_confl, proof);
                        }
                        return self.analyze_and_learn(next_confl, proof);
                    }
                    return Some(bt_level);
                }
            }
            if p == 0 {
                // Todos los literales del trail han sido resueltos: contradicción global en nivel 0.
                self.work.weaken_all_to_contradiction();
                if let Some(log) = proof.as_deref_mut() {
                    log.emit_pol(&self.work.ops);
                }
                self.root_unsat = true;
                return None;
            }
            p -= 1;
            let q = self.trail[p];
            let v = (q >> 1) as usize;
            // Solo necesitamos resolver si `~q` aparece en `work` (está falsificado por `q`).
            let Some((a, wlit)) = self.work.lit_of(v) else {
                continue;
            };
            if wlit != (q ^ 1) {
                continue;
            }
            let reason_cid = self.reasons[v];
            if reason_cid == NO_REASON {
                // `q` es una decisión; puede ocurrir si `cur_lvl` bajó al resolver más allá de un nivel.
                // Si aún estamos falsificados bajo `trail[0..p]`, continuamos retrocediendo; si no,
                // ya debería haber disparado `check_asserting` arriba.
                if self.work.slack_at(&self.vals, &self.assign_pos, p) < 0 {
                    continue;
                }
                // Forzar aserción en `levels[v] - 1`.
                let bt = self.levels[v].saturating_sub(1);
                self.work.simplify_gcd();
                let terms = self.work.export_terms();
                let deg = self.work.degree;
                let lbd = self.compute_lbd(&terms);
                let pid = if let Some(log) = proof.as_deref_mut() {
                    log.emit_pol(&self.work.ops)
                } else {
                    0
                };
                self.decay_scores();
                self.backjump(bt);
                if let Some(next_confl) = self.add_constraint(terms, deg, pid, true, lbd) {
                    return self.analyze_and_learn(next_confl, proof);
                }
                return Some(bt);
            }
            self.constraints[reason_cid].used = true;
            // Buscar el coeficiente `b` de `q` en su razón `reason_cid`.
            let b = self.constraints[reason_cid].terms.iter()
                .find(|&&(_, lit)| lit == q)
                .map(|&(w, _)| w)
                .expect("La razon siempre contiene el literal propagado");
            let reason_clone = self.constraints[reason_cid].clone();
            for &(_, lit) in &reason_clone.terms {
                self.bump_var((lit >> 1) as usize);
            }
            self.work.resolve_with_reason(&reason_clone, q, a, b, &self.vals, &self.assign_pos, p);
            // Comprobar contradicción global temprana (`sum_coeffs < degree`).
            let sum_now: u128 = self.work.active.iter().map(|&u| self.work.coeffs[u].unsigned_abs()).sum();
            if sum_now < self.work.degree {
                self.work.weaken_all_to_contradiction();
                if let Some(log) = proof.as_deref_mut() {
                    log.emit_pol(&self.work.ops);
                }
                self.root_unsat = true;
                return None;
            }
        }
    }

    /// Comprueba si `work` es asertiva al desasignar todo lo que está por encima de algún `bt_level < cur_lvl`.
    /// Devuelve `Some(bt_level)` con el menor nivel donde `work` ya propaga (o entra en conflicto).
    fn check_asserting(&self, cur_lvl: usize, limit: usize) -> Option<usize> {
        // Primero comprobar si al quitar TODO el nivel `cur_lvl`, `work` propaga o entra en conflicto.
        let mut lhs_below_cur: i128 = 0;
        let mut max_coeff_at_or_above: u128 = 0;
        let mut falsified_levels: Vec<(usize, u128)> = Vec::new();
        for &v in &self.work.active {
            let Some((w, lit)) = self.work.lit_of(v) else { continue };
            let assigned_here = self.vals[lit as usize] != 0 && self.assign_pos[v] < limit;
            if assigned_here && self.levels[v] < cur_lvl {
                if self.vals[lit as usize] == -1 {
                    falsified_levels.push((self.levels[v], w));
                } else {
                    lhs_below_cur += w as i128;
                }
            } else {
                // Libre o en `cur_lvl`: al retroceder debajo de `cur_lvl` queda libre.
                lhs_below_cur += w as i128;
                max_coeff_at_or_above = max_coeff_at_or_above.max(w);
            }
        }
        let slack_below_cur = lhs_below_cur - self.work.degree as i128;
        if slack_below_cur >= max_coeff_at_or_above as i128 {
            return None;
        }
        // Es asertiva (o incluso conflictiva debajo de `cur_lvl`)!
        // Encontrar el nivel más bajo `bt < cur_lvl` donde sigue propagando o entrando en conflicto.
        falsified_levels.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        let mut bt = falsified_levels.first().map_or(0, |&(lvl, _)| lvl);
        // Intentar bajar `bt` aún más mientras `slack < max_unassigned_coeff`.
        let mut slack = slack_below_cur;
        let mut max_free = max_coeff_at_or_above;
        let mut idx = 0;
        while idx < falsified_levels.len() {
            let lvl = falsified_levels[idx].0;
            if lvl == 0 {
                break;
            }
            let mut next_slack = slack;
            let mut next_max = max_free;
            let mut j = idx;
            while j < falsified_levels.len() && falsified_levels[j].0 == lvl {
                let w = falsified_levels[j].1;
                next_slack += w as i128;
                next_max = next_max.max(w);
                j += 1;
            }
            if next_slack < next_max as i128 {
                slack = next_slack;
                max_free = next_max;
                idx = j;
                bt = if idx < falsified_levels.len() { falsified_levels[idx].0 } else { 0 };
            } else {
                break;
            }
        }
        Some(bt)
    }

    fn compute_lbd(&self, terms: &[(u128, u32)]) -> u32 {
        let mut lvls: Vec<usize> = terms.iter()
            .filter_map(|&(_, lit)| {
                let v = (lit >> 1) as usize;
                (self.vals[lit as usize] != 0 && self.levels[v] > 0).then_some(self.levels[v])
            })
            .collect();
        lvls.sort_unstable();
        lvls.dedup();
        lvls.len().max(1) as u32
    }

    fn reduce_database<W: Write>(&mut self, mut proof: Option<&mut OpbProofWriter<W>>) {
        self.cleanup_step = (self.cleanup_step + 500).min(16_000);
        self.next_cleanup = self.stats.conflicts + self.cleanup_step;
        let mut locked = vec![false; self.constraints.len()];
        for &lit in &self.trail {
            let r = self.reasons[(lit >> 1) as usize];
            if r != NO_REASON {
                locked[r] = true;
            }
        }
        let mut cands: Vec<usize> = Vec::new();
        for (cid, c) in self.constraints.iter_mut().enumerate() {
            if !c.learned || c.deleted || locked[cid] || c.lbd <= 2 {
                c.used = false;
                continue;
            }
            if c.used {
                c.used = false;
                continue;
            }
            cands.push(cid);
        }
        if cands.len() < 128 {
            return;
        }
        cands.sort_unstable_by(|&a, &b| {
            self.constraints[b].lbd.cmp(&self.constraints[a].lbd)
                .then_with(|| self.constraints[b].terms.len().cmp(&self.constraints[a].terms.len()))
        });
        let remove = cands.len() / 2;
        let mut del_ids: Vec<u64> = Vec::with_capacity(remove);
        for &cid in &cands[..remove] {
            self.constraints[cid].deleted = true;
            if self.constraints[cid].proof_id != 0 {
                del_ids.push(self.constraints[cid].proof_id);
            }
            self.learned_count = self.learned_count.saturating_sub(1);
            self.stats.deleted += 1;
        }
        if let Some(log) = proof.as_deref_mut() {
            if !del_ids.is_empty() {
                log.emit_delete_ids(&del_ids);
            }
        }
        // Purgar ocurrencias borradas de forma perezosa cuando crece el vector.
        for list in &mut self.occ {
            list.retain(|&(cid, _)| !self.constraints[cid].deleted);
        }
    }

    fn decide(&mut self) -> Option<u32> {
        let mut best_var: Option<usize> = None;
        let mut best_score = -1.0_f64;
        for v in 0..self.vars {
            if self.vals[v << 1] == 0 && self.scores[v] > best_score {
                best_score = self.scores[v];
                best_var = Some(v);
            }
        }
        let v = best_var?;
        self.stats.decisions += 1;
        Some(((v as u32) << 1) | u32::from(self.phases[v]))
    }

    fn search<W: Write>(
        &mut self,
        mut proof: Option<&mut OpbProofWriter<W>>,
        stop: Option<&AtomicBool>,
    ) -> Option<bool> {
        if self.root_unsat {
            return Some(false);
        }
        loop {
            if stop.is_some_and(|s| s.load(Ordering::Acquire)) {
                return None;
            }
            if let Some(confl) = self.propagate() {
                self.stats.conflicts += 1;
                if self.analyze_and_learn(confl, proof.as_deref_mut()).is_none() {
                    return Some(false);
                }
                continue;
            }
            if self.stats.conflicts >= self.next_cleanup && self.learned_count >= 512 {
                self.reduce_database(proof.as_deref_mut());
            }
            if self.stats.conflicts >= self.next_restart && self.level() > 0 {
                self.backjump(0);
                self.stats.restarts += 1;
                let (u, v) = self.luby;
                self.luby = if u & u.wrapping_neg() == v { (u + 1, 1) } else { (u, 2 * v) };
                self.next_restart = self.stats.conflicts + 100 * v;
            }
            let Some(lit) = self.decide() else {
                return Some(true);
            };
            self.trail_limits.push(self.trail.len());
            self.assign(lit, NO_REASON);
        }
    }

    fn extract_model(&self) -> Vec<bool> {
        let mut model = vec![false; self.vars + 1];
        for v in 0..self.vars {
            model[v + 1] = self.vals[v << 1] == 1;
        }
        model
    }
}

/// Resuelve un problema OPB directamente mediante Planos de Corte (sin bit-blasting a CNF),
/// emitiendo opcionalmente una prueba VeriPB 3.0.2 nativa (`veripb --opb`).
pub fn solve_opb<W: Write>(
    problem: &Problem,
    mut proof: Option<&mut OpbProofWriter<W>>,
    stop: Option<&AtomicBool>,
    mut on_improved: Option<&mut dyn FnMut(i128)>,
) -> (CpOutcome, CpStats) {
    let vars = problem.variables as usize;
    let mut solver = CpSolver::new(vars, proof.is_some());
    // Cargar fases iniciales a partir del objetivo si existe (favorece soluciones de bajo coste).
    if let Some(obj) = &problem.objective {
        for &(coeff, lit) in obj {
            let v = lit.unsigned_abs() as usize - 1;
            if v < vars && coeff != 0 {
                // Para minimizar `coeff * lit`, preferir asignar `lit` a falso si `coeff > 0`, verdadero si `coeff < 0`.
                let want_true = (coeff < 0) == (lit > 0);
                solver.phases[v] = u8::from(!want_true);
            }
        }
    }
    // Cargar restricciones originales (ID VeriPB = i + 1).
    let mut initial_conflict: Option<usize> = None;
    for (i, c) in problem.constraints.iter().enumerate() {
        let orig_id = i as u64 + 1;
        let (terms, deg, needs_sat) = normalize_raw(&c.terms, i128::from(c.degree), vars);
        if deg == 0 {
            continue;
        }
        let pid = if let Some(log) = proof.as_deref_mut() {
            if needs_sat {
                log.emit_pol(&[PolOp::Id(orig_id), PolOp::Sat])
            } else {
                orig_id
            }
        } else {
            0
        };
        if let Some(confl) = solver.add_constraint(terms, deg, pid, false, 1) {
            initial_conflict = Some(confl);
            break;
        }
    }
    if let Some(confl) = initial_conflict {
        solver.analyze_and_learn(confl, proof.as_deref_mut());
        if let Some(log) = proof.as_deref() {
            solver.stats.proof_lines = log.lines();
        }
        return (CpOutcome::Unsat, solver.stats);
    }

    let Some(obj_terms) = &problem.objective else {
        let outcome = match solver.search(proof.as_deref_mut(), stop) {
            Some(true) => CpOutcome::Sat(solver.extract_model()),
            Some(false) => CpOutcome::Unsat,
            None => CpOutcome::Stopped { best: None },
        };
        if let Some(log) = proof.as_deref() {
            solver.stats.proof_lines = log.lines();
        }
        return (outcome, solver.stats);
    };

    // Optimización directa con mejora iterativa + `soli` en VeriPB 3.0.2.
    let mut best: Option<(Vec<bool>, i128)> = None;
    let neg_obj: Option<Vec<(i64, i32)>> = obj_terms.iter()
        .map(|&(a, l)| a.checked_neg().map(|na| (na, l)))
        .collect();
    let Some(neg_obj) = neg_obj else {
        return (CpOutcome::Stopped { best: None }, solver.stats);
    };

    loop {
        match solver.search(proof.as_deref_mut(), stop) {
            Some(true) => {
                let model = solver.extract_model();
                debug_assert!(opb::satisfies(problem, &model));
                let cost = opb::evaluate(obj_terms, &model);
                if let Some(cb) = on_improved.as_deref_mut() {
                    cb(cost);
                }
                best = Some((model.clone(), cost));
                // Emitir `soli` (VeriPB agrega automáticamente `obj <= cost - 1`).
                let soli_id = if let Some(log) = proof.as_deref_mut() {
                    log.emit_soli(&model, vars)
                } else {
                    0
                };
                // Añadir `-obj >= -(cost - 1)` al nivel 0 conservando todas las cláusulas aprendidas.
                solver.backjump(0);
                let target_deg = -(cost - 1);
                let (terms, deg, needs_sat) = normalize_raw(&neg_obj, target_deg, vars);
                let pid = if let Some(log) = proof.as_deref_mut() {
                    if needs_sat && deg > 0 {
                        log.emit_pol(&[PolOp::Id(soli_id), PolOp::Sat])
                    } else {
                        soli_id
                    }
                } else {
                    0
                };
                if deg == 0 {
                    // Imposible tras `soli` (el coste siempre mejora), pero guardamos invariante.
                    continue;
                }
                if let Some(confl) = solver.add_constraint(terms, deg, pid, false, 1) {
                    if solver.analyze_and_learn(confl, proof.as_deref_mut()).is_none() {
                        if let Some(log) = proof.as_deref() {
                            solver.stats.proof_lines = log.lines();
                        }
                        let (m, c) = best.unwrap();
                        return (CpOutcome::Optimum { model: m, cost: c }, solver.stats);
                    }
                }
            }
            Some(false) => {
                if let Some(log) = proof.as_deref() {
                    solver.stats.proof_lines = log.lines();
                }
                return match best {
                    Some((model, cost)) => (CpOutcome::Optimum { model, cost }, solver.stats),
                    None => (CpOutcome::Unsat, solver.stats),
                };
            }
            None => {
                if let Some(log) = proof.as_deref() {
                    solver.stats.proof_lines = log.lines();
                }
                return (CpOutcome::Stopped { best }, solver.stats);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Palomar PHP(n+1, n) codificado en OPB: exponencial para resolución pura, polinomial
    /// (milisegundos) para Planos de Corte incluso sin ruptura de simetrías.
    #[test]
    fn cutting_planes_refutes_php20_without_symmetry_in_milliseconds() {
        const HOLES: usize = 20;
        const PIGEONS: usize = HOLES + 1;
        let var = |p: usize, h: usize| (p * HOLES + h + 1) as i32;
        let mut constraints = Vec::new();
        for p in 0..PIGEONS {
            let terms = (0..HOLES).map(|h| (1_i64, var(p, h))).collect();
            constraints.push(opb::Constraint { terms, degree: 1 });
        }
        for h in 0..HOLES {
            // sum_p x_{p,h} <= 1  <=>  sum_p -x_{p,h} >= -1
            let terms = (0..PIGEONS).map(|p| (-1_i64, var(p, h))).collect();
            constraints.push(opb::Constraint { terms, degree: -1 });
        }
        let problem = Problem { variables: (PIGEONS * HOLES) as u32, objective: None, constraints };
        let mut proof_buf = Vec::new();
        let mut writer = OpbProofWriter::new(&mut proof_buf, problem.constraints.len());
        let (outcome, stats) = solve_opb(&problem, Some(&mut writer), None, None);
        assert!(matches!(outcome, CpOutcome::Unsat));
        assert!(stats.conflicts < 5000, "PHP20 deberia refutarse en pocos conflictos, tuvo {}", stats.conflicts);
        let _ = writer.finish(false, None, true).unwrap();
        assert!(!proof_buf.is_empty());
    }

    /// Mochila con pesos enormes (~10^12): intratable por bit-blasting clásico, instantánea aquí.
    #[test]
    fn cutting_planes_optimizes_large_weight_knapsack_exactly() {
        let src = "\
min: -1000000000000 x1 -1500000000000 x2 -900000000000 x3 -2000000000000 x4 ;
-400000000000 x1 -600000000000 x2 -350000000000 x3 -850000000000 x4 >= -1000000000000 ;
";
        let problem = opb::parse(src).unwrap();
        let (outcome, _) = solve_opb::<Vec<u8>>(&problem, None, None, None);
        match outcome {
            CpOutcome::Optimum { model, cost } => {
                assert!(opb::satisfies(&problem, &model));
                // x1 + x2: peso 1.0e12, valor -2.5e12 (optimo: x2+x3=9.5e11 -> -2.4e12; x4 solo -> -2.0e12).
                assert_eq!(cost, -2_500_000_000_000);
            }
            other => panic!("Esperado Optimum, recibido {other:?}"),
        }
    }
}
