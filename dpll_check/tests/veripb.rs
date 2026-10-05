#![forbid(unsafe_code)]
#![deny(warnings)]

#[allow(dead_code)]
#[path = "../src/hybrid.rs"]
mod hybrid;

use hybrid::proof::PbProof;
use hybrid::{solve_hybrid, solve_hybrid_with_proof, verify_model, HybridConfig};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

// An independent checker for the pseudo-Boolean trace, sharing no code with the engine.
// It checks the mathematics of every step: reverse unit propagation over clauses and
// linear constraints, redundancy of definitions, presence of deleted clauses, and for each
// dominance step that the witness is a symmetry of the input formula and that the
// constraint is exactly z <= sigma(z) for the loaded order. It reads the VeriPB 3.0 lines
// the engine writes, in the layout it writes them; it is not a VeriPB parser. The official
// checker is run by the ignored test `the_official_checker_verifies_the_traces`.

// Non-negative integers in base 10^9, least significant limb first.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Big(Vec<u32>);

const BASE: u64 = 1_000_000_000;

impl Big {
    fn zero() -> Self {
        Big(Vec::new())
    }
    fn parse(text: &str) -> Self {
        assert!(!text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit()), "Not a number: {text}");
        let mut limbs: Vec<u32> = text.as_bytes().rchunks(9)
            .map(|chunk| std::str::from_utf8(chunk).unwrap().parse().unwrap()).collect();
        while limbs.last() == Some(&0) {
            limbs.pop();
        }
        Big(limbs)
    }
    fn add(&self, other: &Big) -> Big {
        let mut limbs = Vec::with_capacity(self.0.len().max(other.0.len()) + 1);
        let mut carry = 0;
        for index in 0..self.0.len().max(other.0.len()) {
            let sum = u64::from(*self.0.get(index).unwrap_or(&0)) + u64::from(*other.0.get(index).unwrap_or(&0)) + carry;
            limbs.push((sum % BASE) as u32);
            carry = sum / BASE;
        }
        if carry > 0 {
            limbs.push(carry as u32);
        }
        Big(limbs)
    }
}
impl Ord for Big {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.len().cmp(&other.0.len()).then_with(|| self.0.iter().rev().cmp(other.0.iter().rev()))
    }
}
impl PartialOrd for Big {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Sum of weight * literal >= degree, every weight positive and one term per variable.
struct Linear {
    terms: Vec<(i32, Big)>,
    degree: Big,
}

struct Checker {
    original: BTreeSet<Vec<i32>>,
    inputs: usize,
    clauses: Vec<Vec<i32>>,
    linear: Vec<Linear>,
    // Variables of the loaded order, most significant first, and the weight of each.
    order: Vec<i32>,
    weights: Vec<Big>,
    refuted: bool,
}

fn distinct(clause: &[i32]) -> Vec<i32> {
    let mut literals = clause.to_vec();
    literals.sort_unstable();
    literals.dedup();
    literals
}

fn tautology(clause: &[i32]) -> bool {
    clause.iter().any(|literal| clause.contains(&-literal))
}

fn name(token: &str, prefix: &str) -> i32 {
    token.strip_prefix(prefix).and_then(|digits| digits.parse().ok())
        .unwrap_or_else(|| panic!("Expected {prefix}<n>, found {token}"))
}

// "-4 x3 +4 x7 1 ~x2": (negative coefficient, signed literal, magnitude).
fn terms(text: &str) -> Vec<(bool, i32, Big)> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    assert_eq!(tokens.len() % 2, 0, "Terms come in pairs: {text}");
    tokens.chunks(2).map(|pair| {
        let negative = pair[0].starts_with('-');
        let magnitude = Big::parse(pair[0].trim_start_matches(['+', '-']));
        let literal = match pair[1].strip_prefix('~') {
            Some(plain) => -name(plain, "x"),
            None => name(pair[1], "x"),
        };
        (negative, literal, magnitude)
    }).collect()
}

fn clause_of(text: &str) -> Vec<i32> {
    let body = text.strip_suffix(">= 1").unwrap_or_else(|| panic!("Not a clause: {text}"));
    let unit = Big::parse("1");
    terms(body).into_iter().map(|(negative, literal, magnitude)| {
        assert!(!negative && magnitude == unit, "Not a clause: {text}");
        literal
    }).collect()
}

impl Checker {
    fn new(formula: &[Vec<i32>]) -> Self {
        Self {
            original: formula.iter().filter(|clause| !tautology(clause)).map(|clause| distinct(clause)).collect(),
            inputs: formula.len(),
            clauses: formula.iter().map(|clause| distinct(clause)).collect(),
            linear: Vec::new(),
            order: Vec::new(),
            weights: Vec::new(),
            refuted: false,
        }
    }
    // Unit propagation over clauses and linear constraints; true if it reaches a conflict.
    fn refutes(&self, assumptions: &[i32]) -> bool {
        let last = self.clauses.iter().flatten().chain(assumptions).copied()
            .chain(self.linear.iter().flat_map(|constraint| constraint.terms.iter().map(|term| term.0)))
            .map(i32::unsigned_abs).max().unwrap_or(0);
        // 0 unknown, 1 true, -1 false, indexed by variable.
        let mut values = vec![0_i8; last as usize + 1];
        let value = |values: &[i8], literal: i32| values[literal.unsigned_abs() as usize] * literal.signum() as i8;
        for &literal in assumptions {
            if value(&values, literal) == -1 {
                return true;
            }
            values[literal.unsigned_abs() as usize] = literal.signum() as i8;
        }
        loop {
            let mut changed = false;
            for clause in &self.clauses {
                let mut open = None;
                let mut pending = 0;
                let mut satisfied = false;
                for &literal in clause {
                    match value(&values, literal) {
                        1 => satisfied = true,
                        0 => {
                            pending += 1;
                            open = Some(literal);
                        }
                        _ => {}
                    }
                }
                if satisfied {
                    continue;
                }
                match (pending, open) {
                    (0, _) => return true,
                    (1, Some(literal)) => {
                        values[literal.unsigned_abs() as usize] = literal.signum() as i8;
                        changed = true;
                    }
                    _ => {}
                }
            }
            for constraint in &self.linear {
                // What the literals not yet false can still contribute.
                let reachable = constraint.terms.iter()
                    .filter(|term| value(&values, term.0) != -1)
                    .fold(Big::zero(), |sum, term| sum.add(&term.1));
                if reachable < constraint.degree {
                    return true;
                }
                for (literal, weight) in &constraint.terms {
                    // Without this literal the degree is out of reach: it must be true.
                    if value(&values, *literal) == 0 && reachable < constraint.degree.add(weight) {
                        values[literal.unsigned_abs() as usize] = literal.signum() as i8;
                        changed = true;
                    }
                }
            }
            if !changed {
                return false;
            }
        }
    }
    fn rup(&self, clause: &[i32]) -> bool {
        let negated: Vec<i32> = clause.iter().map(|literal| -literal).collect();
        self.refutes(&negated)
    }
    // def_order ... end def_order; load_order: the order must be the lexicographic one,
    // sum of 2^(n-1-i) (v_i - u_i) >= 0. For any such linear definition the transitivity
    // proof is the sum of u <= v, v <= w and the negation of u <= w, which is 0 >= 1.
    fn define_order(&mut self, lines: &[&str]) -> Result<usize, String> {
        let frame = [
            (0, "vars"), (3, "aux;"), (4, "end vars;"), (5, "def"), (7, "end def;"), (8, "transitivity"),
            (9, "vars"), (11, "end vars;"), (12, "proof"), (13, "proofgoal #1"), (14, "pol 1 2 + 3 +;"),
            (15, "qed #1 : -1;"), (16, "qed proof;"), (17, "end transitivity;"), (18, "end def_order;"),
        ];
        if lines.len() < 20 || frame.iter().any(|&(index, text)| lines[index] != text) {
            return Err("malformed order definition".into());
        }
        let list = |line: &str, keyword: &str, prefix: &str| -> Result<usize, String> {
            let body = line.strip_prefix(keyword).and_then(|rest| rest.strip_suffix(';'))
                .ok_or(format!("expected {keyword}"))?;
            let names: Vec<&str> = body.split_whitespace().collect();
            let fine = names.iter().enumerate().all(|(index, token)| *token == format!("{prefix}{}", index + 1));
            if fine { Ok(names.len()) } else { Err(format!("bad {keyword} list")) }
        };
        let count = list(lines[1], "left", "u")?;
        if count == 0 || list(lines[2], "right", "v")? != count || list(lines[10], "fresh_right", "w")? != count {
            return Err("order variable lists differ in length".into());
        }
        let mut weights = vec![Big::parse("1")];
        for _ in 1..count {
            let last = &weights[weights.len() - 1];
            weights.push(last.add(last));
        }
        weights.reverse();
        let tokens: Vec<&str> = lines[6].strip_suffix(">= 0;").ok_or("order must end in >= 0;")?.split_whitespace().collect();
        if tokens.len() != 4 * count {
            return Err("order definition has the wrong number of terms".into());
        }
        for (index, group) in tokens.chunks(4).enumerate() {
            let fine = group[0].strip_prefix('-').is_some_and(|digits| Big::parse(digits) == weights[index])
                && group[1] == format!("u{}", index + 1)
                && group[2].strip_prefix('+').is_some_and(|digits| Big::parse(digits) == weights[index])
                && group[3] == format!("v{}", index + 1);
            if !fine {
                return Err(format!("order term {index} is not -2^k u +2^k v"));
            }
        }
        let loaded = lines[19].strip_prefix("load_order lex ").and_then(|rest| rest.strip_suffix(';'))
            .ok_or("the order is not loaded")?;
        self.order = loaded.split_whitespace().map(|token| name(token, "x")).collect();
        if self.order.len() != count || self.order.iter().collect::<BTreeSet<_>>().len() != count {
            return Err("the order is loaded over the wrong variables".into());
        }
        // Loading moves nothing to the core: nothing has been derived yet.
        if self.clauses.len() != self.inputs || !self.linear.is_empty() {
            return Err("the order is loaded after derivations".into());
        }
        self.weights = weights;
        Ok(20)
    }
    // dom C : sigma : subproof. Checked here: sigma permutes its support and maps every
    // input clause to an input clause, and C is sum of w_i (sigma(z_i) - z_i) >= 0 over the
    // ordered variables, with like terms collected. The rule's goals then hold: sigma(z) <= z
    // follows from the negation of C, and the strict goal #2 assumes z <= sigma(z), which is
    // C itself, so adding it to the negation of C gives 0 >= 1 as the subproof says.
    fn dominance(&mut self, constraint: &str, witness: &str) -> Result<(), String> {
        let tokens: Vec<&str> = witness.split_whitespace().collect();
        if tokens.len() % 3 != 0 || tokens.chunks(3).any(|triple| triple[1] != "->") {
            return Err(format!("malformed witness {witness}"));
        }
        let map: BTreeMap<i32, i32> = tokens.chunks(3).map(|triple| (name(triple[0], "x"), name(triple[2], "x"))).collect();
        let images: BTreeSet<i32> = map.values().copied().collect();
        if map.len() != tokens.len() / 3 || images != map.keys().copied().collect() {
            return Err(format!("witness {witness} is not a permutation"));
        }
        let image = |literal: i32| map.get(&literal.abs()).map_or(literal, |&target| target * literal.signum());
        for clause in &self.original {
            let mapped = distinct(&clause.iter().map(|&literal| image(literal)).collect::<Vec<i32>>());
            if !self.original.contains(&mapped) {
                return Err(format!("witness {witness} sends {clause:?} outside the formula"));
            }
        }
        // What each variable gains as an image and loses at its own position.
        let mut expected: BTreeMap<i32, (Big, Big)> = BTreeMap::new();
        for (variable, weight) in self.order.iter().zip(&self.weights) {
            if let Some(&target) = map.get(variable) {
                expected.entry(target).or_insert((Big::zero(), Big::zero())).0 = weight.clone();
                expected.entry(*variable).or_insert((Big::zero(), Big::zero())).1 = weight.clone();
            }
        }
        let body = constraint.strip_suffix(">= 0").ok_or("dominance constraint must end in >= 0")?;
        let stated: Vec<(i32, bool, Big)> = terms(body).into_iter().map(|(negative, literal, magnitude)| {
            assert!(literal > 0, "The order inequality is written over plain variables");
            (literal, negative, magnitude)
        }).collect();
        // One term per variable, equal to gain minus loss: checked without subtracting.
        let mut seen = BTreeSet::new();
        let exact = !stated.is_empty() && stated.iter().all(|(variable, negative, magnitude)| {
            seen.insert(*variable) && magnitude > &Big::zero() && expected.get(variable).is_some_and(|(gain, loss)| {
                if *negative { gain.add(magnitude) == *loss } else { loss.add(magnitude) == *gain }
            })
        });
        if !exact || expected.iter().any(|(variable, (gain, loss))| gain != loss && !seen.contains(variable)) {
            return Err(format!("{constraint} is not the lexicographic inequality of {witness}"));
        }
        // -w x is w ~x with the degree raised by w.
        let mut degree = Big::zero();
        let terms = stated.into_iter().map(|(variable, negative, weight)| {
            if negative {
                degree = degree.add(&weight);
                (-variable, weight)
            } else {
                (variable, weight)
            }
        }).collect();
        self.linear.push(Linear { terms, degree });
        Ok(())
    }
    // Checks a whole trace. `Ok(true)` means it concludes UNSAT with 0 >= 1 derived last.
    fn check(&mut self, trace: &str) -> Result<bool, String> {
        let lines: Vec<&str> = trace.lines().collect();
        if lines.first() != Some(&"pseudo-Boolean proof version 3.0") || lines.get(1) != Some(&format!("f {};", self.inputs).as_str()) {
            return Err("bad header".into());
        }
        let Some(end) = lines.len().checked_sub(3).filter(|&end| end >= 2) else {
            return Err("missing footer".into());
        };
        if lines[end] != "output NONE;" || lines[end + 2] != "end pseudo-Boolean proof;" {
            return Err("bad footer".into());
        }
        let mut index = 2;
        while index < end {
            let line = lines[index];
            index += 1;
            if line == "def_order lex" {
                index += self.define_order(&lines[index..end]).map_err(|error| format!("line {index}: {error}"))?;
                continue;
            }
            let Some(statement) = line.strip_suffix(';').or_else(|| line.strip_suffix(" : subproof")) else {
                return Err(format!("line {index}: unterminated rule {line}"));
            };
            if let Some(rest) = statement.strip_prefix("rup ") {
                let clause = clause_of(rest);
                if !self.rup(&clause) {
                    return Err(format!("line {index}: {clause:?} is not RUP"));
                }
                self.refuted = clause.is_empty();
                self.clauses.push(distinct(&clause));
            } else if let Some(rest) = statement.strip_prefix("red ") {
                let (constraint, witness) = rest.split_once(" : ").ok_or("red without witness")?;
                let clause = clause_of(constraint);
                let tokens: Vec<&str> = witness.split_whitespace().collect();
                let Some(&pivot) = clause.first() else { return Err("empty definition".into()) };
                // The witness sets the first literal of the clause to true.
                if tokens.len() != 3 || name(tokens[0], "x") != pivot.abs() || tokens[1] != "->" || tokens[2] != if pivot > 0 { "1" } else { "0" } {
                    return Err(format!("line {index}: witness {witness} does not satisfy {clause:?}"));
                }
                // It must leave the order and the dominance constraints untouched.
                let ordered = self.order.contains(&pivot.abs())
                    || self.linear.iter().any(|constraint| constraint.terms.iter().any(|term| term.0.abs() == pivot.abs()));
                if ordered {
                    return Err(format!("line {index}: a defined variable occurs in the order"));
                }
                // Every clause the witness falsifies must follow anyway.
                let fine = self.clauses.iter().filter(|other| other.contains(&-pivot)).all(|other| {
                    let mut resolvent = clause.clone();
                    resolvent.extend(other.iter().copied().filter(|&literal| literal != -pivot));
                    self.rup(&resolvent)
                });
                if !fine {
                    return Err(format!("line {index}: {clause:?} is not redundant"));
                }
                self.refuted = false;
                self.clauses.push(distinct(&clause));
            } else if let Some(rest) = statement.strip_prefix("del spec ") {
                // Only derived clauses may go: the input formula is the core.
                let wanted = distinct(&clause_of(rest));
                match self.clauses[self.inputs..].iter().position(|known| *known == wanted) {
                    Some(position) => {
                        self.clauses.swap_remove(self.inputs + position);
                    }
                    None => return Err(format!("line {index}: deleted clause {wanted:?} is not a derived clause")),
                }
            } else if let Some(rest) = statement.strip_prefix("dom ") {
                let (constraint, witness) = rest.split_once(" : ").ok_or("dom without witness")?;
                if !line.ends_with(" : subproof") || index + 4 > end || lines[index..index + 4] != ["proofgoal #2", "pol -1 -2 +;", "qed #2 : -1;", "qed dom;"] {
                    return Err(format!("line {index}: malformed dominance step"));
                }
                index += 4;
                self.dominance(constraint, witness).map_err(|error| format!("line {index}: {error}"))?;
                self.refuted = false;
            } else {
                return Err(format!("line {index}: unknown rule {line}"));
            }
        }
        match lines[end + 1] {
            "conclusion UNSAT : -1;" if self.refuted => Ok(true),
            "conclusion NONE;" => Ok(false),
            other => Err(format!("{other} does not match the trace")),
        }
    }
}

struct Run {
    satisfiable: bool,
    trace: String,
    dominances: u64,
    deletions: u64,
    generators: usize,
    predicates: usize,
    decisions: usize,
}

fn certified(formula: &[Vec<i32>], config: HybridConfig) -> Run {
    let mut proof = PbProof::new(Vec::new(), formula.len());
    let report = solve_hybrid_with_proof(formula, config, &mut proof);
    if let Some(model) = &report.model {
        assert!(verify_model(formula, model));
    }
    let (dominances, deletions, steps, written) = (proof.dominances(), proof.deletions(), proof.steps(), proof.bytes());
    let trace = String::from_utf8(proof.finish().unwrap()).unwrap();
    // After the last step only the three closing lines are written.
    assert!(trace.len() as u64 > written);
    let derived = trace.lines().filter(|line| ["rup ", "red ", "dom "].iter().any(|rule| line.starts_with(rule))).count();
    assert_eq!(derived as u64, steps);
    Run {
        satisfiable: report.model.is_some(),
        trace,
        dominances,
        deletions,
        generators: report.symmetry.generators,
        predicates: report.symmetry.clauses,
        decisions: report.cdcl.decisions,
    }
}

fn assert_valid(formula: &[Vec<i32>]) -> Run {
    let run = certified(formula, HybridConfig::default());
    assert_eq!(Checker::new(formula).check(&run.trace), Ok(!run.satisfiable), "Invalid trace for {formula:?}");
    // The certified run and the plain pipeline must agree on the answer.
    assert_eq!(run.satisfiable, solve_hybrid(formula, HybridConfig::default()).model.is_some());
    run
}

fn pigeonhole(pigeons: usize, holes: usize) -> Vec<Vec<i32>> {
    let mut formula: Vec<Vec<i32>> = (0..pigeons)
        .map(|pigeon| (0..holes).map(|hole| (pigeon * holes + hole + 1) as i32).collect())
        .collect();
    for hole in 0..holes {
        for first in 0..pigeons {
            for second in first + 1..pigeons {
                formula.push(vec![-((first * holes + hole + 1) as i32), -((second * holes + hole + 1) as i32)]);
            }
        }
    }
    formula
}

// The benchmark's encoding: each pigeon row split into clauses of three literals.
fn pigeonhole_3sat(pigeons: usize, holes: usize) -> Vec<Vec<i32>> {
    let mut variables = pigeons * holes;
    let mut formula = Vec::new();
    for pigeon in 0..pigeons {
        let row: Vec<i32> = (0..holes).map(|hole| (pigeon * holes + hole + 1) as i32).collect();
        if row.len() <= 3 {
            formula.push(row);
            continue;
        }
        variables += 1;
        let mut auxiliary = variables as i32;
        formula.push(vec![row[0], row[1], auxiliary]);
        for &literal in &row[2..row.len() - 2] {
            variables += 1;
            formula.push(vec![-auxiliary, literal, variables as i32]);
            auxiliary = variables as i32;
        }
        formula.push(vec![-auxiliary, row[row.len() - 2], row[row.len() - 1]]);
    }
    formula.extend(pigeonhole(pigeons, holes).into_iter().skip(pigeons));
    formula
}

fn colouring(vertices: usize, edges: &[(usize, usize)], colours: usize) -> Vec<Vec<i32>> {
    let variable = |vertex: usize, colour: usize| (vertex * colours + colour + 1) as i32;
    let mut formula: Vec<Vec<i32>> = (0..vertices)
        .map(|vertex| (0..colours).map(|colour| variable(vertex, colour)).collect())
        .collect();
    for &(first, second) in edges {
        for colour in 0..colours {
            formula.push(vec![-variable(first, colour), -variable(second, colour)]);
        }
    }
    formula
}

fn xor_clauses(variables: &[i32], value: bool, formula: &mut Vec<Vec<i32>>) {
    for pattern in 0..1_u32 << variables.len() {
        if (pattern.count_ones() & 1 == 1) != value {
            formula.push(variables.iter().enumerate()
                .map(|(bit, &variable)| if pattern >> bit & 1 == 1 { -variable } else { variable }).collect());
        }
    }
}

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

#[test]
fn pigeonhole_is_refuted_with_certified_symmetry_breaking() {
    for (pigeons, holes) in [(3, 2), (4, 3), (5, 4), (6, 5), (9, 8), (10, 9), (13, 12)] {
        let run = assert_valid(&pigeonhole(pigeons, holes));
        assert!(!run.satisfiable);
        // Symmetry stays on: every generator is justified and no decision is needed.
        assert!(run.generators > 0 && run.predicates > 0);
        assert_eq!(run.dominances as usize, run.generators);
        assert_eq!(run.decisions, 0);
    }
}

#[test]
fn the_benchmark_instance_is_certified_with_symmetry_on() {
    // PHP 10/9 as the stress benchmark runs it: rows of three literals plus a tautology.
    let mut formula = pigeonhole_3sat(10, 9);
    formula.push(vec![1, -1]);
    let run = assert_valid(&formula);
    assert!(!run.satisfiable);
    assert!(run.dominances > 0 && run.predicates > 0);
    let plain = solve_hybrid(&formula, HybridConfig::default());
    // Certifying costs no search: same generators, same predicates, same decisions.
    assert_eq!((run.generators, run.predicates), (plain.symmetry.generators, plain.symmetry.clauses));
    assert_eq!(run.decisions, plain.cdcl.decisions);
}

#[test]
fn other_symmetric_families_are_certified_too() {
    let complete: Vec<(usize, usize)> = (0..6).flat_map(|a| (a + 1..6).map(move |b| (a, b))).collect();
    let cycle: Vec<(usize, usize)> = (0..7).map(|vertex| (vertex, (vertex + 1) % 7)).collect();
    for formula in [colouring(6, &complete, 5), colouring(7, &cycle, 2), colouring(7, &cycle, 3), pigeonhole(5, 5), pigeonhole(4, 7)] {
        let run = assert_valid(&formula);
        assert!(run.dominances > 0);
    }
}

#[test]
fn parity_and_symmetry_share_one_trace() {
    // Tseitin on the Petersen graph: symmetric and refuted by elimination alone.
    let edges: Vec<(usize, usize)> = (0..5)
        .flat_map(|i| [(i, (i + 1) % 5), (i, i + 5), (i + 5, (i + 2) % 5 + 5)])
        .collect();
    let mut formula = Vec::new();
    for vertex in 0..10 {
        let incident: Vec<i32> = edges.iter().enumerate()
            .filter(|(_, &(a, b))| a == vertex || b == vertex)
            .map(|(index, _)| index as i32 + 1).collect();
        xor_clauses(&incident, vertex == 0, &mut formula);
    }
    let run = assert_valid(&formula);
    assert!(!run.satisfiable);
    assert!(run.trace.contains("\nred "), "The elimination defines auxiliaries");

    // Interchangeable variables next to a linear system that forces a unit.
    let mut formula = vec![vec![1, 2, 3], vec![-1, -2], vec![-1, -3], vec![-2, -3]];
    xor_clauses(&[4, 5], true, &mut formula);
    xor_clauses(&[4, 5, 6], false, &mut formula);
    formula.push(vec![-6, 1]);
    formula.push(vec![-6, 2]);
    let run = assert_valid(&formula);
    assert!(!run.satisfiable && run.dominances > 0);
}

#[test]
fn random_symmetric_formulas_produce_valid_traces() {
    let mut seed = 0x510e_527f_ade6_82d1;
    let (mut refuted, mut broken) = (0, 0);
    for case in 0..260 {
        let variables = 4 + (next(&mut seed) % 8) as usize;
        let base: Vec<Vec<i32>> = (0..1 + next(&mut seed) % variables as u64).map(|_| {
            (0..1 + next(&mut seed) % 3).map(|_| {
                let variable = 1 + (next(&mut seed) % variables as u64) as i32;
                if next(&mut seed) % 3 == 0 { -variable } else { variable }
            }).collect()
        }).collect();
        // Closing the clause set under a permutation makes it symmetric by construction.
        let mut image: Vec<i32> = (0..=variables as i32).collect();
        for _ in 0..1 + next(&mut seed) % 3 {
            let (a, b) = (1 + (next(&mut seed) % variables as u64) as usize, 1 + (next(&mut seed) % variables as u64) as usize);
            image.swap(a, b);
        }
        let mut formula = base.clone();
        let mut frontier = base;
        for _ in 0..2 * variables {
            frontier = frontier.iter().map(|clause| {
                clause.iter().map(|&literal| image[literal.unsigned_abs() as usize] * literal.signum()).collect()
            }).collect();
            formula.extend(frontier.iter().cloned());
        }
        if case % 4 == 0 {
            xor_clauses(&[1, 2, 3], next(&mut seed) & 1 == 1, &mut formula);
            xor_clauses(&[2, 3, 4], next(&mut seed) & 1 == 1, &mut formula);
        }
        let run = assert_valid(&formula);
        refuted += usize::from(!run.satisfiable);
        broken += usize::from(run.dominances > 0);
    }
    assert!(refuted > 20 && broken > 100, "{refuted} refutations, {broken} with dominance steps");
}

// cargo test --release --test veripb -- --ignored --nocapture
#[test]
#[ignore = "a wide differential run, meant for release builds"]
fn wide_fuzz_of_certified_runs() {
    let mut seed = 0x9b05_688c_2b3e_6c1f;
    let mut below = |bound: usize| (next(&mut seed) % bound as u64) as usize;
    let (mut refuted, mut steps, mut dominances, mut deletions, mut definitions) = (0, 0, 0, 0, 0);
    let cases = 4000;
    for case in 0..cases {
        let mut formula: Vec<Vec<i32>> = Vec::new();
        match case % 4 {
            // Random 3-SAT around the threshold: search, learned clauses, deletions.
            0 => {
                let variables = 20 + below(50);
                for _ in 0..variables * (40 + below(12)) / 10 {
                    formula.push((0..3).map(|_| (1 + below(variables) as i32) * if below(2) == 0 { 1 } else { -1 }).collect());
                }
            }
            // A clause set closed under a random permutation.
            1 => {
                let variables = 4 + below(16);
                let mut image: Vec<i32> = (0..=variables as i32).collect();
                for _ in 0..1 + below(4) {
                    image.swap(1 + below(variables), 1 + below(variables));
                }
                let mut frontier: Vec<Vec<i32>> = (0..1 + below(variables)).map(|_| {
                    (0..1 + below(3)).map(|_| (1 + below(variables) as i32) * if below(3) == 0 { -1 } else { 1 }).collect()
                }).collect();
                for _ in 0..2 * variables {
                    formula.extend(frontier.iter().cloned());
                    frontier = frontier.iter().map(|clause| {
                        clause.iter().map(|&literal| image[literal.unsigned_abs() as usize] * literal.signum()).collect()
                    }).collect();
                }
            }
            // Pigeons and holes under scrambled identifiers, clauses and literals.
            2 => {
                let holes = 2 + below(7);
                let pigeons = holes + below(3);
                formula = pigeonhole(pigeons.max(2), holes);
                let mut ids: Vec<i32> = (1..=(pigeons.max(2) * holes) as i32).collect();
                for index in (1..ids.len()).rev() {
                    ids.swap(index, below(index + 1));
                }
                for clause in &mut formula {
                    for literal in clause.iter_mut() {
                        *literal = ids[literal.unsigned_abs() as usize - 1] * literal.signum();
                    }
                    let length = clause.len();
                    clause.swap(0, below(length));
                }
                for index in (1..formula.len()).rev() {
                    formula.swap(index, below(index + 1));
                }
            }
            // Clauses around a linear system: parity derivations next to everything else.
            _ => {
                let variables = 8 + below(16);
                for _ in 0..2 + below(variables) {
                    let width = 2 + below(4);
                    let mut row: Vec<i32> = Vec::new();
                    while row.len() < width {
                        let variable = 1 + below(variables) as i32;
                        if !row.contains(&variable) {
                            row.push(variable);
                        }
                    }
                    xor_clauses(&row, below(2) == 0, &mut formula);
                }
                for _ in 0..below(2 * variables) {
                    formula.push((0..2 + below(2)).map(|_| (1 + below(variables) as i32) * if below(2) == 0 { 1 } else { -1 }).collect());
                }
            }
        }
        let run = assert_valid(&formula);
        refuted += usize::from(!run.satisfiable);
        dominances += run.dominances;
        deletions += run.deletions;
        steps += run.trace.lines().filter(|line| line.starts_with("rup ")).count();
        definitions += run.trace.lines().filter(|line| line.starts_with("red ")).count();
    }
    println!("{cases} formulas, {refuted} refuted: {steps} rup, {definitions} red, {dominances} dom, {deletions} del");
    assert!(refuted > cases / 4 && dominances > 0 && deletions > 0 && definitions > 0);
    // A long search with thousands of database reductions, every deletion checked.
    let formula = pigeonhole(8, 7);
    let run = certified(&formula, HybridConfig { symmetry: false, ..HybridConfig::default() });
    assert_eq!(Checker::new(&formula).check(&run.trace), Ok(true));
    println!("search only, 8 pigeons: {} rup, {} del", run.trace.lines().filter(|line| line.starts_with("rup ")).count(), run.deletions);
    assert!(run.deletions > 1000);
}

fn write_dimacs(path: &std::path::Path, formula: &[Vec<i32>]) {
    let variables = formula.iter().flatten().map(|literal| literal.unsigned_abs()).max().unwrap_or(0);
    let mut text = format!("p cnf {variables} {}\n", formula.len());
    for clause in formula {
        for literal in clause {
            text.push_str(&format!("{literal} "));
        }
        text.push_str("0\n");
    }
    std::fs::write(path, text).unwrap();
}

// The official checker itself (VeriPB 3.0.2 at the time of writing):
// cargo test --release --test veripb -- --ignored official
#[test]
#[ignore = "needs the official `veripb` binary on PATH"]
fn the_official_checker_verifies_the_traces() {
    let directory = std::env::temp_dir().join(format!("dpll_check_veripb_{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let official = |name: &str, formula: &[Vec<i32>], trace: &str| -> String {
        let (cnf, pbp) = (directory.join(format!("{name}.cnf")), directory.join(format!("{name}.pbp")));
        write_dimacs(&cnf, formula);
        std::fs::write(&pbp, trace).unwrap();
        let output = std::process::Command::new("veripb").arg("--cnf").arg(&cnf).arg(&pbp).output()
            .expect("`veripb` must be installed and on PATH");
        String::from_utf8_lossy(&output.stdout).into_owned() + &String::from_utf8_lossy(&output.stderr)
    };
    let mut stress = pigeonhole_3sat(10, 9);
    stress.push(vec![1, -1]);
    let petersen: Vec<(usize, usize)> = (0..5)
        .flat_map(|i| [(i, (i + 1) % 5), (i, i + 5), (i + 5, (i + 2) % 5 + 5)])
        .collect();
    let mut tseitin = Vec::new();
    for vertex in 0..10 {
        let incident: Vec<i32> = petersen.iter().enumerate()
            .filter(|(_, &(a, b))| a == vertex || b == vertex)
            .map(|(index, _)| index as i32 + 1).collect();
        xor_clauses(&incident, vertex == 0, &mut tseitin);
    }
    // Symmetry next to a linear system: dominance, definitions and search in one trace.
    let mut mixed = pigeonhole(6, 5);
    xor_clauses(&[31, 32, 33, 34, 35, 36, 37], true, &mut mixed);
    xor_clauses(&[31, 32, 38, 39, 40, 41, 42], false, &mut mixed);
    mixed.push(vec![-33, 1]);
    let complete: Vec<(usize, usize)> = (0..6).flat_map(|a| (a + 1..6).map(move |b| (a, b))).collect();
    let on = HybridConfig::default();
    let off = HybridConfig { symmetry: false, ..on };
    let cases = [
        ("stress_pigeonhole_10_9", stress.clone(), on),
        ("pigeonhole_13_12", pigeonhole(13, 12), on),
        ("colouring_k6_5", colouring(6, &complete, 5), on),
        ("tseitin_petersen", tseitin, on),
        ("pigeonhole_with_parity", mixed, on),
        ("search_pigeonhole_7_6", pigeonhole(7, 6), off),
        ("empty_clause", vec![vec![1, 2], vec![]], on),
        ("satisfiable_pigeonhole_5_5", pigeonhole(5, 5), on),
    ];
    for (name, formula, config) in &cases {
        let run = certified(formula, *config);
        let verdict = official(name, formula, &run.trace);
        let expected = if run.satisfiable { "s VERIFIED NO CONCLUSION" } else { "s VERIFIED UNSATISFIABLE" };
        assert!(verdict.contains(expected), "{name}: {verdict}");
        println!("{name}: {expected} ({} bytes, {} dom, {} del)", run.trace.len(), run.dominances, run.deletions);
    }
    // The checker is not being lenient: without its dominance step the predicates fail.
    let run = certified(&stress, on);
    let lines: Vec<&str> = run.trace.lines().collect();
    let first = lines.iter().position(|line| line.starts_with("dom ")).unwrap();
    let stripped: String = lines.iter().enumerate().filter(|&(index, _)| !(first..first + 5).contains(&index))
        .map(|(_, line)| format!("{line}\n")).collect();
    let verdict = official("stress_without_a_dominance_step", &stress, &stripped);
    assert!(!verdict.contains("VERIFIED") && verdict.contains("Error"), "{verdict}");
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn learned_and_deleted_clauses_are_logged_as_rup_and_deletion_steps() {
    // Without the symmetry layer the refutation is all search, with database reductions.
    let formula = pigeonhole(7, 6);
    let run = certified(&formula, HybridConfig { symmetry: false, ..HybridConfig::default() });
    assert!(!run.satisfiable);
    assert_eq!((run.dominances, run.generators), (0, 0));
    assert!(run.deletions > 0, "The search never reduced its database");
    assert_eq!(run.trace.lines().filter(|line| line.starts_with("del spec ")).count() as u64, run.deletions);
    assert!(!run.trace.contains("def_order"));
    assert_eq!(Checker::new(&formula).check(&run.trace), Ok(true));
    // Deleting a clause that was never derived is an error, as it is for the real checker.
    let bogus = run.trace.replacen("\nrup ", "\ndel spec 1 x1 1 x2 1 x3 >= 1;\nrup ", 1);
    assert!(Checker::new(&formula).check(&bogus).is_err());
}

#[test]
fn a_forged_dominance_step_is_rejected() {
    let formula = pigeonhole(5, 4);
    let run = certified(&formula, HybridConfig::default());
    assert_eq!(Checker::new(&formula).check(&run.trace), Ok(true));
    // Redirect two images of the first witness: it stops being a symmetry of the formula.
    let line = run.trace.lines().find(|line| line.starts_with("dom ")).unwrap();
    let parts: Vec<&str> = line.split(" : ").collect();
    let mut tokens: Vec<String> = parts[1].split_whitespace().map(str::to_owned).collect();
    let last = tokens.len() - 1;
    tokens.swap(2, last);
    let forged = run.trace.replacen(line, &format!("{} : {} : subproof", parts[0], tokens.join(" ")), 1);
    assert!(Checker::new(&formula).check(&forged).is_err());
    // Dropping the dominance steps leaves their predicates without justification.
    let lines: Vec<&str> = run.trace.lines().collect();
    let mut stripped = String::new();
    let mut index = 0;
    while index < lines.len() {
        if lines[index].starts_with("dom ") {
            index += 5;
            continue;
        }
        stripped.push_str(lines[index]);
        stripped.push('\n');
        index += 1;
    }
    assert!(Checker::new(&formula).check(&stripped).is_err());
    // A symmetry-breaking unit nobody justified is not RUP either.
    let unjustified = run.trace.replacen("def_order lex", "rup 1 ~x1 >= 1;\ndef_order lex", 1);
    assert!(Checker::new(&formula).check(&unjustified).is_err());
}

#[test]
fn the_trace_is_framed_as_a_veripb_3_proof() {
    let formula = pigeonhole(4, 3);
    let run = certified(&formula, HybridConfig::default());
    let lines: Vec<&str> = run.trace.lines().collect();
    assert_eq!(lines[0], "pseudo-Boolean proof version 3.0");
    assert_eq!(lines[1], format!("f {};", formula.len()));
    assert_eq!(lines[2], "def_order lex");
    assert_eq!(&lines[lines.len() - 3..], ["output NONE;", "conclusion UNSAT : -1;", "end pseudo-Boolean proof;"]);
    assert!(lines.contains(&"rup >= 1;"));
    // Every rule ends in a semicolon; the only other lines open a block.
    let openers = ["def_order lex", "vars", "def", "transitivity", "proof", "proofgoal #1", "proofgoal #2"];
    for line in &lines[1..] {
        assert!(line.ends_with(';') || line.ends_with(" : subproof") || openers.contains(line), "{line}");
    }

    // A satisfiable run claims nothing.
    let run = certified(&pigeonhole(3, 3), HybridConfig::default());
    assert!(run.satisfiable);
    assert!(run.trace.ends_with("output NONE;\nconclusion NONE;\nend pseudo-Boolean proof;\n"));

    // An empty input clause is already the contradiction: the trace only points at it.
    let formula = vec![vec![1, 2], vec![]];
    let run = certified(&formula, HybridConfig::default());
    assert_eq!(Checker::new(&formula).check(&run.trace), Ok(true));
}
