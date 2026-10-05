#![forbid(unsafe_code)]
#![deny(warnings)]

#[allow(dead_code)]
#[path = "../src/hybrid.rs"]
mod hybrid;

use hybrid::proof::{Proof, ProofFormat};
use hybrid::{solve_hybrid, solve_hybrid_with_proof, verify_model, HybridConfig};

// An independent forward DRAT checker, deliberately naive: it shares no code with the
// engine. A line is accepted if the clause is RUP, or RAT on its first literal.
struct Checker {
    clauses: Vec<Vec<i32>>,
}

#[derive(Debug, PartialEq)]
enum Step {
    Add(Vec<i32>),
    Delete(Vec<i32>),
}

// A repeated literal must not count twice when looking for a unit clause.
fn distinct(clause: &[i32]) -> Vec<i32> {
    let mut literals = clause.to_vec();
    literals.sort_unstable();
    literals.dedup();
    literals
}

impl Checker {
    fn new(formula: &[Vec<i32>]) -> Self {
        Self { clauses: formula.iter().map(|clause| distinct(clause)).collect() }
    }
    // Unit propagation from the given assumptions; true if it reaches a conflict.
    fn refutes(&self, assumptions: &[i32]) -> bool {
        // 0 unknown, 1 true, -1 false, indexed by variable.
        let last = self.clauses.iter().flatten().chain(assumptions).map(|literal| literal.unsigned_abs()).max().unwrap_or(0);
        let mut values = vec![0_i8; last as usize + 1];
        for &literal in assumptions {
            let wanted = if literal > 0 { 1 } else { -1 };
            if values[literal.unsigned_abs() as usize] == -wanted {
                return true;
            }
            values[literal.unsigned_abs() as usize] = wanted;
        }
        loop {
            let mut changed = false;
            for clause in &self.clauses {
                let mut open = None;
                let mut pending = 0;
                let mut satisfied = false;
                for &literal in clause {
                    match values[literal.unsigned_abs() as usize] * literal.signum() as i8 {
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
            if !changed {
                return false;
            }
        }
    }
    fn rup(&self, clause: &[i32]) -> bool {
        let negated: Vec<i32> = clause.iter().map(|literal| -literal).collect();
        self.refutes(&negated)
    }
    fn rat(&self, clause: &[i32]) -> bool {
        let Some(&pivot) = clause.first() else { return false; };
        self.clauses.iter().filter(|other| other.contains(&-pivot)).all(|other| {
            let mut resolvent = clause.to_vec();
            resolvent.extend(other.iter().copied().filter(|&literal| literal != -pivot));
            self.rup(&resolvent)
        })
    }
    // Checks every step in order. `Ok(true)` means the empty clause was derived.
    fn check(&mut self, steps: &[Step]) -> Result<bool, String> {
        let mut refuted = false;
        for (index, step) in steps.iter().enumerate() {
            match step {
                Step::Add(clause) => {
                    if !self.rup(clause) && !self.rat(clause) {
                        return Err(format!("step {index}: {clause:?} is neither RUP nor RAT"));
                    }
                    refuted |= clause.is_empty();
                    self.clauses.push(distinct(clause));
                }
                Step::Delete(clause) => {
                    let wanted = distinct(clause);
                    let position = self.clauses.iter().position(|known| *known == wanted);
                    match position {
                        Some(position) => {
                            self.clauses.swap_remove(position);
                        }
                        None => return Err(format!("step {index}: deleted clause {clause:?} is absent")),
                    }
                }
            }
        }
        Ok(refuted)
    }
}

fn parse_text(bytes: &[u8]) -> Vec<Step> {
    std::str::from_utf8(bytes).unwrap().lines().map(|line| {
        let deletion = line.starts_with("d ");
        let mut literals: Vec<i32> = line.trim_start_matches("d ").split_whitespace()
            .map(|token| token.parse().unwrap()).collect();
        assert_eq!(literals.pop(), Some(0), "Every line ends with 0");
        if deletion { Step::Delete(literals) } else { Step::Add(literals) }
    }).collect()
}

fn parse_binary(bytes: &[u8]) -> Vec<Step> {
    let mut steps = Vec::new();
    let mut position = 0;
    while position < bytes.len() {
        let deletion = match bytes[position] {
            b'a' => false,
            b'd' => true,
            other => panic!("Unexpected step marker {other}"),
        };
        position += 1;
        let mut literals = Vec::new();
        loop {
            let (mut code, mut shift) = (0_u64, 0);
            loop {
                let byte = bytes[position];
                position += 1;
                code |= u64::from(byte & 127) << shift;
                shift += 7;
                if byte & 128 == 0 {
                    break;
                }
            }
            if code == 0 {
                break;
            }
            let variable = (code / 2) as i32;
            literals.push(if code & 1 == 1 { -variable } else { variable });
        }
        steps.push(if deletion { Step::Delete(literals) } else { Step::Add(literals) });
    }
    steps
}

// Solves with a trace and returns (has a model, checked steps, proof statistics).
fn certified(formula: &[Vec<i32>], format: ProofFormat) -> (bool, Vec<Step>, u64, u64) {
    let mut proof = Proof::new(Vec::new(), format);
    let report = solve_hybrid_with_proof(formula, HybridConfig::default(), &mut proof);
    // A symmetry predicate is not RAT, so the layer must stay out of a certified run.
    assert_eq!(report.symmetry.clauses, 0);
    let (additions, deletions) = (proof.additions(), proof.deletions());
    let expected_bytes = proof.bytes();
    let bytes = proof.finish().unwrap();
    assert_eq!(bytes.len() as u64, expected_bytes);
    let steps = match format {
        ProofFormat::Text => parse_text(&bytes),
        ProofFormat::Binary => parse_binary(&bytes),
    };
    assert_eq!(steps.len() as u64, additions + deletions);
    if let Some(model) = &report.model {
        assert!(verify_model(formula, model));
    }
    (report.model.is_some(), steps, additions, deletions)
}

fn assert_certified_unsat(formula: &[Vec<i32>]) -> (u64, u64) {
    let (satisfiable, steps, additions, deletions) = certified(formula, ProofFormat::Text);
    assert!(!satisfiable);
    assert_eq!(steps.last(), Some(&Step::Add(Vec::new())), "A refutation ends with the empty clause");
    assert_eq!(Checker::new(formula).check(&steps), Ok(true));
    (additions, deletions)
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
fn search_refutations_are_valid_drat() {
    for (pigeons, holes) in [(3, 2), (4, 3), (5, 4), (6, 5)] {
        let (additions, _) = assert_certified_unsat(&pigeonhole(pigeons, holes));
        assert!(additions > 0);
    }
    // Contradictory units and an explicit empty clause need no search at all.
    assert_certified_unsat(&[vec![1], vec![-1]]);
    let (satisfiable, steps, ..) = certified(&[vec![]], ProofFormat::Text);
    assert!(!satisfiable && steps.is_empty());
}

#[test]
fn deleted_clauses_are_logged_and_the_trace_stays_valid() {
    // Long enough for the database reduction to run several times.
    let formula = pigeonhole(7, 6);
    let (additions, deletions) = assert_certified_unsat(&formula);
    assert!(additions > 500 && deletions > 0, "{additions} additions, {deletions} deletions");

    let mut seed = 0x243f_6a88_85a3_08d3;
    let mut refuted = 0;
    for _ in 0..8 {
        let formula: Vec<Vec<i32>> = (0..260).map(|_| {
            (0..3).map(|_| {
                let variable = 1 + (next(&mut seed) % 52) as i32;
                if next(&mut seed) & 1 == 0 { variable } else { -variable }
            }).collect()
        }).collect();
        if solve_hybrid(&formula, HybridConfig::default()).model.is_none() {
            assert_certified_unsat(&formula);
            refuted += 1;
        }
    }
    assert!(refuted > 0);
}

#[test]
fn gaussian_refutations_are_certified_without_search() {
    // Tseitin on the Petersen graph with one odd vertex: no clause betrays the parity clash.
    let edges: Vec<(usize, usize)> = (0..5)
        .flat_map(|i| [(i, (i + 1) % 5), (i, i + 5), (i + 5, (i + 2) % 5 + 5)])
        .collect();
    let mut formula = Vec::new();
    for vertex in 0..10 {
        let incident: Vec<i32> = edges.iter().enumerate()
            .filter(|(_, &(a, b))| a == vertex || b == vertex)
            .map(|(index, _)| index as i32 + 1).collect();
        assert_eq!(incident.len(), 3);
        xor_clauses(&incident, vertex == 0, &mut formula);
    }
    let mut proof = Proof::new(Vec::new(), ProofFormat::Text);
    let report = solve_hybrid_with_proof(&formula, HybridConfig::default(), &mut proof);
    assert!(report.model.is_none() && report.parity.refuted);
    assert_eq!(report.cdcl.decisions, 0);
    let steps = parse_text(&proof.finish().unwrap());
    // The derivation defines auxiliaries above every original identifier.
    assert!(steps.iter().any(|step| matches!(step, Step::Add(clause) if clause.iter().any(|l| l.unsigned_abs() > 15))));
    assert_eq!(steps.last(), Some(&Step::Add(Vec::new())));
    assert_eq!(Checker::new(&formula).check(&steps), Ok(true));
}

#[test]
fn injected_parity_rows_are_derived_before_the_search_uses_them() {
    // x1 xor x2 = 1 and x1 xor x2 xor x3 = 0 force x3, which two clauses then contradict.
    let mut formula = Vec::new();
    xor_clauses(&[1, 2], true, &mut formula);
    xor_clauses(&[1, 2, 3], false, &mut formula);
    formula.push(vec![-3, 4]);
    formula.push(vec![-3, -4]);
    assert_certified_unsat(&formula);

    // Two wide equations whose sum is chained through auxiliaries, then contradicted.
    let mut formula = Vec::new();
    xor_clauses(&[1, 2, 3, 4, 5, 6, 7], true, &mut formula);
    xor_clauses(&[1, 2, 8, 9, 10, 11, 12], false, &mut formula);
    for variable in 3..=12 {
        formula.push(vec![-variable]);
    }
    let mut proof = Proof::new(Vec::new(), ProofFormat::Text);
    let report = solve_hybrid_with_proof(&formula, HybridConfig::default(), &mut proof);
    assert!(report.model.is_none());
    let steps = parse_text(&proof.finish().unwrap());
    assert_eq!(Checker::new(&formula).check(&steps), Ok(true));
}

#[test]
fn mixed_parity_and_clause_refutations_are_valid_drat() {
    let mut seed = 0xbb67_ae85_84ca_a73b;
    let (mut refuted, mut reduced) = (0, 0);
    for case in 0..220 {
        let variables = 4 + (next(&mut seed) % 7) as i32;
        let mut formula = Vec::new();
        for _ in 0..2 + next(&mut seed) % variables as u64 {
            let width = 1 + (next(&mut seed) % [3, 4, 6][case % 3]) as usize;
            let mut row: Vec<i32> = Vec::new();
            while row.len() < width.min(variables as usize) {
                let variable = 1 + (next(&mut seed) % variables as u64) as i32;
                if !row.contains(&variable) {
                    row.push(variable);
                }
            }
            xor_clauses(&row, next(&mut seed) & 1 == 1, &mut formula);
        }
        for _ in 0..next(&mut seed) % 6 {
            formula.push((0..1 + next(&mut seed) % 3).map(|_| {
                let variable = 1 + (next(&mut seed) % variables as u64) as i32;
                if next(&mut seed) & 1 == 0 { variable } else { -variable }
            }).collect());
        }
        let (satisfiable, steps, ..) = certified(&formula, ProofFormat::Text);
        // The certified run and the plain pipeline must agree on the answer.
        assert_eq!(satisfiable, solve_hybrid(&formula, HybridConfig::default()).model.is_some());
        // Every step must be justified, whether or not the instance ends up refuted.
        let outcome = Checker::new(&formula).check(&steps);
        assert_eq!(outcome, Ok(!satisfiable), "case {case}: {formula:?}");
        refuted += usize::from(!satisfiable);
        reduced += usize::from(steps.iter().any(|step| matches!(step, Step::Add(clause) if clause.len() == 3)));
    }
    assert!(refuted > 40 && reduced > 40, "{refuted} refutations, {reduced} with derivations");
}

#[test]
fn a_truncated_trace_is_rejected_by_the_checker() {
    let formula = pigeonhole(5, 4);
    let (_, steps, ..) = certified(&formula, ProofFormat::Text);
    assert_eq!(Checker::new(&formula).check(&steps), Ok(true));
    // The empty clause alone is not a consequence the checker can see.
    let truncated = [Step::Add(Vec::new())];
    assert!(Checker::new(&formula).check(&truncated).is_err());
    // Neither is a clause the formula does not imply.
    assert!(Checker::new(&formula).check(&[Step::Add(vec![1])]).is_err());
    assert!(Checker::new(&formula).check(&[Step::Delete(vec![1, 2, 3])]).is_err());
}

#[test]
fn binary_and_text_traces_carry_the_same_steps() {
    for formula in [pigeonhole(5, 4), pigeonhole(6, 5)] {
        let (_, text, ..) = certified(&formula, ProofFormat::Text);
        let (_, binary, ..) = certified(&formula, ProofFormat::Binary);
        assert_eq!(text, binary);
    }
}

#[test]
fn a_trace_larger_than_the_buffer_arrives_complete() {
    // Thousands of learned clauses: the buffer is drained several times along the way.
    let formula = pigeonhole(7, 6);
    let mut proof = Proof::new(Vec::new(), ProofFormat::Text);
    let report = solve_hybrid_with_proof(&formula, HybridConfig::default(), &mut proof);
    assert!(report.model.is_none());
    let expected = proof.bytes();
    let steps = proof.additions() + proof.deletions();
    let bytes = proof.finish().unwrap();
    assert!(bytes.len() > 1 << 16);
    assert_eq!(bytes.len() as u64, expected);
    assert_eq!(parse_text(&bytes).len() as u64, steps);
}

#[test]
fn satisfiable_runs_return_a_verified_model_and_valid_steps() {
    let mut formula = Vec::new();
    xor_clauses(&[1, 2, 3], true, &mut formula);
    xor_clauses(&[2, 3, 4], false, &mut formula);
    formula.push(vec![4, 5, 6]);
    let (satisfiable, steps, ..) = certified(&formula, ProofFormat::Binary);
    assert!(satisfiable);
    assert_eq!(Checker::new(&formula).check(&steps), Ok(false));
}
