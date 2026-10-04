#![forbid(unsafe_code)]
#![deny(warnings)]

// VSIDS inicia con las ocurrencias y suma 1.0 por variable distinta de la cláusula conflictiva.
// Después de cada conflicto real, todos los puntajes se multiplican por DECAY = 0.95.
// Los puntajes y los watches persisten al retroceder; Undo restaura las asignaciones.
pub fn solve_dpll(formula: Vec<Vec<i32>>, mut assignment: Vec<i32>) -> Option<Vec<i32>> {
    use std::collections::VecDeque;

    const DECAY: f64 = 0.95;

    enum Undo {
        Assigned(usize),
    }

    struct State {
        literals: Vec<usize>,
        offsets: Vec<usize>,
        watches: Vec<Vec<usize>>,
        watched_positions: Vec<[usize; 2]>,
        values: Vec<i8>,
        scores: Vec<f64>,
        conflict_clause: Option<usize>,
        units: VecDeque<(usize, usize)>,
        trail: Vec<usize>,
        propagation_head: usize,
        undo: Vec<Undo>,
    }

    impl State {
        fn value(&self, literal: usize) -> i8 {
            let value = self.values[literal >> 1];
            if literal & 1 == 0 { value } else { -value }
        }

        fn assign(&mut self, literal: usize) -> bool {
            let variable = literal >> 1;
            let value = if literal & 1 == 0 { 1 } else { -1 };
            if self.values[variable] != 0 {
                return self.values[variable] == value;
            }
            self.values[variable] = value;
            self.trail.push(literal);
            self.undo.push(Undo::Assigned(variable));
            true
        }

        fn propagate(&mut self) -> bool {
            loop {
                while let Some((literal, clause)) = self.units.pop_front() {
                    if !self.assign(literal) {
                        self.conflict_clause = Some(clause);
                        return false;
                    }
                }
                if self.propagation_head == self.trail.len() {
                    return true;
                }

                let false_literal = self.trail[self.propagation_head] ^ 1;
                self.propagation_head += 1;
                let mut index = 0;

                // Solo se visitan cláusulas que observan el literal recién falsificado.
                while index < self.watches[false_literal].len() {
                    let clause = self.watches[false_literal][index];
                    let positions = self.watched_positions[clause];
                    let slot = if self.literals[positions[0]] == false_literal {
                        0
                    } else {
                        debug_assert_eq!(self.literals[positions[1]], false_literal);
                        1
                    };
                    let other = self.literals[positions[slot ^ 1]];
                    let other_value = self.value(other);

                    if other_value == 1 {
                        index += 1;
                        continue;
                    }

                    let replacement = (self.offsets[clause]..self.offsets[clause + 1])
                        .find(|&position| {
                            position != positions[0]
                                && position != positions[1]
                                && self.value(self.literals[position]) != -1
                        });

                    if let Some(position) = replacement {
                        let new_literal = self.literals[position];
                        self.watched_positions[clause][slot] = position;
                        self.watches[false_literal].swap_remove(index);
                        self.watches[new_literal].push(clause);
                        // La entrada intercambiada todavía debe procesarse.
                        continue;
                    }

                    if other_value == -1 {
                        self.conflict_clause = Some(clause);
                        return false;
                    }

                    self.units.push_back((other, clause));
                    index += 1;
                }
            }
        }

        fn update_vsids(&mut self) {
            let Some(clause) = self.conflict_clause.take() else {
                return;
            };
            let start = self.offsets[clause];
            let end = self.offsets[clause + 1];
            for index in start..end {
                let variable = self.literals[index] >> 1;
                if self.literals[start..index]
                    .iter()
                    .any(|&literal| literal >> 1 == variable)
                {
                    continue;
                }
                self.scores[variable] += 1.0;
            }
            for score in &mut self.scores {
                *score *= DECAY;
            }
        }

        fn choose_literal(&self) -> Option<usize> {
            let mut best_variable = None;
            let mut best_score = f64::NEG_INFINITY;
            for (variable, &score) in self.scores.iter().enumerate() {
                if self.values[variable] == 0 && score > best_score {
                    best_variable = Some(variable);
                    best_score = score;
                }
            }
            // Empates por identificador menor; polaridad positiva primero.
            best_variable.map(|variable| variable << 1)
        }

        fn rollback(&mut self, trail_mark: usize, undo_mark: usize) {
            while self.undo.len() > undo_mark {
                match self.undo.pop().expect("Undo entry exists") {
                    Undo::Assigned(variable) => self.values[variable] = 0,
                }
            }
            self.trail.truncate(trail_mark);
            self.propagation_head = trail_mark;
            self.units.clear();
            // Desasignar no invalida los watches; no es necesario restaurarlos.
        }
    }

    struct Decision {
        literal: usize,
        trail_mark: usize,
        undo_mark: usize,
        tried_opposite: bool,
    }

    if formula.iter().any(Vec::is_empty) {
        return None;
    }
    if formula.is_empty() {
        return Some(assignment);
    }

    let mut variables: Vec<u32> = formula
        .iter()
        .flatten()
        .map(|&literal| {
            assert!(literal != 0 && literal != i32::MIN, "Invalid SAT literal");
            literal.unsigned_abs()
        })
        .collect();
    variables.sort_unstable();
    variables.dedup();
    let literal_count = variables.len().checked_mul(2).expect("Too many variables");
    let mut state = State {
        literals: Vec::new(),
        offsets: Vec::with_capacity(formula.len() + 1),
        watches: vec![Vec::new(); literal_count],
        watched_positions: Vec::with_capacity(formula.len()),
        values: vec![0; variables.len()],
        scores: vec![0.0; variables.len()],
        conflict_clause: None,
        units: VecDeque::new(),
        trail: Vec::with_capacity(variables.len()),
        propagation_head: 0,
        undo: Vec::with_capacity(variables.len()),
    };

    state.offsets.push(0);
    for (clause_index, clause) in formula.into_iter().enumerate() {
        let start = state.literals.len();
        for literal in clause {
            let variable = variables.binary_search(&literal.unsigned_abs())
                .expect("Variable was indexed");
            let packed = (variable << 1) | usize::from(literal < 0);
            state.literals.push(packed);
            state.scores[variable] += 1.0;
        }
        let end = state.literals.len();
        state.offsets.push(end);
        if end - start == 1 {
            state.watched_positions.push([start, start]);
            state.units.push_back((state.literals[start], clause_index));
        } else {
            state.watched_positions.push([start, start + 1]);
            state.watches[state.literals[start]].push(clause_index);
            state.watches[state.literals[start + 1]].push(clause_index);
        }
    }

    let mut decisions: Vec<Decision> = Vec::with_capacity(variables.len());
    let mut consistent = state.propagate();

    loop {
        if consistent {
            // Un modelo completo evita recorrer todas las cláusulas en cada decisión.
            let Some(literal) = state.choose_literal() else {
                // assignment conserva la semántica de prefijo del código original.
                assignment.extend(state.trail.iter().map(|&literal| {
                    let variable = variables[literal >> 1] as i32;
                    if literal & 1 == 0 { variable } else { -variable }
                }));
                return Some(assignment);
            };
            decisions.push(Decision {
                literal,
                trail_mark: state.trail.len(),
                undo_mark: state.undo.len(),
                tried_opposite: false,
            });
            consistent = state.assign(literal) && state.propagate();
        } else {
            // Consumir el conflicto evita repetir bump/decay al retirar varios niveles.
            state.update_vsids();
            let decision = decisions.last_mut()?;
            state.rollback(decision.trail_mark, decision.undo_mark);
            if decision.tried_opposite {
                decisions.pop();
            } else {
                decision.tried_opposite = true;
                consistent = state.assign(decision.literal ^ 1) && state.propagate();
            }
        }
    }
}

pub type Clause = Vec<i32>;
pub type Formula = Vec<Clause>;

fn main() {
    use std::hint::black_box;
    use std::time::Instant;

    println!("case,n,clauses,status,elapsed_ms");

    for n in [10_i32, 100, 1000] {
        let mut chain: Formula = (1..n).rev().map(|v| vec![-v, v + 1]).collect();
        chain.push(vec![1]);

        let independent: Formula = (1..n - 1)
            .step_by(3)
            .map(|v| vec![v, v + 1, v + 2])
            .collect();

        let rollback: Formula = (1..n - 2)
            .step_by(3)
            .flat_map(|v| {
                [vec![v, v + 1], vec![-v, v + 2], vec![-v, -(v + 2)]]
            })
            .collect();

        for (name, formula) in [
            ("reverse_chain", chain),
            ("independent", independent),
            ("rollback", rollback),
        ] {
            let clause_count = formula.len();
            let input = formula.clone();

            let start = Instant::now();
            let result = black_box(solve_dpll(black_box(input), Vec::new()));
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            // La verificación queda fuera del intervalo cronometrado.
            let model = result.expect("Benchmark formula must be satisfiable");
            assert!(formula.iter().all(|clause| {
                clause.iter().any(|literal| model.contains(literal))
            }));
            for &literal in &model {
                assert!(!model.contains(&-literal));
            }

            println!("{name},{n},{clause_count},SAT,{elapsed_ms:.6}");
        }
    }
}
