// Entrada pseudo-booleana (formato de la competicion PB), con objetivo `min:` opcional. Las
// restricciones lineales se traducen a CNF con un sumador binario; quien la resuelve y comprueba
// es solver.rs.
//
// Que se comprueba y que no. Un modelo se evalua contra las restricciones del fichero, sin
// pasar por la CNF. Un UNSAT se comprueba con VeriPB, pero contra la CNF traducida: el
// traductor queda dentro de lo que hay que creer, y por eso tests/motor_opb.rs lo compara con
// la fuerza bruta. Con objetivo, el optimo es la ultima solucion mas la refutacion de
// «coste < optimo», que tambien pasa por VeriPB.
use crate::dimacs::Formula;

// a1 l1 + a2 l2 + ... >= degree, con literales DIMACS y coeficientes de cualquier signo.
#[derive(Clone, Debug, PartialEq)]
pub struct Constraint {
    pub terms: Vec<(i64, i32)>,
    pub degree: i64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Problem {
    pub variables: u32,
    pub objective: Option<Vec<(i64, i32)>>,
    pub constraints: Vec<Constraint>,
}

fn parse_terms(tokens: &[&str], variables: &mut u32) -> Result<Vec<(i64, i32)>, String> {
    let mut terms = Vec::new();
    let mut coefficient: Option<i64> = None;
    for &token in tokens {
        let (negated, name) = token.strip_prefix('~').map_or((false, token), |rest| (true, rest));
        if let Some(id) = name.strip_prefix('x') {
            let id: i32 = id.parse().ok().filter(|&id| id > 0).ok_or(format!("Variable no valida: {token}"))?;
            *variables = (*variables).max(id as u32);
            let literal = if negated { -id } else { id };
            // En este formato, dos variables seguidas son un producto, no una suma.
            let coefficient = coefficient.take().ok_or(format!("Termino no lineal o sin coeficiente en {token}"))?;
            terms.push((coefficient, literal));
        } else {
            if coefficient.is_some() {
                return Err(format!("Coeficiente sin variable antes de {token}"));
            }
            coefficient = Some(token.parse().map_err(|_| format!("Coeficiente no valido: {token}"))?);
        }
    }
    if coefficient.is_some() {
        return Err("Coeficiente sin variable al final de un termino".into());
    }
    Ok(terms)
}

pub fn parse(text: &str) -> Result<Problem, String> {
    let mut problem = Problem::default();
    let body: String = text.lines().filter(|line| !line.trim_start().starts_with('*')).collect::<Vec<_>>().join(" ");
    let mut statements: Vec<&str> = body.split(';').collect();
    if !statements.pop().unwrap_or("").trim().is_empty() {
        return Err("La ultima restriccion no termina en ;".into());
    }
    for statement in statements {
        let statement = statement.trim();
        if statement.is_empty() {
            continue;
        }
        if let Some(objective) = statement.strip_prefix("min:") {
            let tokens: Vec<&str> = objective.split_whitespace().collect();
            if problem.objective.is_some() {
                return Err("Mas de un objetivo".into());
            }
            problem.objective = Some(parse_terms(&tokens, &mut problem.variables)?);
            continue;
        }
        let tokens: Vec<&str> = statement.split_whitespace().collect();
        let [left @ .., relation, degree] = tokens.as_slice() else {
            return Err(format!("Restriccion no valida: {statement}"));
        };
        let degree: i64 = degree.parse().map_err(|_| format!("Termino independiente no valido: {degree}"))?;
        let terms = parse_terms(left, &mut problem.variables)?;
        let flip = |terms: &[(i64, i32)]| -> Result<Vec<(i64, i32)>, String> {
            terms.iter().map(|&(a, l)| a.checked_neg().map(|a| (a, l)).ok_or("Coeficiente fuera de rango".to_string())).collect()
        };
        let negative = degree.checked_neg().ok_or("Termino independiente fuera de rango")?;
        match *relation {
            ">=" => problem.constraints.push(Constraint { terms, degree }),
            "<=" => problem.constraints.push(Constraint { terms: flip(&terms)?, degree: negative }),
            "=" => {
                problem.constraints.push(Constraint { terms: flip(&terms)?, degree: negative });
                problem.constraints.push(Constraint { terms, degree });
            }
            other => return Err(format!("Relacion no valida: {other}. Solo >=, <= y = sobre terminos lineales")),
        }
    }
    Ok(problem)
}

fn value(model: &[bool], literal: i32) -> bool {
    model.get(literal.unsigned_abs() as usize).copied().unwrap_or(false) == (literal > 0)
}

// Coste o suma de los terminos bajo el modelo, sin pasar por la CNF.
pub fn evaluate(terms: &[(i64, i32)], model: &[bool]) -> i128 {
    terms.iter().filter(|&&(_, literal)| value(model, literal)).map(|&(a, _)| i128::from(a)).sum()
}

pub fn satisfies(problem: &Problem, model: &[bool]) -> bool {
    problem.constraints.iter().all(|constraint| evaluate(&constraint.terms, model) >= i128::from(constraint.degree))
}

// Traductor a CNF. Las salidas de cada sumador quedan definidas en las dos direcciones.
struct Encoder {
    next: i32,
    formula: Formula,
}
impl Encoder {
    fn fresh(&mut self) -> i32 {
        self.next += 1;
        self.next - 1
    }
    fn half_adder(&mut self, a: i32, b: i32) -> (i32, i32) {
        let (sum, carry) = (self.fresh(), self.fresh());
        self.formula.extend([
            vec![-a, -b, -sum], vec![a, b, -sum], vec![-a, b, sum], vec![a, -b, sum],
            vec![-a, -b, carry], vec![a, -carry], vec![b, -carry],
        ]);
        (sum, carry)
    }
    fn full_adder(&mut self, a: i32, b: i32, c: i32) -> (i32, i32) {
        let (sum, carry) = (self.fresh(), self.fresh());
        self.formula.extend([
            vec![a, b, c, -sum], vec![a, -b, -c, -sum], vec![-a, b, -c, -sum], vec![-a, -b, c, -sum],
            vec![-a, -b, -c, sum], vec![-a, b, c, sum], vec![a, -b, c, sum], vec![a, b, -c, sum],
            vec![-a, -b, carry], vec![-a, -c, carry], vec![-b, -c, carry],
            vec![a, b, -carry], vec![a, c, -carry], vec![b, c, -carry],
        ]);
        (sum, carry)
    }
    // Cada bit de un coeficiente va a la columna de su peso; cada columna se comprime hasta
    // un solo bit y sus acarreos pasan a la siguiente.
    fn sum_bits(&mut self, terms: &[(u128, i32)]) -> Vec<i32> {
        let mut columns: Vec<Vec<i32>> = Vec::new();
        for &(coefficient, literal) in terms {
            for bit in 0..128 - coefficient.leading_zeros() as usize {
                if coefficient >> bit & 1 == 1 {
                    if columns.len() <= bit {
                        columns.resize(bit + 1, Vec::new());
                    }
                    columns[bit].push(literal);
                }
            }
        }
        let mut bits = Vec::new();
        let mut index = 0;
        while index < columns.len() {
            let mut column = std::mem::take(&mut columns[index]);
            while column.len() > 1 {
                let carry;
                let sum;
                if column.len() == 2 {
                    (sum, carry) = self.half_adder(column[0], column[1]);
                    column.drain(..2);
                } else {
                    (sum, carry) = self.full_adder(column[0], column[1], column[2]);
                    column.drain(..3);
                }
                column.push(sum);
                if columns.len() <= index + 1 {
                    columns.push(Vec::new());
                }
                columns[index + 1].push(carry);
            }
            // Columna vacia: 0 como literal 0, que `at_least` trata como falso.
            bits.push(column.first().copied().unwrap_or(0));
            index += 1;
        }
        bits
    }
    fn constraint(&mut self, constraint: &Constraint) -> Result<(), String> {
        let overflow = || "Los coeficientes de una restriccion no caben en 127 bits".to_string();
        let mut degree = i128::from(constraint.degree);
        let mut terms: Vec<(u128, i32)> = Vec::with_capacity(constraint.terms.len());
        let mut total: u128 = 0;
        for &(coefficient, literal) in &constraint.terms {
            if coefficient == 0 {
                continue;
            }
            // -a l = a (no l) - a
            let (magnitude, literal) = if coefficient < 0 {
                degree -= i128::from(coefficient);
                (u128::from(coefficient.unsigned_abs()), -literal)
            } else {
                (coefficient as u128, literal)
            };
            total = total.checked_add(magnitude).filter(|&sum| sum < 1 << 126).ok_or_else(overflow)?;
            terms.push((magnitude, literal));
        }
        if degree <= 0 {
            return Ok(());
        }
        let degree = degree as u128;
        if total < degree {
            self.formula.push(Vec::new());
            return Ok(());
        }
        if terms.iter().all(|&(magnitude, _)| magnitude >= degree) {
            self.formula.push(terms.iter().map(|&(_, literal)| literal).collect());
            return Ok(());
        }
        let bits = self.sum_bits(&terms);
        self.at_least(&bits, degree);
        Ok(())
    }
    // suma >= k si y solo si, en cada bit a 1 de k, ese bit de la suma vale 1 o la suma tiene
    // un 1 mas arriba en un bit a 0 de k.
    fn at_least(&mut self, bits: &[i32], degree: u128) {
        for index in 0..bits.len().max(128 - degree.leading_zeros() as usize) {
            if degree >> index & 1 == 0 {
                continue;
            }
            let mut clause: Vec<i32> = bits.get(index).copied().into_iter().collect();
            for higher in index + 1..bits.len() {
                if degree >> higher & 1 == 0 {
                    clause.push(bits[higher]);
                }
            }
            clause.retain(|&literal| literal != 0);
            self.formula.push(clause);
        }
    }
}

pub fn encode(problem: &Problem, bound: Option<i128>) -> Result<(Formula, u32), String> {
    let mut encoder = Encoder { next: problem.variables as i32 + 1, formula: Formula::new() };
    for constraint in &problem.constraints {
        encoder.constraint(constraint)?;
    }
    if let (Some(objective), Some(bound)) = (&problem.objective, bound) {
        // coste <= bound, como -coste >= -bound.
        let degree = i64::try_from(-bound).map_err(|_| "Coste fuera de rango")?;
        let terms: Option<Vec<(i64, i32)>> = objective.iter().map(|&(a, l)| a.checked_neg().map(|a| (a, l))).collect();
        encoder.constraint(&Constraint { terms: terms.ok_or("Coeficiente fuera de rango")?, degree })?;
    }
    Ok((encoder.formula, (encoder.next - 1) as u32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid::{solve_hybrid, HybridConfig};

    #[test]
    fn the_format_is_read_with_its_three_relations_and_negated_literals() {
        let problem = parse("* comentario\nmin: +1 x1 +2 ~x2 ;\n+1 x1 -3 x2 >= -2 ;\n2 x1 <= 1 ;\n+1 x3 = 1 ;\n").unwrap();
        assert_eq!(problem.variables, 3);
        assert_eq!(problem.objective, Some(vec![(1, 1), (2, -2)]));
        assert_eq!(problem.constraints.len(), 4);
        assert_eq!(problem.constraints[1], Constraint { terms: vec![(-2, 1)], degree: -1 });
    }

    #[test]
    fn malformed_input_is_rejected() {
        for text in ["+1 x1 >= 1", "+1 x1 x2 >= 1 ;", "x1 >= 1 ;", "+1 >= 1 ;", "+1 x0 >= 1 ;", "+1 x1 > 1 ;", "min: +1 x1 ; min: +1 x2 ;", "+1 x1 >= a ;"] {
            assert!(parse(text).is_err(), "{text}");
        }
    }

    // La CNF con las variables de entrada fijadas tiene solucion exactamente cuando se cumple
    // la restriccion, para coeficientes de todos los signos y tamanos.
    #[test]
    fn the_encoding_is_exact_on_every_assignment() {
        let cases = [
            (vec![(1, 1), (1, 2), (1, 3), (1, 4)], 2),
            (vec![(3, 1), (5, 2), (7, 3), (2, 4)], 8),
            (vec![(-3, 1), (5, -2), (-7, 3), (2, 4)], -2),
            (vec![(1 << 40, 1), (1 << 40, 2), (1, 3), (1, 4)], (1 << 40) + 2),
            (vec![(2, 1), (2, 1), (-2, -1), (3, 2)], 3),
            (vec![(1, 1), (1, 2)], 3),
            (vec![(1, 1), (1, 2)], 0),
        ];
        for (terms, degree) in cases {
            let constraint = Constraint { terms, degree };
            let problem = Problem { variables: 4, objective: None, constraints: vec![constraint.clone()] };
            let (formula, _) = encode(&problem, None).unwrap();
            for bits in 0..16_u32 {
                let model: Vec<bool> = (0..=4).map(|id| id > 0 && bits >> (id - 1) & 1 == 1).collect();
                let mut fixed = formula.clone();
                fixed.extend((1..=4).map(|id| vec![if model[id as usize] { id } else { -id }]));
                let expected = evaluate(&constraint.terms, &model) >= i128::from(constraint.degree);
                assert_eq!(solve_hybrid(&fixed, HybridConfig::default()).model.is_some(), expected, "{constraint:?} {bits:04b}");
            }
        }
    }
}
