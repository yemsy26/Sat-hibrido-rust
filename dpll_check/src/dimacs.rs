// Lectura de formulas DIMACS CNF.
pub type Formula = Vec<Vec<i32>>;

// El numero de clausulas de la traza es el de clausulas leidas, no el de la cabecera.
pub fn parse_dimacs(text: &str) -> Result<Formula, String> {
    let mut formula = Formula::new();
    let mut clause = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('%') {
            break;
        }
        if line.starts_with('c') || line.starts_with('p') {
            continue;
        }
        for token in line.split_whitespace() {
            let literal: i32 = token.parse().map_err(|_| format!("Literal no valido: {token}"))?;
            if literal == i32::MIN {
                return Err(format!("Literal fuera de rango: {token}"));
            }
            if literal == 0 {
                formula.push(std::mem::take(&mut clause));
            } else {
                clause.push(literal);
            }
        }
    }
    if clause.is_empty() { Ok(formula) } else { Err("La ultima clausula no termina en 0".into()) }
}

// La cabecera lleva el numero real de clausulas: es el que espera la traza.
pub fn write_dimacs(path: &std::path::Path, variables: u32, formula: &Formula) -> std::io::Result<()> {
    use std::io::Write;
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(out, "p cnf {variables} {}", formula.len())?;
    for clause in formula {
        for literal in clause {
            write!(out, "{literal} ")?;
        }
        writeln!(out, "0")?;
    }
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimacs_is_read_clause_by_clause() {
        let formula = parse_dimacs("c x\np cnf 3 3\n1 -2 0 3\n0\n\n-1 0\n%\n9 0\n").unwrap();
        assert_eq!(formula, vec![vec![1, -2], vec![3], vec![-1]]);
        assert!(parse_dimacs("1 2").is_err());
        assert!(parse_dimacs("1 x 0").is_err());
    }
}
