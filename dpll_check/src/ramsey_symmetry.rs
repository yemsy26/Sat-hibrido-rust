// Mejora 2, fase 2: ruptura de la simetria de vertices en formulas de Ramsey.
//
// La formula de gen_ramsey sobre K_N no cambia si se permutan los vertices. Aqui se exige
// que cada vertice v no sea mayor que v + 1: la fila de v en la matriz de adyacencia, leida
// por columna creciente y sin las columnas v y v + 1, es lexicograficamente menor o igual
// que la de v + 1.
//
// Por que es correcto. Tomese el orden de las variables tal como estan numeradas (por filas
// del triangulo superior), con falso antes que verdadero. El primer grafo de cada clase de
// isomorfia en ese orden cumple x <= t(x) para toda permutacion t de los vertices; en
// particular para cada trasposicion t = (v v+1). Las posiciones que t mueve van por pares:
// {k, v} con {k, v+1} si k < v, y {v, k} con {v+1, k} si k > v + 1. La segunda de cada par
// solo se mira si la primera empata, y entonces empata tambien, asi que x <= t(x) es justo
// la comparacion de arriba. Se impone entera la condicion de todas las trasposiciones
// adyacentes, con un mismo orden: cada clase conserva al menos un representante y la
// formula sigue siendo satisfacible si lo era. No queda traza de nada de esto.
//
// Comparar solo las columnas k > v + 1 tambien es correcto, pero mucho mas debil: con 7
// vertices deja pasar 83,664 grafos de 2,097,152, frente a 4,045 con la fila entera.
use super::symmetry_breaker::SymmetryBreaker;
use std::collections::HashSet;

// Variable de la arista {i, j}, con 1 <= i < j <= n: las aristas en orden lexicografico
// desde 1, que es como las numera gen_ramsey.
fn edge_var(i: u32, j: u32, n: u32) -> u32 {
    debug_assert!(1 <= i && i < j && j <= n);
    (i - 1) * n - (i - 1) * i / 2 + (j - i)
}

// Variables de arista de K_n; `None` si no caben en un identificador.
pub fn edge_count(n_vertices: u32) -> Option<u32> {
    u32::try_from(u64::from(n_vertices) * u64::from(n_vertices.saturating_sub(1)) / 2).ok()
        .filter(|&edges| edges < i32::MAX as u32)
}

// La formula, con esta numeracion de aristas, no cambia al permutar los vertices. Basta
// comprobarlo con la trasposicion (1 2) y el ciclo (1 2 ... n), que generan todas las
// permutaciones. Es la condicion que hace correcta la inyeccion de abajo.
pub fn is_vertex_symmetric(n_vertices: u32, formula: &[Vec<i32>]) -> bool {
    let Some(edges) = edge_count(n_vertices) else { return false };
    if formula.iter().flatten().any(|literal| literal.unsigned_abs() == 0 || literal.unsigned_abs() > edges) {
        return false;
    }
    if n_vertices < 2 {
        return true;
    }
    // Extremos de cada variable de arista.
    let mut ends = vec![(0, 0); edges as usize + 1];
    for i in 1..n_vertices {
        for j in i + 1..=n_vertices {
            ends[edge_var(i, j, n_vertices) as usize] = (i, j);
        }
    }
    let normal = |clause: &[i32]| -> Vec<i32> {
        let mut literals = clause.to_vec();
        literals.sort_unstable();
        literals.dedup();
        literals
    };
    let clauses: HashSet<Vec<i32>> = formula.iter().map(|clause| normal(clause)).collect();
    let swap = |vertex: u32| match vertex { 1 => 2, 2 => 1, other => other };
    let cycle = |vertex: u32| vertex % n_vertices + 1;
    let invariant = |image: &dyn Fn(u32) -> u32| {
        let mut mapped: Vec<i32> = Vec::new();
        clauses.iter().all(|clause| {
            mapped.clear();
            mapped.extend(clause.iter().map(|&literal| {
                let (i, j) = ends[literal.unsigned_abs() as usize];
                let (a, b) = (image(i), image(j));
                edge_var(a.min(b), a.max(b), n_vertices) as i32 * literal.signum()
            }));
            mapped.sort_unstable();
            // Una permutacion lleva clausulas distintas a clausulas distintas: si todas las
            // imagenes estan en la formula, la formula es la misma.
            clauses.contains(&mapped)
        })
    };
    invariant(&swap) && invariant(&cycle)
}

// Anade a `formula` las clausulas que ordenan cada par de vertices consecutivos: las
// aristas de v hacia todo k distinto de v y de v + 1 no superan, lexicograficamente, a las
// de v + 1 hacia los mismos k. Con N >= 3 son (N-1)(3N-8) clausulas y (N-1)(N-3) variables
// auxiliares, que salen de `breaker`. Solo es correcto sobre una formula que
// `is_vertex_symmetric` acepte.
pub fn inject_ramsey_symmetries(n_vertices: u32, breaker: &mut SymmetryBreaker, formula: &mut Vec<Vec<i32>>) {
    // Arista de `vertex` con `other`, con los extremos en el orden que pide edge_var.
    let edge = |vertex: u32, other: u32| edge_var(vertex.min(other), vertex.max(other), n_vertices);
    for v in 1..n_vertices {
        // Las mismas columnas y en el mismo orden para los dos grupos.
        let columns = (1..=n_vertices).filter(|&k| k != v && k != v + 1);
        let group_a: Vec<u32> = columns.clone().map(|k| edge(v, k)).collect();
        let group_b: Vec<u32> = columns.map(|k| edge(v + 1, k)).collect();
        formula.extend(breaker.lexicographic_order(&group_a, &group_b));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Formula de Ramsey como la de gen_ramsey: sin K_s del color 1 ni K_t del color 0.
    fn ramsey(n: u32, s: usize, t: usize) -> Vec<Vec<i32>> {
        fn subsets(n: u32, size: usize, from: u32, chosen: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
            if chosen.len() == size {
                out.push(chosen.clone());
                return;
            }
            for vertex in from..=n {
                chosen.push(vertex);
                subsets(n, size, vertex + 1, chosen, out);
                chosen.pop();
            }
        }
        let mut formula = Vec::new();
        for (size, sign) in [(s, -1), (t, 1)] {
            let mut all = Vec::new();
            subsets(n, size, 1, &mut Vec::new(), &mut all);
            for clique in all {
                let mut clause = Vec::new();
                for (position, &i) in clique.iter().enumerate() {
                    for &j in &clique[position + 1..] {
                        clause.push(sign * edge_var(i, j, n) as i32);
                    }
                }
                formula.push(clause);
            }
        }
        formula
    }

    #[test]
    fn edges_are_numbered_like_the_generator_does() {
        for n in [2, 3, 7, 43] {
            let mut expected = 0;
            for i in 1..n {
                for j in i + 1..=n {
                    expected += 1;
                    assert_eq!(edge_var(i, j, n), expected);
                }
            }
            assert_eq!(edge_count(n), Some(expected));
        }
        assert_eq!((edge_count(0), edge_count(1), edge_count(43)), (Some(0), Some(0), Some(903)));
        assert_eq!(edge_count(100_000), None);
    }

    #[test]
    fn ramsey_formulas_are_vertex_symmetric_and_others_are_not() {
        assert!(is_vertex_symmetric(6, &ramsey(6, 3, 3)));
        assert!(is_vertex_symmetric(7, &ramsey(7, 3, 4)));
        assert!(is_vertex_symmetric(8, &ramsey(8, 5, 5)));
        // Una clausula sobre una sola arista distingue a sus dos vertices de los demas.
        let mut marked = ramsey(6, 3, 3);
        marked.push(vec![edge_var(2, 5, 6) as i32]);
        assert!(!is_vertex_symmetric(6, &marked));
        // Una formula de otro numero de vertices usa variables que aqui no son aristas.
        assert!(!is_vertex_symmetric(5, &ramsey(6, 3, 3)));
        // Simetrica bajo (1 2) pero no bajo el ciclo.
        assert!(!is_vertex_symmetric(4, &[vec![edge_var(1, 2, 4) as i32]]));
    }

    // Valor de un literal: 1 verdadero, -1 falso, 0 si su variable esta sin fijar.
    fn state(values: &[i8], literal: i32) -> i8 {
        values[literal.unsigned_abs() as usize] * literal.signum() as i8
    }

    // Grafos (un bit por arista) que las clausulas inyectadas dejan pasar: aquellos para los
    // que algun valor de las auxiliares las cumple todas.
    fn survivors(n: u32) -> (Vec<bool>, usize) {
        let edges = edge_count(n).unwrap();
        let mut breaker = SymmetryBreaker::new(edges + 1);
        let mut clauses = Vec::new();
        inject_ramsey_symmetries(n, &mut breaker, &mut clauses);
        let variables = edges as usize + breaker.auxiliaries();
        let kept = (0..1_u64 << edges).map(|graph| {
            let mut values = vec![0_i8; variables + 1];
            for edge in 0..edges as usize {
                values[edge + 1] = if graph >> edge & 1 == 1 { 1 } else { -1 };
            }
            // Con las aristas fijadas, propagacion unitaria sobre las auxiliares. Son
            // demasiadas para probar todos sus valores.
            loop {
                let mut changed = false;
                for clause in &clauses {
                    if clause.iter().any(|&literal| state(&values, literal) > 0) {
                        continue;
                    }
                    let mut open = clause.iter().copied().filter(|&literal| state(&values, literal) == 0);
                    match (open.next(), open.next()) {
                        (None, _) => return false,
                        (Some(unit), None) => {
                            values[unit.unsigned_abs() as usize] = unit.signum() as i8;
                            changed = true;
                        }
                        _ => {}
                    }
                }
                if !changed {
                    break;
                }
            }
            // Sin conflicto: las auxiliares que quedan libres, a falso, cumplen el resto.
            assert!(clauses.iter().all(|clause| clause.iter().any(|&literal| {
                state(&values, literal) > 0 || (state(&values, literal) == 0 && literal < 0)
            })), "Unit propagation did not decide the graph {graph:b} on {n} vertices");
            true
        }).collect();
        (kept, clauses.len())
    }

    // Todas las permutaciones de 1..=n.
    fn permutations(n: u32) -> Vec<Vec<u32>> {
        let mut all = vec![vec![]];
        for _ in 0..n {
            all = all.into_iter().flat_map(|prefix: Vec<u32>| {
                (1..=n).filter(|vertex| !prefix.contains(vertex)).map(|vertex| {
                    let mut longer = prefix.clone();
                    longer.push(vertex);
                    longer
                }).collect::<Vec<_>>()
            }).collect();
        }
        all
    }

    #[test]
    fn every_graph_keeps_an_isomorphic_copy() {
        // Exhaustivo: para todo grafo de hasta 6 vertices, alguna reordenacion de sus
        // vertices cumple las clausulas inyectadas.
        for n in 2..=6_u32 {
            let edges = edge_count(n).unwrap();
            let (kept, clauses) = survivors(n);
            let relabel: Vec<Vec<u32>> = permutations(n).into_iter().map(|image| {
                // Para cada variable de arista, la variable a la que va.
                let mut table = vec![0; edges as usize];
                for i in 1..n {
                    for j in i + 1..=n {
                        let (a, b) = (image[i as usize - 1], image[j as usize - 1]);
                        table[edge_var(i, j, n) as usize - 1] = edge_var(a.min(b), a.max(b), n) - 1;
                    }
                }
                table
            }).collect();
            for graph in 0..1_u64 << edges {
                let found = relabel.iter().any(|table| {
                    let moved = table.iter().enumerate()
                        .fold(0_u64, |bits, (edge, &target)| bits | (graph >> edge & 1) << target);
                    kept[moved as usize]
                });
                assert!(found, "The graph {graph:b} on {n} vertices lost every copy");
            }
            let passing = kept.iter().filter(|&&keep| keep).count();
            // Rompe de verdad: de 2, 8, 64, 1024 y 32768 grafos quedan estos (hay 2, 4, 11,
            // 34 y 156 clases de isomorfia).
            assert_eq!(passing, [2, 4, 11, 46, 325][n as usize - 2], "graphs kept on {n} vertices");
            assert_eq!(clauses, if n < 3 { 0 } else { ((n - 1) * (3 * n - 8)) as usize });
        }
    }
}
