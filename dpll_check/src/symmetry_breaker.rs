// Mejora 2: ruptura de simetrias por orden lexicografico. Dados dos grupos de variables que
// una simetria de la formula intercambia posicion a posicion, `lexicographic_order` devuelve
// las clausulas que imponen A <= B, con la primera posicion como la mas significativa y
// falso < verdadero.
//
// Quien llama responde de que la simetria existe: intercambiar a_i con b_i en todas las
// posiciones, junto con lo demas que esa simetria mueva, debe dejar la formula igual. Con
// eso, de cada par de soluciones simetricas sobrevive al menos una. Para romper varias
// simetrias a la vez, todas las llamadas deben comparar las variables en un mismo orden
// global. Este modulo no comprueba nada de eso ni deja traza: es una herramienta aislada,
// sin conexion con el motor.
//
// El motor ya rompe por su cuenta las simetrias que descubre (hybrid_symmetry.rs), con
// cadenas sin variables auxiliares que son una version debilitada de este orden. Aqui el
// orden es exacto y los grupos los da quien conoce el problema.

pub struct SymmetryBreaker {
    // Siguiente identificador de variable sin usar: de aqui salen las auxiliares.
    next: u32,
    auxiliaries: usize,
}

impl SymmetryBreaker {
    // `first_free` es el primer identificador de variable que la formula no usa.
    pub fn new(first_free: u32) -> Self {
        assert!(first_free > 0, "Los identificadores de variable empiezan en 1");
        Self { next: first_free, auxiliaries: 0 }
    }
    // Primer identificador libre tras las auxiliares creadas hasta ahora.
    pub fn next_free(&self) -> u32 {
        self.next
    }
    pub fn auxiliaries(&self) -> usize {
        self.auxiliaries
    }
    fn fresh(&mut self) -> i32 {
        let variable = i32::try_from(self.next).expect("Identificador de variable fuera de rango");
        self.next += 1;
        self.auxiliaries += 1;
        variable
    }
    // Clausulas que imponen `group_a <= group_b` en orden lexicografico. Con m posiciones
    // distintas devuelve 3m - 2 clausulas y crea m - 1 variables auxiliares.
    //
    // La auxiliar t_k dice "las k primeras posiciones empatan" (t_0 es cierto). Para cada k:
    //   t_(k-1) y a_k  implican  b_k       con lo anterior empatado, aqui no hay 1 frente a 0
    //   t_(k-1) y a_k  implican  t_k       a_k = 1 obliga a b_k = 1: el empate sigue
    //   t_(k-1) y no b_k  implican  t_k    b_k = 0 obliga a a_k = 0: el empate sigue
    // Las auxiliares solo quedan forzadas mientras hay empate; en cuanto A es menor en una
    // posicion pueden ser falsas y el resto de las clausulas se cumple solo.
    pub fn lexicographic_order(&mut self, group_a: &[u32], group_b: &[u32]) -> Vec<Vec<i32>> {
        assert_eq!(group_a.len(), group_b.len(), "Los dos grupos se comparan posicion a posicion");
        let literal = |variable: u32| -> i32 {
            assert!(variable > 0 && variable < self.next, "La variable {variable} no es de la formula");
            i32::try_from(variable).expect("Identificador de variable fuera de rango")
        };
        // Una posicion con la misma variable a los dos lados empata siempre: no aporta nada.
        let pairs: Vec<(i32, i32)> = group_a.iter().zip(group_b)
            .filter(|(a, b)| a != b)
            .map(|(&a, &b)| (literal(a), literal(b)))
            .collect();
        let mut clauses = Vec::with_capacity(3 * pairs.len());
        // Literal que niega "todo lo anterior empata"; al principio no hay nada que negar.
        let mut untied: Option<i32> = None;
        for (position, &(a, b)) in pairs.iter().enumerate() {
            let guarded = |literals: [i32; 2]| -> Vec<i32> { untied.into_iter().chain(literals).collect() };
            clauses.push(guarded([-a, b]));
            if position + 1 < pairs.len() {
                let tied = self.fresh();
                clauses.push(guarded([-a, tied]));
                clauses.push(guarded([b, tied]));
                untied = Some(-tied);
            }
        }
        clauses
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Valor de la variable `id` en una asignacion dada como bits: bit id - 1.
    fn value(bits: u32, literal: i32) -> bool {
        (bits >> (literal.unsigned_abs() - 1) & 1 == 1) == (literal > 0)
    }

    // Existe un valor de las auxiliares (ids por encima de `originals`) que cumple todo.
    fn extends(clauses: &[Vec<i32>], originals: u32, auxiliaries: usize, bits: u32) -> bool {
        (0..1_u32 << auxiliaries).any(|extra| {
            let all = bits | extra << originals;
            clauses.iter().all(|clause| clause.iter().any(|&literal| value(all, literal)))
        })
    }

    // Comparacion directa: la primera posicion es la mas significativa y falso < verdadero.
    fn at_most(group_a: &[u32], group_b: &[u32], bits: u32) -> bool {
        let read = |group: &[u32]| -> Vec<bool> { group.iter().map(|&variable| value(bits, variable as i32)).collect() };
        read(group_a) <= read(group_b)
    }

    #[test]
    fn the_clauses_hold_exactly_when_the_first_group_is_not_greater() {
        for width in 1..=5_u32 {
            let group_a: Vec<u32> = (1..=width).collect();
            let group_b: Vec<u32> = (width + 1..=2 * width).collect();
            let mut breaker = SymmetryBreaker::new(2 * width + 1);
            let clauses = breaker.lexicographic_order(&group_a, &group_b);
            assert_eq!(clauses.len() as u32, 3 * width - 2);
            assert_eq!((breaker.auxiliaries() as u32, breaker.next_free()), (width - 1, 3 * width));
            assert!(clauses.iter().all(|clause| clause.len() <= 3));
            for bits in 0..1_u32 << (2 * width) {
                assert_eq!(
                    extends(&clauses, 2 * width, breaker.auxiliaries(), bits),
                    at_most(&group_a, &group_b, bits),
                    "width {width}, assignment {bits:b}"
                );
            }
        }
    }

    #[test]
    fn the_order_propagates_by_unit_resolution_along_a_tied_prefix() {
        // A = (x1..x4), B = (x5..x8). Con las tres primeras posiciones empatadas y a4 = 1,
        // la propagacion unitaria sola debe llegar a b4 = 1.
        let mut breaker = SymmetryBreaker::new(9);
        let clauses = breaker.lexicographic_order(&[1, 2, 3, 4], &[5, 6, 7, 8]);
        let mut fixed: Vec<i32> = vec![1, 5, -2, -6, 3, 7, 4];
        loop {
            let unit = clauses.iter().find_map(|clause| {
                if clause.iter().any(|literal| fixed.contains(literal)) {
                    return None;
                }
                let open: Vec<i32> = clause.iter().copied().filter(|literal| !fixed.contains(&-literal)).collect();
                (open.len() == 1).then(|| open[0])
            });
            match unit {
                Some(literal) => fixed.push(literal),
                None => break,
            }
        }
        assert!(fixed.contains(&8), "b4 is not forced: {fixed:?}");
        // Las tres auxiliares de empate han quedado verdaderas por el camino.
        assert!([9, 10, 11].iter().all(|tied| fixed.contains(tied)));
    }

    #[test]
    fn shared_positions_are_skipped_and_calls_do_not_share_auxiliaries() {
        // La primera posicion es la misma variable a los dos lados: solo cuentan las otras.
        let (group_a, group_b) = ([1, 2, 3], [1, 3, 2]);
        let mut breaker = SymmetryBreaker::new(4);
        let clauses = breaker.lexicographic_order(&group_a, &group_b);
        assert_eq!((clauses.len(), breaker.auxiliaries()), (4, 1));
        for bits in 0..1_u32 << 3 {
            assert_eq!(extends(&clauses, 3, 1, bits), at_most(&group_a, &group_b, bits));
        }
        // Una segunda llamada sigue numerando desde donde quedo la primera.
        let more = breaker.lexicographic_order(&[1, 2], &[2, 3]);
        assert_eq!((breaker.auxiliaries(), breaker.next_free()), (2, 6));
        assert!(more.iter().flatten().all(|literal| literal.unsigned_abs() != 4));
        assert!(more.iter().flatten().any(|literal| literal.unsigned_abs() == 5));
        // Grupos iguales o vacios no imponen nada.
        assert!(breaker.lexicographic_order(&[1, 2], &[1, 2]).is_empty());
        assert!(breaker.lexicographic_order(&[], &[]).is_empty());
        assert_eq!(breaker.auxiliaries(), 2);
    }

    #[test]
    #[should_panic(expected = "posicion a posicion")]
    fn groups_of_different_length_are_rejected() {
        SymmetryBreaker::new(10).lexicographic_order(&[1, 2], &[3]);
    }

    #[test]
    #[should_panic(expected = "no es de la formula")]
    fn a_variable_beyond_the_formula_is_rejected() {
        SymmetryBreaker::new(4).lexicographic_order(&[1, 2], &[3, 4]);
    }
}
