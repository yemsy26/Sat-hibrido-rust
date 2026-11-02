// Entrada .opb por el comando real, comparada con la fuerza bruta.
use std::path::PathBuf;
use std::process::{Command, Output};

const EXE: &str = env!("CARGO_BIN_EXE_motor");

struct Scratch(PathBuf);
impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("motor_opb_{name}_{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn file(&self, name: &str, text: &str) -> String {
        let path = self.0.join(name);
        std::fs::write(&path, text).unwrap();
        path.to_string_lossy().into_owned()
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn motor(arguments: &[&str]) -> Output {
    Command::new(EXE).args(arguments).output().unwrap()
}

fn generator(seed: u64) -> impl FnMut(u64) -> u64 {
    let mut state = seed;
    move |bound| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    }
}

type Terms = Vec<(i64, i32)>;

fn random_terms(next: &mut impl FnMut(u64) -> u64, variables: u64) -> Terms {
    (0..1 + next(4)).map(|_| {
        let literal = (1 + next(variables) as i32) * if next(2) == 0 { 1 } else { -1 };
        (next(13) as i64 - 6, literal)
    }).collect()
}

fn text_of(terms: &Terms) -> String {
    terms.iter().map(|&(a, l)| format!("{a:+} {}x{} ", if l < 0 { "~" } else { "" }, l.abs())).collect()
}

fn sum(terms: &Terms, bits: u32) -> i64 {
    terms.iter().filter(|&&(_, l)| (bits >> (l.unsigned_abs() - 1) & 1 == 1) == (l > 0)).map(|&(a, _)| a).sum()
}

fn model_bits(stdout: &str) -> u32 {
    let line = stdout.lines().find(|line| line.starts_with("v ")).expect("sin linea v");
    line[2..].split_whitespace().fold(0, |bits, token| match token.strip_prefix('x') {
        Some(id) => bits | 1 << (id.parse::<u32>().unwrap() - 1),
        None => bits,
    })
}

// Restricciones de los tres tipos y, en la mitad de las rondas, objetivo.
fn fuzz(rounds: u32, seed: u64, extra: &[&str]) {
    let scratch = Scratch::new(&format!("fuzz{seed:x}"));
    let mut next = generator(seed);
    let (mut sat, mut unsat, mut optimal) = (0, 0, 0);
    for round in 0..rounds {
        let variables = 2 + next(6);
        let mut text = String::new();
        let objective = (next(2) == 0).then(|| random_terms(&mut next, variables));
        if let Some(objective) = &objective {
            text += &format!("min: {};\n", text_of(objective));
        }
        let mut constraints: Vec<(Terms, u64, i64)> = Vec::new();
        for _ in 0..1 + next(5) {
            let (terms, relation, degree) = (random_terms(&mut next, variables), next(3), next(13) as i64 - 6);
            text += &format!("{}{} {degree} ;\n", text_of(&terms), [">=", "<=", "="][relation as usize]);
            constraints.push((terms, relation, degree));
        }
        // Para que todas las variables existan en el fichero.
        for id in 1..=variables {
            text += &format!("+1 x{id} +1 ~x{id} >= 1 ;\n");
        }
        let holds = |bits: u32| constraints.iter().all(|(terms, relation, degree)| match relation {
            0 => sum(terms, bits) >= *degree,
            1 => sum(terms, bits) <= *degree,
            _ => sum(terms, bits) == *degree,
        });
        let feasible: Vec<u32> = (0..1_u32 << variables).filter(|&bits| holds(bits)).collect();
        let input = scratch.file("f.opb", &text);
        let mut arguments = vec!["solve", input.as_str()];
        arguments.extend_from_slice(extra);
        let output = motor(&arguments);
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let context = format!("ronda {round}:\n{text}\n{stdout}\n{}", String::from_utf8_lossy(&output.stderr));
        match (feasible.is_empty(), &objective) {
            (true, _) => {
                assert_eq!(output.status.code(), Some(20), "{context}");
                unsat += 1;
            }
            (false, None) => {
                assert_eq!(output.status.code(), Some(10), "{context}");
                assert!(holds(model_bits(&stdout)), "{context}");
                sat += 1;
            }
            (false, Some(objective)) => {
                assert_eq!(output.status.code(), Some(30), "{context}");
                let bits = model_bits(&stdout);
                assert!(holds(bits), "{context}");
                let best = feasible.iter().map(|&bits| sum(objective, bits)).min().unwrap();
                assert_eq!(sum(objective, bits), best, "{context}");
                assert!(stdout.contains(&format!("o {best}\n")) || stdout.contains(&format!("o {best}\r\n")), "{context}");
                optimal += 1;
            }
        }
    }
    assert!(sat > 5 && unsat > 5 && optimal > 5, "banco sin cobertura: {sat} SAT, {unsat} UNSAT, {optimal} optimos");
}

#[test]
fn feasibility_and_optimum_agree_with_brute_force() {
    fuzz(150, 0x9e37_79b9_7f4a_7c15, &["--no-verify"]);
}

#[test]
#[ignore = "requiere veripb"]
fn every_refutation_through_the_verifier_agrees_with_brute_force() {
    fuzz(150, 0x2545_f491_4f6c_dd1d, &[]);
}

#[test]
fn bad_input_gives_an_error_and_no_verdict() {
    let scratch = Scratch::new("bad");
    let input = scratch.file("f.opb", "+1 x1 x2 >= 1 ;\n");
    let output = motor(&["solve", &input, "--no-verify"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.contains("s SATISFIABLE") && !stdout.contains("s UNSATISFIABLE"));
    assert_ne!(output.status.code(), Some(10));
    assert_ne!(output.status.code(), Some(20));
}

// Ocho reinas sin atacarse: un problema de conteo escrito en el formato, con solucion conocida.
#[test]
fn eight_queens_is_solved_and_its_model_checked() {
    const N: i32 = 8;
    let var = |row: i32, column: i32| row * N + column + 1;
    let mut text = String::new();
    for row in 0..N {
        let line: String = (0..N).map(|column| format!("+1 x{} ", var(row, column))).collect();
        text += &format!("{line}= 1 ;\n");
        let line: String = (0..N).map(|other| format!("+1 x{} ", var(other, row))).collect();
        text += &format!("{line}= 1 ;\n");
    }
    for a in 0..N * N {
        for b in a + 1..N * N {
            if (a / N - b / N).abs() == (a % N - b % N).abs() {
                text += &format!("+1 x{} +1 x{} <= 1 ;\n", a + 1, b + 1);
            }
        }
    }
    let scratch = Scratch::new("queens");
    let input = scratch.file("q.opb", &text);
    let output = motor(&["solve", &input, "--no-verify", "--timeout", "120"]);
    assert_eq!(output.status.code(), Some(10), "{}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let queens = stdout.lines().find(|line| line.starts_with("v ")).unwrap().split_whitespace().filter(|token| token.starts_with('x')).count();
    assert_eq!(queens, N as usize);
}
