// Comprobacion externa de un UNSAT: el motor solo escribe la prueba y la juzga VeriPB.
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

const VERIFIER: &str = "veripb";
// Unica linea que acepta un UNSAT; cualquier otra salida deja el resultado sin certificar.
const ACCEPTED: &str = "s VERIFIED UNSATISFIABLE";

#[derive(Debug, PartialEq)]
pub enum ProofCheck {
    Verified,
    Rejected(String),
    Stopped,
}

// Se pregunta antes de buscar: no tiene sentido gastar horas en un UNSAT que no se podra comprobar.
pub fn ensure_available() -> Result<(), String> {
    Command::new(VERIFIER).arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().map(|_| ()).map_err(|error| {
        format!("No se pudo ejecutar {VERIFIER} ({error}). Instalalo como indica el README, seccion Instalacion, o usa --no-verify para aceptar un UNSAT sin certificado")
    })
}

fn read_all(mut pipe: impl Read + Send + 'static) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        // Una lectura cortada deja la salida incompleta y la linea de aceptacion no aparece.
        let _ = pipe.read_to_end(&mut bytes);
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

// `stop` corta la comprobacion: mata a VeriPB y no da veredicto.
pub fn check_unsat_proof(cnf: &Path, proof: &Path, stop: Option<&AtomicBool>) -> Result<ProofCheck, String> {
    let mut child = Command::new(VERIFIER)
        .arg("--cnf").arg(cnf).arg(proof)
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("No se pudo ejecutar {VERIFIER}: {error}"))?;
    let stdout = read_all(child.stdout.take().expect("stdout con pipe"));
    let stderr = read_all(child.stderr.take().expect("stderr con pipe"));
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            break status;
        }
        if stop.is_some_and(|flag| flag.load(Ordering::Acquire)) {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(ProofCheck::Stopped);
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let (stdout, stderr) = (stdout.join().unwrap_or_default(), stderr.join().unwrap_or_default());
    if status.success() && stdout.lines().chain(stderr.lines()).any(|line| line.trim() == ACCEPTED) {
        return Ok(ProofCheck::Verified);
    }
    let last = stdout.lines().chain(stderr.lines()).filter(|line| !line.trim().is_empty()).last().unwrap_or("sin salida");
    Ok(ProofCheck::Rejected(format!("{status}: {last}")))
}
