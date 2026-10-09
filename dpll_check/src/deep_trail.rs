#[derive(Debug)]
pub struct DeepTrailSnapshot {
    pub depth: usize,
    pub total: usize,
    pub literals: Vec<i32>,
}

#[derive(Debug)]
pub enum DeepTrailMessage {
    Snapshot(DeepTrailSnapshot),
    Finish,
}

// El guard vive en hunt hasta despues del join de todos los workers.
// Finish evita que Drop espere a que se destruyan los Sender guardados en progress.
#[derive(Debug)]
pub struct DeepTrailWriter {
    sender: std::sync::mpsc::Sender<DeepTrailMessage>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl DeepTrailWriter {
    pub fn start(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if path.is_dir() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput,
                "deep-trail necesita un archivo, no un directorio"));
        }
        let file_name = path.file_name().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "deep-trail necesita un archivo")
        })?;
        let mut temporary_name = file_name.to_os_string();
        temporary_name.push(format!(".{}.tmp", std::process::id()));
        let parent = path.parent().filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| std::path::Path::new("."));
        let temporary = parent.join(temporary_name);
        // Comprobar permisos antes de iniciar la busqueda; no tocar un snapshot anterior.
        drop(std::fs::File::create(&temporary)?);
        let (sender, receiver) = std::sync::mpsc::channel();
        let thread = std::thread::Builder::new().name("deep-trail-writer".into())
            .spawn(move || {
                let mut newest_seen = 0;
                while let Ok(message) = receiver.recv() {
                    let mut snapshot = match message {
                        DeepTrailMessage::Snapshot(snapshot) => snapshot,
                        DeepTrailMessage::Finish => break,
                    };
                    let mut finish = false;
                    // Unificar rafagas: escribir solo el mayor record ya disponible.
                    while let Ok(message) = receiver.try_recv() {
                        match message {
                            DeepTrailMessage::Snapshot(candidate) => {
                                if candidate.depth > snapshot.depth {
                                    snapshot = candidate;
                                }
                            }
                            DeepTrailMessage::Finish => {
                                finish = true;
                                break;
                            }
                        }
                    }
                    if snapshot.depth > newest_seen {
                        newest_seen = snapshot.depth;
                        if let Err(error) = write_deep_trail(&path, &temporary, &snapshot) {
                            eprintln!("c deep-trail: no se pudo publicar profundidad {}: {error}",
                                snapshot.depth);
                        }
                    }
                    if finish {
                        break;
                    }
                }
                // Solo se borra el temporal privado; el snapshot publicado se conserva.
                let _ = std::fs::remove_file(&temporary);
            })?;
        Ok(Self { sender, thread: Some(thread) })
    }

    pub fn sender(&self) -> std::sync::mpsc::Sender<DeepTrailMessage> {
        self.sender.clone()
    }

    fn finish(&mut self) {
        if let Some(thread) = self.thread.take() {
            let _ = self.sender.send(DeepTrailMessage::Finish);
            if thread.join().is_err() {
                eprintln!("c deep-trail: el escritor termino con un panic");
            }
        }
    }
}

impl Drop for DeepTrailWriter {
    fn drop(&mut self) {
        self.finish();
    }
}

fn write_deep_trail(
    path: &std::path::Path,
    temporary: &std::path::Path,
    snapshot: &DeepTrailSnapshot,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut output = std::io::BufWriter::new(std::fs::File::create(temporary)?);
    writeln!(output, "c partial assignment; not a SAT certificate")?;
    writeln!(output, "c depth {} total {}", snapshot.depth, snapshot.total)?;
    write!(output, "v")?;
    for &literal in &snapshot.literals {
        write!(output, " {literal}")?;
    }
    writeln!(output, " 0")?;
    output.flush()?;
    drop(output);
    // Mismo directorio: el lector solo ve el snapshot antiguo o el nuevo completo.
    // Un lector de Windows puede mantener abierto el destino brevemente.
    // Los reintentos se ejecutan solo en el escritor, nunca en un worker.
    for attempt in 0..20 {
        match std::fs::rename(temporary, path) {
            Ok(()) => return Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied
                && attempt < 19 => std::thread::sleep(std::time::Duration::from_millis(10)),
            Err(error) => return Err(error),
        }
    }
    unreachable!("the last rename attempt returns")
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    fn path() -> std::path::PathBuf {
        let directory = std::path::Path::new("results").join(format!("writer-test-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&directory).unwrap();
        directory.join("snapshot.sat")
    }
    fn snapshot(depth: usize) -> DeepTrailMessage {
        DeepTrailMessage::Snapshot(DeepTrailSnapshot { depth, total: 2583, literals: (1..=depth as i32).collect() })
    }
    #[test]
    fn final_record_is_monotonic_and_guard_finishes_with_live_sender() {
        let path = path();
        let writer = DeepTrailWriter::start(&path).unwrap();
        let sender = writer.sender();
        sender.send(snapshot(100)).unwrap();
        sender.send(snapshot(101)).unwrap();
        sender.send(snapshot(99)).unwrap();
        drop(writer);
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("c depth 101 total 2583"));
        assert!(contents.ends_with(" 101 0\n"));
        assert!(sender.send(snapshot(102)).is_err());
        if path.exists() { std::fs::remove_file(&path).unwrap(); }
        std::fs::remove_dir(path.parent().unwrap()).unwrap();
    }
    #[test]
    fn replaces_existing_snapshot_in_windows() {
        let path = path();
        std::fs::write(&path, "old").unwrap();
        let writer = DeepTrailWriter::start(&path).unwrap();
        writer.sender().send(snapshot(101)).unwrap();
        drop(writer);
        assert!(std::fs::read_to_string(&path).unwrap().contains("c depth 101"));
        if path.exists() { std::fs::remove_file(&path).unwrap(); }
        std::fs::remove_dir(path.parent().unwrap()).unwrap();
    }
    #[test]
    fn invalid_parent_fails_before_thread_launch() {
        let path = path();
        assert!(DeepTrailWriter::start(path.parent().unwrap().join("absent").join("snapshot.sat")).is_err());
        if path.exists() { std::fs::remove_file(&path).unwrap(); }
        std::fs::remove_dir(path.parent().unwrap()).unwrap();
    }
}
