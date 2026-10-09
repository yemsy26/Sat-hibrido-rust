//! Exportación opt-in de cubos de cola pesada. No contiene código SAT ni unsafe.
//! El worker usa try_send; toda la escritura y los reintentos viven en un hilo.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::Arc;

#[derive(Debug)]
pub struct HardCubeSnapshot {
    /// Recubings acumulados en la genealogía de este cubo, no del enjambre.
    pub recubings: u32,
    pub cube: Vec<i32>,
    /// Trail DIMACS antes del backjump que sigue al conflicto de presupuesto.
    /// Puede contener un conflicto: representa el intento, no un certificado SAT.
    pub literals: Vec<i32>,
    pub total: usize,
    pub depth: usize,
}

#[derive(Debug)]
enum Message {
    Snapshot(HardCubeSnapshot),
    Finish,
}

#[derive(Clone, Debug)]
pub struct HardCubeSender {
    sender: SyncSender<Message>,
    closed: Arc<AtomicBool>,
}

impl HardCubeSender {
    /// Best-effort: si el buffer está lleno el llamador descarta el snapshot.
    /// No espera al disco, al escritor ni a un slot libre del canal.
    pub fn try_send(
        &self,
        snapshot: HardCubeSnapshot,
    ) -> Result<(), TrySendError<HardCubeSnapshot>> {
        if self.closed.load(Ordering::Acquire) {
            return Err(TrySendError::Disconnected(snapshot));
        }
        match self.sender.try_send(Message::Snapshot(snapshot)) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(Message::Snapshot(snapshot))) => {
                Err(TrySendError::Full(snapshot))
            }
            Err(TrySendError::Disconnected(Message::Snapshot(snapshot))) => {
                Err(TrySendError::Disconnected(snapshot))
            }
            Err(_) => unreachable!("try_send solo produce mensajes Snapshot"),
        }
    }
}

/// Mantenga este guard hasta después del join de los workers SAT.
/// Finish cierra el escritor aunque progress conserve clones del sender.
#[derive(Debug)]
pub struct HardCubeWriter {
    sender: HardCubeSender,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl HardCubeWriter {
    pub fn start(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if path.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "hard-cube necesita un archivo, no un directorio",
            ));
        }
        let file_name = path.file_name().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "hard-cube necesita un archivo")
        })?;
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        // Temporal propio en el mismo directorio. Comprobar acceso antes de
        // buscar, sin truncar un fragmento ya publicado ni reutilizar temporales.
        let temporary = reserve_temporary(parent, file_name)?;
        let cleanup = temporary.clone();
        let (sender, receiver) = mpsc::sync_channel(1);
        let closed = Arc::new(AtomicBool::new(false));
        let thread = std::thread::Builder::new()
            .name("hard-cube-writer".into())
            .spawn(move || run_writer(&path, &temporary, receiver));
        let thread = match thread {
            Ok(thread) => thread,
            Err(error) => {
                let _ = std::fs::remove_file(cleanup);
                return Err(error);
            }
        };
        Ok(Self {
            sender: HardCubeSender { sender, closed },
            thread: Some(thread),
        })
    }

    pub fn sender(&self) -> HardCubeSender {
        self.sender.clone()
    }

    fn finish(&mut self) {
        if let Some(thread) = self.thread.take() {
            self.sender.closed.store(true, Ordering::Release);
            // Solo el guard puede esperar; la capacidad uno acota memoria.
            // Al cerrar productores, el último slot pendiente se drena antes
            // de Finish. No depende de destruir los clones de progress.
            let _ = self.sender.sender.send(Message::Finish);
            if thread.join().is_err() {
                eprintln!("c hard-cube: el escritor terminó con un panic");
            }
        }
    }
}

impl Drop for HardCubeWriter {
    fn drop(&mut self) {
        self.finish();
    }
}

fn reserve_temporary(parent: &Path, file_name: &std::ffi::OsStr) -> io::Result<PathBuf> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    for _ in 0..32 {
        let mut name = file_name.to_os_string();
        name.push(format!(
            ".{}.{}.hard.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let path = parent.join(name);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => {
                drop(file);
                return Ok(path);
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "demasiados temporales hard-cube antiguos",
    ))
}

fn prefer(current: &mut Option<HardCubeSnapshot>, candidate: HardCubeSnapshot) {
    if candidate.recubings <= 3 {
        return;
    }
    if current
        .as_ref()
        .is_none_or(|snapshot| candidate.recubings >= snapshot.recubings)
    {
        // Mismo lineage: elegir el último mensaje, no la profundidad mayor.
        *current = Some(candidate);
    }
}

fn run_writer(path: &Path, temporary: &Path, receiver: Receiver<Message>) {
    let mut published_lineage = 0;
    while let Ok(message) = receiver.recv() {
        let mut snapshot = None;
        match message {
            Message::Snapshot(candidate) => prefer(&mut snapshot, candidate),
            Message::Finish => break,
        }
        let mut finish = false;
        // Coalescing de lo que ya está disponible; no espera nuevos mensajes.
        while let Ok(message) = receiver.try_recv() {
            match message {
                Message::Snapshot(candidate) => prefer(&mut snapshot, candidate),
                Message::Finish => {
                    finish = true;
                    break;
                }
            }
        }
        if let Some(snapshot) = snapshot.filter(|snapshot| snapshot.recubings >= published_lineage)
        {
            match write_snapshot(path, temporary, &snapshot) {
                Ok(()) => published_lineage = snapshot.recubings,
                Err(error) => eprintln!(
                    "c hard-cube: no se pudo publicar lineage {} profundidad {}: {error}",
                    snapshot.recubings, snapshot.depth
                ),
            }
        }
        if finish {
            break;
        }
    }
    // Nunca quitar el destino: debe sobrevivir al cierre del solver.
    let _ = std::fs::remove_file(temporary);
}

fn write_snapshot(path: &Path, temporary: &Path, snapshot: &HardCubeSnapshot) -> io::Result<()> {
    if snapshot.depth != snapshot.literals.len()
        || snapshot.depth > snapshot.total
        || snapshot.literals.contains(&0)
        || snapshot.cube.contains(&0)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "snapshot parcial inválido",
        ));
    }
    let mut output = io::BufWriter::new(std::fs::File::create(temporary)?);
    writeln!(output, "c partial assignment; not a SAT certificate")?;
    writeln!(
        output,
        "c captured at budget conflict before backjump; may contain a conflict"
    )?;
    writeln!(
        output,
        "c depth {} total {}",
        snapshot.depth, snapshot.total
    )?;
    writeln!(output, "c recubings {}", snapshot.recubings)?;
    write!(output, "c cube")?;
    for &literal in &snapshot.cube {
        write!(output, " {literal}")?;
    }
    writeln!(output, " 0")?;
    write!(output, "v")?;
    for &literal in &snapshot.literals {
        write!(output, " {literal}")?;
    }
    writeln!(output, " 0")?;
    output.flush()?;
    drop(output);
    // Rename en el mismo directorio reemplaza el archivo completo. Un lector
    // Windows puede bloquear temporalmente el destino; nunca se borra primero.
    for attempt in 0..20 {
        match std::fs::rename(temporary, path) {
            Ok(()) => return Ok(()),
            Err(error) if error.kind() == io::ErrorKind::PermissionDenied && attempt < 19 => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(error) => return Err(error),
        }
    }
    unreachable!("el último intento devuelve su resultado")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn path() -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let directory = Path::new("results").join(format!(
            "hard-cube-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        directory.join("fragmento.sat")
    }
    fn snapshot(recubings: u32, marker: i32) -> HardCubeSnapshot {
        HardCubeSnapshot {
            recubings,
            cube: vec![1, -2],
            literals: vec![1, -2, marker],
            total: 20,
            depth: 3,
        }
    }
    fn clean(path: &Path) {
        if path.exists() {
            std::fs::remove_file(path).unwrap();
        }
        std::fs::remove_dir(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn guard_drains_pending_snapshot_with_live_sender_and_replaces_atomically() {
        let path = path();
        std::fs::write(&path, "old snapshot\n").unwrap();
        let writer = HardCubeWriter::start(&path).unwrap();
        let sender = writer.sender();
        sender.try_send(snapshot(4, 5)).unwrap();
        drop(writer);
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text, "c partial assignment; not a SAT certificate\nc captured at budget conflict before backjump; may contain a conflict\nc depth 3 total 20\nc recubings 4\nc cube 1 -2 0\nv 1 -2 5 0\n");
        assert!(matches!(
            sender.try_send(snapshot(5, 6)),
            Err(TrySendError::Disconnected(_))
        ));
        assert_eq!(
            std::fs::read_dir(path.parent().unwrap()).unwrap().count(),
            1
        );
        clean(&path);
    }

    #[test]
    fn coalescing_keeps_highest_lineage_and_latest_tie() {
        let path = path();
        let temporary = path.with_extension("tmp");
        let (sender, receiver) = mpsc::channel();
        for (lineage, marker) in [(3, 3), (4, 4), (7, 7), (7, 8), (6, 6)] {
            sender
                .send(Message::Snapshot(snapshot(lineage, marker)))
                .unwrap();
        }
        sender.send(Message::Finish).unwrap();
        run_writer(&path, &temporary, receiver);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("c recubings 7\n"));
        assert!(text.ends_with("v 1 -2 8 0\n"));
        assert!(!temporary.exists());
        clean(&path);
    }

    #[test]
    fn later_lower_lineage_does_not_overwrite_an_already_published_fragment() {
        let path = path();
        let writer = HardCubeWriter::start(&path).unwrap();
        let sender = writer.sender();
        sender.try_send(snapshot(9, 9)).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if std::fs::read_to_string(&path).is_ok_and(|text| text.contains("c recubings 9\n")) {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "writer did not publish first snapshot"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        sender.try_send(snapshot(8, 8)).unwrap();
        drop(writer);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("c recubings 9\n"));
        assert!(text.ends_with("v 1 -2 9 0\n"));
        clean(&path);
    }

    #[test]
    fn concurrent_reader_only_sees_complete_dimacs_replacements() {
        let path = path();
        let temporary = path.with_extension("tmp");
        let make_snapshot = |depth: usize| HardCubeSnapshot {
            recubings: 4,
            cube: vec![],
            total: 16_384,
            depth,
            literals: (1..=depth as i32).collect(),
        };
        write_snapshot(&path, &temporary, &make_snapshot(1)).unwrap();
        let running = Arc::new(AtomicBool::new(true));
        let reader_flag = Arc::clone(&running);
        let reader_path = path.clone();
        let (ready_tx, ready_rx) = mpsc::channel();
        let reader = std::thread::spawn(move || {
            let mut reads = 0;
            loop {
                let text = std::fs::read_to_string(&reader_path).unwrap();
                assert!(text.starts_with("c partial assignment; not a SAT certificate\n"));
                assert!(text.ends_with(" 0\n"));
                let depth = text
                    .lines()
                    .find(|line| line.starts_with("c depth "))
                    .unwrap()
                    .split_whitespace()
                    .nth(2)
                    .unwrap()
                    .parse::<usize>()
                    .unwrap();
                let literals: Vec<i32> = text
                    .lines()
                    .find(|line| line.starts_with("v "))
                    .unwrap()
                    .split_whitespace()
                    .skip(1)
                    .map(|token| token.parse().unwrap())
                    .collect();
                assert_eq!(literals.len(), depth + 1);
                assert_eq!(literals[depth], 0);
                assert!(literals[..depth].iter().copied().eq(1..=depth as i32));
                reads += 1;
                if reads == 1 {
                    ready_tx.send(()).unwrap();
                }
                if !reader_flag.load(Ordering::Acquire) {
                    break;
                }
            }
            reads
        });
        ready_rx.recv().unwrap();
        for depth in [8_192, 64, 16_384, 512, 4_096] {
            write_snapshot(&path, &temporary, &make_snapshot(depth)).unwrap();
        }
        running.store(false, Ordering::Release);
        assert!(reader.join().unwrap() > 0);
        clean(&path);
    }

    #[test]
    fn bounded_sender_drops_full_snapshot_without_waiting() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let sender = HardCubeSender {
            sender,
            closed: Arc::new(AtomicBool::new(false)),
        };
        sender.try_send(snapshot(4, 4)).unwrap();
        assert!(matches!(
            sender.try_send(snapshot(5, 5)),
            Err(TrySendError::Full(_))
        ));
        let Message::Snapshot(queued) = receiver.recv().unwrap() else {
            panic!("expected snapshot");
        };
        assert_eq!(queued.recubings, 4);
        drop(receiver);
        assert!(matches!(
            sender.try_send(snapshot(6, 6)),
            Err(TrySendError::Disconnected(_))
        ));
    }

    #[test]
    fn ignores_shallow_lineage_without_truncating_previous_fragment() {
        let path = path();
        std::fs::write(&path, "previous fragment\n").unwrap();
        let writer = HardCubeWriter::start(&path).unwrap();
        writer.sender().try_send(snapshot(3, 4)).unwrap();
        drop(writer);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "previous fragment\n"
        );
        clean(&path);
    }

    #[test]
    fn invalid_path_fails_before_launch_and_invalid_snapshot_preserves_destination() {
        let path = path();
        assert!(HardCubeWriter::start(path.parent().unwrap()).is_err());
        assert!(
            HardCubeWriter::start(path.parent().unwrap().join("absent").join("x.sat")).is_err()
        );
        std::fs::write(&path, "previous fragment\n").unwrap();
        let writer = HardCubeWriter::start(&path).unwrap();
        let mut invalid = snapshot(4, 4);
        invalid.depth = 19;
        writer.sender().try_send(invalid).unwrap();
        drop(writer);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "previous fragment\n"
        );
        assert_eq!(
            std::fs::read_dir(path.parent().unwrap()).unwrap().count(),
            1
        );
        clean(&path);
    }
}
