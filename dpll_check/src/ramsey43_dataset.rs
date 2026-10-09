//! Optional offline dataset sink. Only already sampled host batches enter this
//! module: the search thread never waits for disk or for space in the queue.
//!
//! Wire format is explicit little endian (no Rust struct-layout dependency).
//! The 64-byte NR43DS01 header is followed by fixed 2496-byte parent records.
//! Files are consumed after `finish`: shutdown drains the bounded queue and
//! flushes its 4 MiB buffer. No fsync or small write is issued per branch.

use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const MAGIC: &[u8; 8] = b"NR43DS01";
pub const VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 64;
pub const RECORD_BYTES: usize = 2496;
pub const CANDIDATES: usize = 32;
pub const FEATURES: usize = 32;
pub const PADDING_EDGE: u16 = 903;
pub const METRIC_BOUNDED_DFS_TRANSITIONS: u32 = 1;
pub const STATUS_PADDING: u8 = 0;
pub const STATUS_IMMEDIATE_K5: u8 = 1;
pub const STATUS_REFUTED: u8 = 2;
pub const STATUS_CENSORED: u8 = 3;
pub const STATUS_SAT: u8 = 4;
const IO_BUFFER_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct DatasetConfig {
    pub path: PathBuf,
    pub sample_every: u32,
    pub horizon: u32,
    pub seed: u64,
    pub queue_batches: usize,
    pub max_batch_records: usize,
}

impl DatasetConfig {
    pub fn validate(&self) -> io::Result<()> {
        if self.sample_every == 0 || self.horizon == 0 {
            return Err(invalid("dataset sample_every and horizon must be positive"));
        }
        if self.queue_batches == 0 || self.max_batch_records == 0 {
            return Err(invalid(
                "dataset queue and batch capacities must be positive",
            ));
        }
        self.max_batch_records
            .checked_mul(RECORD_BYTES)
            .and_then(|bytes| bytes.checked_mul(self.queue_batches))
            .ok_or_else(|| invalid("dataset queue capacity overflows usize"))?;
        Ok(())
    }
}

/// Each candidate row describes the unmodified parent graph. Its two cost and
/// status entries describe independent red/blue bounded probes of that parent.
#[derive(Clone, Debug)]
pub struct DatasetRecord {
    pub run_id: u64,
    pub decision_seq: u64,
    pub universe: u32,
    pub depth: u32,
    pub chosen_edge: u16,
    pub chosen_phase: u8,
    pub canonical_degree: u8,
    pub order_seed: u32,
    pub prefix_bits: u32,
    pub horizon: u32,
    pub candidate_count: u32,
    pub parent_hash: u64,
    pub x: [u16; CANDIDATES * FEATURES],
    pub ids: [u16; CANDIDATES],
    pub cost: [u32; CANDIDATES * 2],
    pub status: [u8; CANDIDATES * 2],
}

impl DatasetRecord {
    /// Validation also prevents a stale/padded GPU slot from becoming a label.
    pub fn validate(&self, horizon: u32) -> io::Result<()> {
        if self.horizon != horizon || horizon == 0 {
            return Err(invalid("dataset record/header probe horizons disagree"));
        }
        if self.chosen_edge >= PADDING_EDGE || self.chosen_phase > 1 {
            return Err(invalid("dataset selected edge/color is invalid"));
        }
        if self.canonical_degree != 255 && !(18..=21).contains(&self.canonical_degree) {
            return Err(invalid("dataset canonical degree must be 18..21 or 255"));
        }
        if self.depth > u32::from(PADDING_EDGE) || self.prefix_bits > u32::from(PADDING_EDGE) {
            return Err(invalid("dataset depth or prefix exceeds K43 edges"));
        }
        if self.candidate_count == 0 || self.candidate_count as usize > CANDIDATES {
            return Err(invalid("dataset candidate count must be 1..32"));
        }
        for candidate in 0..CANDIDATES {
            let row = &self.x[candidate * FEATURES..(candidate + 1) * FEATURES];
            let offset = candidate * 2;
            if candidate >= self.candidate_count as usize {
                if self.ids[candidate] != PADDING_EDGE
                    || row.iter().any(|&value| value != 0)
                    || self.cost[offset..offset + 2]
                        .iter()
                        .any(|&value| value != 0)
                    || self.status[offset..offset + 2]
                        .iter()
                        .any(|&value| value != STATUS_PADDING)
                {
                    return Err(invalid("dataset padding must have ID 903 and zero payload"));
                }
                continue;
            }
            let edge = self.ids[candidate];
            if edge >= PADDING_EDGE || self.ids[..candidate].contains(&edge) {
                return Err(invalid("dataset candidate IDs are invalid or duplicated"));
            }
            // Features v1 are normalized nonnegative finite FP16 values in [0,1].
            if row.iter().any(|&bits| bits > 0x3c00) {
                return Err(invalid("dataset features must be finite FP16 in [0,1]"));
            }
            for color in 0..2 {
                let cost = self.cost[offset + color];
                match self.status[offset + color] {
                    STATUS_IMMEDIATE_K5 if cost == 1 => {}
                    STATUS_REFUTED if (1..=horizon).contains(&cost) => {}
                    STATUS_CENSORED if cost == horizon => {}
                    STATUS_SAT if (1..=horizon).contains(&cost) => {}
                    _ => return Err(invalid("dataset outcome/cost pair violates schema v1")),
                }
            }
        }
        Ok(())
    }

    fn encode(&self, bytes: &mut Vec<u8>) {
        let start = bytes.len();
        append_u64(bytes, self.run_id);
        append_u64(bytes, self.decision_seq);
        append_u32(bytes, self.universe);
        append_u32(bytes, self.depth);
        append_u16(bytes, self.chosen_edge);
        bytes.push(self.chosen_phase);
        bytes.push(self.canonical_degree);
        append_u32(bytes, self.order_seed);
        append_u32(bytes, self.prefix_bits);
        append_u32(bytes, self.horizon);
        append_u32(bytes, self.candidate_count);
        append_u32(bytes, 0);
        append_u64(bytes, self.parent_hash);
        append_u64(bytes, 0);
        for value in self.x {
            append_u16(bytes, value);
        }
        for value in self.ids {
            append_u16(bytes, value);
        }
        for value in self.cost {
            append_u32(bytes, value);
        }
        bytes.extend_from_slice(&self.status);
        debug_assert_eq!(bytes.len() - start, RECORD_BYTES);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DatasetStats {
    /// Records accepted by the in-memory queue; finish verifies disk completion.
    pub submitted_records: u64,
    pub dropped_records: u64,
    pub dropped_batches: u64,
    pub written_records: u64,
    pub written_batches: u64,
}

/// A single producer owns this sink. Queue saturation is an accounted sample
/// drop; writer failure is returned as an error, never silently accepted.
pub struct DatasetExporter {
    sender: Option<SyncSender<Vec<DatasetRecord>>>,
    writer: Option<JoinHandle<io::Result<DatasetStats>>>,
    config: DatasetConfig,
    stats: DatasetStats,
}

impl DatasetExporter {
    pub fn start(config: DatasetConfig) -> io::Result<Self> {
        config.validate()?;
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| invalid("dataset creation clock predates Unix epoch"))?
            .as_nanos();
        let created = u64::try_from(created)
            .map_err(|_| invalid("dataset timestamp overflows u64 nanoseconds"))?;
        // create_new protects previous runs, including when a caller accidentally
        // reuses a CLI output path. The parent directory must already exist.
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&config.path)?;
        let mut output = BufWriter::with_capacity(IO_BUFFER_BYTES, file);
        output.write_all(&encode_header(&config, created))?;
        let (sender, receiver) = mpsc::sync_channel(config.queue_batches);
        let writer = thread::Builder::new()
            .name("ramsey43-dataset-writer".into())
            .spawn(move || writer_loop(receiver, output))?;
        Ok(Self {
            sender: Some(sender),
            writer: Some(writer),
            config,
            stats: DatasetStats::default(),
        })
    }

    /// Returns false when this entire batch was dropped under backpressure.
    /// Validation and try_send perform no filesystem operation or blocking wait.
    pub fn try_submit(&mut self, records: Vec<DatasetRecord>) -> io::Result<bool> {
        if records.is_empty() {
            return Ok(true);
        }
        if records.len() > self.config.max_batch_records {
            return Err(invalid("dataset batch exceeds configured bounded capacity"));
        }
        for record in &records {
            record.validate(self.config.horizon)?;
        }
        let count = records.len() as u64;
        let sender = self.sender.as_ref().ok_or_else(|| {
            io::Error::new(io::ErrorKind::BrokenPipe, "dataset exporter is closed")
        })?;
        match sender.try_send(records) {
            Ok(()) => {
                self.stats.submitted_records += count;
                Ok(true)
            }
            Err(TrySendError::Full(_)) => {
                self.stats.dropped_records += count;
                self.stats.dropped_batches += 1;
                Ok(false)
            }
            Err(TrySendError::Disconnected(_)) => Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "dataset writer stopped; finish() reports the original I/O error",
            )),
        }
    }

    /// Counts available without waiting for disk. Written counters become final
    /// only when finish succeeds, after all accepted batches and flush complete.
    pub fn stats(&self) -> DatasetStats {
        self.stats
    }

    pub fn config(&self) -> &DatasetConfig {
        &self.config
    }

    pub fn finish(mut self) -> io::Result<DatasetStats> {
        self.close()
    }

    fn close(&mut self) -> io::Result<DatasetStats> {
        // Disconnect before joining: the consumer exits after draining its queue.
        drop(self.sender.take());
        if let Some(writer) = self.writer.take() {
            let written = writer
                .join()
                .map_err(|_| io::Error::other("dataset writer thread panicked"))??;
            self.stats.written_records = written.written_records;
            self.stats.written_batches = written.written_batches;
        }
        Ok(self.stats)
    }
}

impl Drop for DatasetExporter {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            eprintln!("c dataset teardown: {error}");
        }
    }
}

fn writer_loop<W: Write>(
    receiver: Receiver<Vec<DatasetRecord>>,
    mut output: W,
) -> io::Result<DatasetStats> {
    let mut stats = DatasetStats::default();
    let mut bytes = Vec::new();
    let mut next_flush = Instant::now() + Duration::from_secs(5);
    let mut dirty = true; // Includes the header buffered before spawning.
    loop {
        match receiver.recv_timeout(next_flush.saturating_duration_since(Instant::now())) {
            Ok(batch) => {
                bytes.clear();
                bytes.reserve(batch.len() * RECORD_BYTES);
                for record in &batch {
                    record.encode(&mut bytes);
                }
                output.write_all(&bytes)?;
                stats.written_records += batch.len() as u64;
                stats.written_batches += 1;
                dirty = true;
            }
            Err(RecvTimeoutError::Timeout) => {
                if dirty {
                    output.flush()?;
                    dirty = false;
                }
                next_flush = Instant::now() + Duration::from_secs(5);
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    output.flush()?;
    Ok(stats)
}

fn encode_header(config: &DatasetConfig, created: u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HEADER_BYTES);
    bytes.extend_from_slice(MAGIC);
    for value in [
        VERSION,
        HEADER_BYTES as u32,
        RECORD_BYTES as u32,
        CANDIDATES as u32,
        FEATURES as u32,
        METRIC_BOUNDED_DFS_TRANSITIONS,
        config.sample_every,
        config.horizon,
    ] {
        append_u32(&mut bytes, value);
    }
    append_u64(&mut bytes, config.seed);
    append_u64(&mut bytes, created);
    append_u64(&mut bytes, 0);
    debug_assert_eq!(bytes.len(), HEADER_BYTES);
    bytes
}

fn append_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn append_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn append_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

/// Useful for fixture verification without touching the GPU. The feature
/// extractor/collector supplies the FNV parent hash, not this disk writer.
pub fn encoded_record(record: &DatasetRecord) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(RECORD_BYTES);
    record.encode(&mut bytes);
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_ID: AtomicU64 = AtomicU64::new(0);

    fn config() -> DatasetConfig {
        DatasetConfig {
            path: std::env::temp_dir().join(format!(
                "ramsey43-dataset-{}-{}.nrds",
                std::process::id(),
                TEMP_ID.fetch_add(1, Ordering::Relaxed)
            )),
            sample_every: 10_000,
            horizon: 8,
            seed: 123456789,
            queue_batches: 2,
            max_batch_records: 32,
        }
    }

    fn record() -> DatasetRecord {
        let mut x = [0; CANDIDATES * FEATURES];
        x[0] = 0x3c00;
        x[1] = 0x3800;
        let mut ids = [PADDING_EDGE; CANDIDATES];
        ids[0] = 42;
        let mut cost = [0; CANDIDATES * 2];
        cost[..2].copy_from_slice(&[1, 8]);
        let mut status = [STATUS_PADDING; CANDIDATES * 2];
        status[..2].copy_from_slice(&[STATUS_IMMEDIATE_K5, STATUS_CENSORED]);
        DatasetRecord {
            run_id: 0x0102030405060708,
            decision_seq: 99,
            universe: 7,
            depth: 13,
            chosen_edge: 42,
            chosen_phase: 1,
            canonical_degree: 18,
            order_seed: 100,
            prefix_bits: 4,
            horizon: 8,
            candidate_count: 1,
            parent_hash: 0x8899aabbccddeeff,
            x,
            ids,
            cost,
            status,
        }
    }

    #[test]
    fn header_and_record_have_exact_little_endian_layout() {
        let header = encode_header(&config(), 42);
        assert_eq!(header.len(), HEADER_BYTES);
        assert_eq!(&header[..8], MAGIC);
        assert_eq!(&header[16..20], &(RECORD_BYTES as u32).to_le_bytes());
        assert_eq!(&header[48..56], &42_u64.to_le_bytes());
        let bytes = encoded_record(&record());
        assert_eq!(bytes.len(), RECORD_BYTES);
        assert_eq!(&bytes[..8], &0x0102030405060708_u64.to_le_bytes());
        assert_eq!(&bytes[48..56], &0x8899aabbccddeeff_u64.to_le_bytes());
        assert_eq!(&bytes[64..68], &[0, 0x3c, 0, 0x38]);
        assert_eq!(&bytes[2112..2116], &[42, 0, 0x87, 3]);
        assert_eq!(&bytes[2176..2184], &[1, 0, 0, 0, 8, 0, 0, 0]);
        assert_eq!(&bytes[2432..2434], &[STATUS_IMMEDIATE_K5, STATUS_CENSORED]);
        assert!(bytes[44..48].iter().chain(&bytes[56..64]).all(|&x| x == 0));
    }

    #[test]
    fn rejects_invalid_labels_and_padding() {
        let good = record();
        good.validate(8).unwrap();
        let mut bad = good.clone();
        bad.status[0] = 5;
        assert!(bad.validate(8).is_err());
        let mut bad = good.clone();
        bad.cost[1] = 7;
        assert!(bad.validate(8).is_err());
        let mut bad = good.clone();
        bad.x[32] = 1;
        assert!(bad.validate(8).is_err());
        let mut bad = good.clone();
        bad.x[0] = 0x7c00;
        assert!(bad.validate(8).is_err());
        assert!(good.validate(7).is_err());
    }

    #[test]
    fn finish_drains_and_refuses_to_overwrite_existing_file() {
        let config = config();
        let path = config.path.clone();
        let mut exporter = DatasetExporter::start(config.clone()).unwrap();
        assert!(exporter.try_submit(vec![record(), record()]).unwrap());
        assert_eq!(
            DatasetExporter::start(config).err().unwrap().kind(),
            io::ErrorKind::AlreadyExists
        );
        let stats = exporter.finish().unwrap();
        assert_eq!(stats.submitted_records, 2);
        assert_eq!(stats.written_records, 2);
        assert_eq!(stats.written_batches, 1);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), HEADER_BYTES + 2 * RECORD_BYTES);
        assert_eq!(
            &bytes[HEADER_BYTES..HEADER_BYTES + RECORD_BYTES],
            &encoded_record(&record())
        );
        // Optional persistent fixture for the Python schema compatibility test.
        if let Some(fixture_path) = std::env::var_os("RAMSEY43_DATASET_FIXTURE") {
            std::fs::write(PathBuf::from(fixture_path), &bytes).unwrap();
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn bounded_queue_drops_without_blocking() {
        let (sender, receiver) = mpsc::sync_channel(1);
        // Keeping an unpolled receiver makes full-queue behavior deterministic.
        let mut exporter = DatasetExporter {
            sender: Some(sender),
            writer: None,
            config: config(),
            stats: DatasetStats::default(),
        };
        assert!(exporter.try_submit(vec![record()]).unwrap());
        assert!(!exporter.try_submit(vec![record(), record()]).unwrap());
        assert_eq!(exporter.stats().submitted_records, 1);
        assert_eq!(exporter.stats().dropped_records, 2);
        assert_eq!(exporter.stats().dropped_batches, 1);
        assert!(exporter.try_submit(vec![record(); 33]).is_err());
        drop(receiver);
        assert_eq!(
            exporter.try_submit(vec![record()]).unwrap_err().kind(),
            io::ErrorKind::BrokenPipe
        );
    }

    #[test]
    fn finish_propagates_writer_and_flush_errors() {
        struct FailingWriter {
            fail_write: bool,
        }
        impl Write for FailingWriter {
            fn write(&mut self, _bytes: &[u8]) -> io::Result<usize> {
                if self.fail_write {
                    Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "fixture disk error",
                    ))
                } else {
                    Ok(_bytes.len())
                }
            }
            fn flush(&mut self) -> io::Result<()> {
                Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "fixture flush error",
                ))
            }
        }
        for fail_write in [true, false] {
            let (sender, receiver) = mpsc::sync_channel(1);
            let writer = thread::spawn(move || writer_loop(receiver, FailingWriter { fail_write }));
            let mut exporter = DatasetExporter {
                sender: Some(sender),
                writer: Some(writer),
                config: config(),
                stats: DatasetStats::default(),
            };
            exporter.try_submit(vec![record()]).unwrap();
            let error = exporter.finish().unwrap_err();
            assert_eq!(
                error.kind(),
                if fail_write {
                    io::ErrorKind::PermissionDenied
                } else {
                    io::ErrorKind::WriteZero
                }
            );
        }
    }
}
