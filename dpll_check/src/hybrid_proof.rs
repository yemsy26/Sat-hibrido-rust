// Trazas DRAT. Cada línea añade una cláusula que es RUP respecto de las anteriores, o RAT
// sobre su primer literal, o borra una cláusula ya presente. El motor solo escribe; la
// comprobación corresponde a un verificador externo (drat-trim, que además produce LRAT
// para un verificador certificado).
use std::io::{self, Write};

pub trait ProofLog {
    fn add(&mut self, clause: &[i32]);
    fn delete(&mut self, clause: &[i32]);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProofFormat {
    Text,
    Binary,
}

// Una llamada al sistema por cada 64 KiB de traza, no por cláusula.
const FLUSH_AT: usize = 1 << 16;

pub struct Proof<W: Write> {
    sink: W,
    buffer: Vec<u8>,
    format: ProofFormat,
    additions: u64,
    deletions: u64,
    bytes: u64,
    error: Option<io::Error>,
}
impl<W: Write> Proof<W> {
    pub fn new(sink: W, format: ProofFormat) -> Self {
        Self {
            sink, format,
            buffer: Vec::with_capacity(FLUSH_AT + 4096),
            additions: 0, deletions: 0, bytes: 0, error: None,
        }
    }
    pub fn additions(&self) -> u64 {
        self.additions
    }
    pub fn deletions(&self) -> u64 {
        self.deletions
    }
    // Bytes ya entregados al destino más los que esperan en el búfer.
    pub fn bytes(&self) -> u64 {
        self.bytes + self.buffer.len() as u64
    }
    fn line(&mut self, clause: &[i32], deletion: bool) {
        match self.format {
            ProofFormat::Text => {
                if deletion {
                    self.buffer.extend_from_slice(b"d ");
                }
                for &literal in clause {
                    if literal < 0 {
                        self.buffer.push(b'-');
                    }
                    let mut digits = [0_u8; 10];
                    let mut value = literal.unsigned_abs();
                    let mut start = digits.len();
                    loop {
                        start -= 1;
                        digits[start] = b'0' + (value % 10) as u8;
                        value /= 10;
                        if value == 0 {
                            break;
                        }
                    }
                    self.buffer.extend_from_slice(&digits[start..]);
                    self.buffer.push(b' ');
                }
                self.buffer.extend_from_slice(b"0\n");
            }
            ProofFormat::Binary => {
                // DRAT binario: 'a' o 'd', literales 2|x| + signo en base 128, y un cero.
                self.buffer.push(if deletion { b'd' } else { b'a' });
                for &literal in clause {
                    let mut code = 2 * u64::from(literal.unsigned_abs()) + u64::from(literal < 0);
                    while code >= 128 {
                        self.buffer.push((code & 127) as u8 | 128);
                        code >>= 7;
                    }
                    self.buffer.push(code as u8);
                }
                self.buffer.push(0);
            }
        }
        if self.buffer.len() >= FLUSH_AT {
            self.drain();
        }
    }
    fn drain(&mut self) {
        if self.error.is_none() {
            if let Err(error) = self.sink.write_all(&self.buffer) {
                self.error = Some(error);
            }
        }
        self.bytes += self.buffer.len() as u64;
        self.buffer.clear();
    }
    // Entrega lo pendiente y devuelve el destino; un error de escritura invalida la traza.
    pub fn finish(mut self) -> io::Result<W> {
        self.drain();
        if let Some(error) = self.error {
            return Err(error);
        }
        self.sink.flush()?;
        Ok(self.sink)
    }
}
impl<W: Write> ProofLog for Proof<W> {
    fn add(&mut self, clause: &[i32]) {
        self.additions += 1;
        self.line(clause, false);
    }
    fn delete(&mut self, clause: &[i32]) {
        self.deletions += 1;
        self.line(clause, true);
    }
}
