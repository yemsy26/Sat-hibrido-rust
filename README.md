# Sat-hibrido-rust

Resolvedor SAT híbrido en Rust (sin `unsafe`): búsqueda CDCL con capa algebraica de paridad sobre GF(2), ruptura de
simetrías, un oráculo de búsqueda local que reorienta la búsqueda y un enjambre de hilos que comparte cláusulas.
Lee fórmulas DIMACS CNF, devuelve el modelo comprobado o la insatisfacibilidad, y puede emitir una traza de prueba
VeriPB 3.0 que certifica la respuesta.

Creado por **Ramón Antonio Burgos Jerez** ([@yemsy26](https://github.com/yemsy26)).

## Qué incluye

- **CDCL**: arena lineal de cláusulas, watchers de 8 bytes, VMTF/EVSIDS y reinicios.
- **Paridad sobre GF(2)**: detecta restricciones XOR camufladas en la CNF y las resuelve por eliminación gaussiana
  (sin esta capa, esas familias crecen de forma exponencial).
- **Simetrías**: detecta automorfismos de la fórmula y añade predicados de ruptura (certificados con la regla de
  dominancia de VeriPB).
- **Oráculo probSAT** con re-ignición: cada cierto número de conflictos reorienta las fases de la búsqueda; el
  resultado SAT/UNSAT siempre lo decide el CDCL.
- **Enjambre** (cube-and-conquer): varios hilos con cubos propios y un anillo de cláusulas compartidas filtradas por
  LBD.
- **Trazas certificables**: DRAT y VeriPB 3.0. Con [Chronos-VFS](https://github.com/yemsy26/Chronos-VFS) la traza
  pasa por un anillo en RAM y se comprime con zstd (`.pbp.zst`).

## Compilar

Rust 1.85 o superior ([rustup](https://rustup.rs/)). Chronos-VFS llega solo como dependencia de git (versión fijada
en `dpll_check/Cargo.toml`).

```sh
git clone https://github.com/yemsy26/Sat-hibrido-rust
cd Sat-hibrido-rust/dpll_check
cargo build --release --bin hunt
cargo test --release
```

El ejecutable queda en `dpll_check/target/release/hunt` (`hunt.exe` en Windows).

## Uso: `hunt`

```sh
hunt formula.cnf --model formula.modelo --timeout 60
hunt formula.cnf --workers 4
hunt formula.cnf --proof traza.pbp
```

| Opción | Qué hace |
| --- | --- |
| `--workers K` | enjambre de K hilos (sin esta opción, uno) |
| `--cubes C` | cubos por hilo (4) |
| `--no-share` | enjambre sin intercambio de cláusulas |
| `--no-symmetry`, `--no-parity` | sin ruptura de simetrías / sin capa de paridad |
| `--flips F`, `--interval I`, `--seed S` | ajustes del oráculo (I = 0 lo apaga) |
| `--proof F.pbp` | traza VeriPB 3.0 (solo en serie) |
| `--proof-cap-gib G` | tope de tamaño de la traza; si se supera, sigue sin certificado |
| `--model F` | escribe el modelo si es satisfacible: una línea `v` con los literales (5 = x5 verdadera, -5 = falsa) |
| `--preprocess-only` | solo simetrías y paridad, sin búsqueda |
| `--timeout S` | se detiene a los S segundos |
| `--heartbeat S` | línea de progreso en stderr cada S segundos |
| `--csv ETIQUETA` | una fila CSV en la salida (`hunt --header` da la cabecera) |

Códigos de salida de la competición SAT: **10** satisfacible, **20** insatisfacible, **124** sin respuesta en el
tiempo dado.

## Otros programas (`dpll_check/src/bin`)

- `certify` y `chronos_certify`: resuelven y dejan la traza VeriPB (en disco o por Chronos-VFS, comprimida).
- `gen_ramsey` y `gen_erdos`: generan en DIMACS las fórmulas de Ramsey R(5,5) y de la discrepancia de Erdős.
- `parity_stress`: estrés de restricciones XOR con y sin la capa algebraica.
- `phase3` … `phase10`: bancos de pruebas de cada fase del desarrollo; sus resultados están en `dpll_check/results/`.
- `calibrate_ramsey.sh` y `run_ramsey_43.sh`: calibración y lanzamiento de R(5,5) con N = 43.

## Versiones publicadas

Cada versión en [Releases](https://github.com/yemsy26/Sat-hibrido-rust/releases) trae el binario para Windows
`hunt-x86_64-pc-windows-msvc.exe` y su `hunt-x86_64-pc-windows-msvc.exe.sha256`. Para comprobarlo en PowerShell:

```powershell
(Get-FileHash .\hunt-x86_64-pc-windows-msvc.exe).Hash
```

## Uso en Aura Sentinel

[Aura Sentinel](https://github.com/yemsy26/aura-sentinel) lo usa como motor complementario: al activarlo en
**Configuración → Motor → Motores complementarios**, descarga el binario de la última versión publicada (comprobando
su SHA-256) o lo compila desde este repositorio, y el agente gana la herramienta `resolver_sat` para sudokus,
horarios, coloreado y otros problemas de restricciones.

## Licencia

Doble licencia, a elección de quien lo use: [MIT](LICENSE-MIT) o [Apache-2.0](LICENSE-APACHE), como Chronos-VFS.
© 2026 Ramón Antonio Burgos Jerez.
