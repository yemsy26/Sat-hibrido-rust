# Sat-hibrido-rust

Motor de resolución con certificado, en Rust: fórmulas SAT en DIMACS y restricciones lineales con objetivo
en OPB. Un modelo siempre se comprueba contra el problema, y una prueba de que no hay solución la revisa un
verificador independiente, VeriPB.

Creado y mantenido por **Ramón Antonio Burgos Jerez** ([@yemsy26](https://github.com/yemsy26)), autor y propietario
del proyecto. Las bibliotecas de terceros que entran en los binarios son de sus autores y figuran en
[`NOTICE`](dpll_check/NOTICE).

La documentación completa está en [`dpll_check/README.md`](dpll_check/README.md).

## Instalar sin compilar

Las [versiones publicadas](https://github.com/yemsy26/Sat-hibrido-rust/releases) traen estos archivos para Windows x64:

| Archivo | Para qué |
|---|---|
| `motor-x86_64-pc-windows-msvc.exe` | El comando `motor`, en un solo archivo |
| `motor-x86_64-pc-windows-msvc.zip` | El comando, la biblioteca `motor_ffi.dll` con `motor_ffi.dll.lib` y `motor.h`, las instrucciones y las licencias |
| `*.sha256` | La suma SHA-256 de cada uno |

1. Instala [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist) 2015 o posterior, si no lo tienes.
2. Descarga el archivo y comprueba su suma en PowerShell:

   ```powershell
   (Get-FileHash .\motor-x86_64-pc-windows-msvc.exe -Algorithm SHA256).Hash
   ```

3. Para certificar los UNSAT, instala VeriPB (ver abajo). Si no lo necesitas, usa `--no-verify`.

## Compilar

Rust 1.85 o superior y un compilador de C, que pide zstd.

```sh
git clone https://github.com/yemsy26/Sat-hibrido-rust
cd Sat-hibrido-rust/dpll_check
cargo build --release
cargo build --release -p motor-ffi
```

Queda el comando `dpll_check/target/release/motor` y la biblioteca `motor_ffi`, con su contrato en
[`motor.h`](dpll_check/crates/motor-ffi/include/motor.h).

## Uso

```sh
motor solve problema.cnf --no-verify --timeout 600
motor solve problema.cnf --workers 4 --timeout 600
motor solve asignacion.opb --timeout 600 --json
```

| Código de salida | Significado |
|---|---|
| 10 | SAT, con el modelo comprobado contra el problema |
| 20 | UNSAT; certificado por VeriPB salvo con `--no-verify` |
| 30 | Óptimo de un `.opb` |
| 124 | Se agotó `--timeout` |
| 1 | Error o resultado descartado |

## Qué es interno y qué no

| Pieza | Dónde está |
|---|---|
| Solver, generadores, API de Rust y de C | En este repositorio |
| Chronos-VFS | En este repositorio, `dpll_check/crates/chronos-vfs` |
| zstd | Lo baja y compila Cargo; queda dentro del binario |
| **VeriPB** | **Programa aparte, en el `PATH`.** Es el único proceso externo que lanza `motor`; está fuera a propósito para que el certificado lo revise un programa que no comparte código con el solver |
| Toolkit CUDA, TensorRT | Instalación de NVIDIA, solo para el motor GPU, que no va en los binarios publicados |

Sin VeriPB y sin `--no-verify`, `motor solve` sale con código 1 antes de buscar. Con `--no-verify` un UNSAT no
queda certificado y se avisa. Cómo instalar VeriPB: [`dpll_check/README.md`](dpll_check/README.md#instalación).

## Versiones publicadas

- [`v0.2.2`](https://github.com/yemsy26/Sat-hibrido-rust/releases/tag/v0.2.2): la versión actual.
- [`v0.2.1`](https://github.com/yemsy26/Sat-hibrido-rust/releases/tag/v0.2.1): los mismos archivos, con este número
  para [Aura Sentinel](https://github.com/yemsy26/aura-sentinel) 4.11.1, que descarga esa versión.

## Licencia

Apache-2.0: ver [`LICENSE`](dpll_check/LICENSE). Chronos-VFS tiene doble licencia, MIT o Apache-2.0.

© 2026 Ramón Antonio Burgos Jerez
