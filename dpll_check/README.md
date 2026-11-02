# Motor de resolución con certificado

![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.6_%7C_13.x-green.svg)
![TensorRT](https://img.shields.io/badge/TensorRT-11.3-76B900.svg)
[![Licencia: Apache 2.0](https://img.shields.io/badge/licencia-Apache_2.0-blue.svg)](LICENSE)

`motor` resuelve problemas finitos sobre variables booleanas y no da un resultado por bueno sin
comprobarlo: un modelo se evalúa contra el fichero de entrada y una prueba de que no hay solución la
revisa un verificador independiente.

Nació para atacar el número de Ramsey R(5,5) y conserva ese motor GPU, que **no ha resuelto R(5,5)
ni está cerca**. Lo que sí resuelve, y lo que no, está medido en [Límites](#límites).

## Qué hace

| Comando | Para qué sirve | Necesita |
|---|---|---|
| `motor solve f.cnf` | Fórmulas booleanas en DIMACS, en serie o con `--workers K` | VeriPB para certificar UNSAT |
| `motor solve f.opb` | Restricciones lineales y objetivo `min:` | VeriPB para certificar UNSAT y óptimos |
| `motor gen ramsey\|erdos` | Genera fórmulas de Ramsey y de discrepancia de Erdős | Nada |
| `motor bench` | Banco del DFS de referencia sobre números de Ramsey conocidos | Nada |
| `motor trace f.cnf` | Mide el puente de la traza; **no** comprueba el resultado | Nada |
| `motor k43` | Búsqueda GPU exclusiva de K43 | GPU NVIDIA y `--features gpu` o `neuro-trt` |

Lo mismo que hace `motor solve` se puede llamar desde otro programa: ver
[Integrarlo en otro programa](#integrarlo-en-otro-programa).

## De qué depende

Todo el código propio se compila en un binario, una biblioteca de Rust (`motor_core`) y una de C
(`motor_ffi`). Lo que antes
vivía aparte ya está dentro, pero **no todo puede estarlo**:

| Pieza | Dónde está | Cuándo hace falta |
|---|---|---|
| Solver, generadores, bancos y motor GPU | En este repositorio | Siempre |
| Chronos-VFS (anillo en RAM y compresión de trazas) | En [`crates/chronos-vfs`](crates/chronos-vfs) | Siempre; se compila solo |
| Rust 1.85 o superior (1.88 con `neuro-mock` o `neuro-trt`) y un compilador de C | Hay que instalarlos | Para compilar; el C lo pide zstd |
| **VeriPB** | **Programa aparte, en el `PATH`** ([cómo instalarlo](#instalación)) | **Para certificar UNSAT y óptimos** |
| Toolkit CUDA y Visual Studio | Instalación de NVIDIA y Microsoft | Solo para la GPU |
| SDK de TensorRT | Instalación de NVIDIA | Solo con `--features neuro-trt` |
| Python | Hay que instalarlo | Solo para `sat/`, `scripts/`, el centinela y el visor |
| CaDiCaL y drat-trim | Los descarga y compila `sat/build_tools.sh` | Solo para `sat/` |

**VeriPB es el único proceso externo que lanza `motor`, y está fuera a propósito.** Un certificado
vale porque lo revisa un programa que no comparte código con quien lo escribe. Si no está, `motor`
avisa antes de empezar a buscar y sale con código 1, aunque la fórmula tuviera solución. Con
`--no-verify` trabaja sin él: cada UNSAT sale con un aviso de que no está certificado, y cada modelo
se sigue comprobando, porque eso lo hace el propio `motor`.

**El SDK de TensorRT tampoco puede ir en el repositorio:** pesa 3,3 GB, tiene ficheros de más de
100 MB (el límite de GitHub) y su licencia es propietaria de NVIDIA. `build.rs` usa `TENSORRT_ROOT`
o, si no está definida, la versión más nueva de `C:\SDK\TensorRT-*` y después `C:\TensorRT`. El
binario encuentra sus DLL sin tocar el `PATH`, pero solo en la máquina donde se compiló.

## Instalación

```powershell
cargo build --release
```

Eso basta para buscar soluciones y para trabajar con `--no-verify`. Para certificar hace falta VeriPB,
que se compila con Cargo y queda en `~/.cargo/bin`, que debe estar en el `PATH`:

```powershell
git clone --branch 3.0.2 --depth 1 https://gitlab.com/MIAOresearch/software/VeriPB.git
cargo install --path VeriPB --locked
veripb --version
```

Las pruebas de este repositorio se han pasado con VeriPB 3.0.2, que necesita Rust 1.84. Su rama `main`
ya pide Rust 1.92.

## Cómo funciona

`motor solve` sigue siempre el mismo recorrido:

1. **Lectura.** Carga el DIMACS. Un `.opb` se traduce antes a CNF con un sumador binario.
2. **Simetrías.** Busca automorfismos de la fórmula y añade restricciones lex-leader que eliminan las
   soluciones equivalentes. Cada una se justifica en la traza con la regla de dominancia.
3. **Paridad.** Extrae las ecuaciones XOR ocultas y las reduce por eliminación gaussiana, algo que
   CDCL por sí solo hace muy mal. Si llega a 0 = 1, la fórmula ya está refutada. Si no, añade solo las
   filas de hasta 4 variables.
4. **Búsqueda.** CDCL con aprendizaje de cláusulas y un oráculo de búsqueda local que orienta las
   decisiones. Con `--workers K` la fórmula se reparte en cubos entre K hilos, que comparten sus
   mejores cláusulas.
5. **Traza.** Cada paso se escribe en formato VeriPB 3.0. Todos los hilos escriben en una sola traza:
   cada uno vuelca lo aprendido antes de publicar una cláusula, de modo que nadie usa algo que aún no
   conste, y un árbol de resolución une los cubos cerrados hasta la cláusula vacía. Con `--proof f.pbp.zst`
   pasa por el anillo en RAM de Chronos-VFS y se comprime al vuelo.
6. **Comprobación.** Un SAT se evalúa contra la fórmula del fichero. Un UNSAT solo se anuncia si VeriPB
   acepta la traza. Si la rechaza, el resultado se descarta.

Por eso un error del solver puede dar un `s UNKNOWN`, pero no un UNSAT certificado falso. Lo que queda
dentro de lo que hay que creer es VeriPB y, en los `.opb`, el traductor a CNF (ver abajo).

## Uso

```powershell
.\target\release\motor solve problema.cnf
.\target\release\motor solve problema.cnf --workers 4 --proof prueba.pbp.zst --timeout 600
.\target\release\motor solve asignacion.opb --timeout 600
.\target\release\motor gen ramsey 14 r35.cnf --cliques 3 5
.\target\release\motor gen erdos 12 e.cnf --c 1
.\target\release\motor bench --budget 1000000 --only 3,6,18
.\target\release\motor trace problema.cnf --route check
```

| Salida | Código | Qué se ha comprobado |
|---|---|---|
| `s SATISFIABLE` y línea `v` | 10 | El modelo, contra el fichero |
| `s UNSATISFIABLE` | 20 | La prueba, con VeriPB, salvo con `--no-verify`, y entonces se avisa |
| `s OPTIMUM FOUND` y línea `v` | 30 | El modelo, contra el fichero, y que no hay otro mejor, con VeriPB |
| `s UNKNOWN` | 124 | Nada: se agotó `--timeout` |
| `s SATISFIABLE` y línea `v` | 124 | En un `.opb`: hay solución comprobada, pero no se llegó a probar que sea óptima |
| `s UNKNOWN` | 1 | Resultado descartado: VeriPB rechazó la prueba o el modelo no era válido |

`--recube-conflicts` y `--ramsey-symm N` no emiten prueba: sin `--no-verify` el comando se niega a
ejecutarlos. `--use-gpu` exige `--workers` y compilar con `--features gpu`. La lista completa de opciones está al
principio de [`src/bin/motor.rs`](src/bin/motor.rs).

### Restricciones lineales y optimización (`.opb`)

Cada línea es una restricción lineal sobre variables booleanas, con `>=`, `<=` o `=`. Un objetivo
`min:` opcional pide la solución de menor coste:

```text
min: +3 x1 +2 x2 +4 x3 ;
+1 x1 +1 x2 +1 x3 >= 2 ;
+2 x1 -1 ~x3 <= 1 ;
```

Sale `o <coste>` por cada mejora y `s OPTIMUM FOUND` al final. Admite `--workers`, `--json`, `--cnf`
para conservar la última CNF y `--proof` para conservar su traza. Una refutación pasa por VeriPB, pero
**sobre la CNF traducida**: el traductor no está certificado y hay que creerlo; `tests/motor_opb.rs` lo
compara con la fuerza bruta. Los términos no lineales (`+1 x1 x2`) se rechazan.

## Integrarlo en otro programa

Las tres vías hacen el mismo recorrido y dan la misma garantía:

* `Sat` y `Optimum` traen un modelo ya evaluado contra el problema.
* `Unsat` y `Optimum` los ha aceptado VeriPB si `proof_checked` es cierto; con `certify` desactivado es
  falso y hay que creer al solver. En un `.opb`, VeriPB comprueba la CNF traducida.
* Un límite de tiempo, una parada o una prueba rechazada nunca salen como `Unsat`.

Para certificar hace falta VeriPB también aquí: es un proceso aparte que la biblioteca lanza por ti. Sin
él, la llamada falla antes de buscar, salvo que se desactive `certify`. Se pueden lanzar varias
resoluciones a la vez, cada una con sus hilos y sus ficheros temporales.

**Desde Rust**, con una de estas dependencias. No está en crates.io. El paquete se llama `dpll_check` y se importa como `motor_core`. Solo
[`motor_core::solver`](src/solver.rs) es la API; el resto de módulos es interno y puede cambiar:

```toml
dpll_check = { path = "ruta/a/dpll_check" }
dpll_check = { git = "https://github.com/yemsy26/Sat-hibrido-rust.git" }
```

```rust
use motor_core::solver::{solve, Options, Problem, Status, StopToken};
use std::time::Duration;

let problem = Problem::from_file("problema.cnf")?; // también Problem::cnf, dimacs, opb y pb
let stop = StopToken::default();                   // stop.stop() desde cualquier hilo
let options = Options {
    workers: Some(4),
    timeout: Some(Duration::from_secs(600)),
    stop: Some(stop.clone()),
    ..Options::default()
};
let solution = solve(&problem, &options)?;
match solution.status {
    Status::Sat | Status::Optimum => println!("{:?} coste {:?}", solution.model, solution.cost),
    Status::Unsat => println!("sin solución; certificado: {}", solution.proof_checked),
    Status::Unknown => println!("sin respuesta: {:?}", solution.halt),
}
```

`Options::on_event` recibe el avance y cada mejora de un `.opb`, desde otro hilo. `solve` no imprime
nada ni termina el proceso, y un fallo interno del motor vuelve como `Err`.

**Desde C, C++, Python u otro lenguaje con FFI**, con `motor_ffi.dll` o `libmotor_ffi.so`. El contrato
completo, con todas las opciones, está en
[`crates/motor-ffi/include/motor.h`](crates/motor-ffi/include/motor.h), y hay ejemplos en
[C](crates/motor-ffi/examples/ejemplo.c) y [Python](crates/motor-ffi/examples/ejemplo.py). Para compilar y
ejecutar el de C en Windows con `gcc`:

```powershell
cargo build --release -p motor-ffi
gcc -Icrates/motor-ffi/include crates/motor-ffi/examples/ejemplo.c target/release/motor_ffi.dll -o ejemplo.exe
Copy-Item target\release\motor_ffi.dll .
.\ejemplo.exe --certify
```

* En Windows, la DLL tiene que estar junto al ejecutable o en el `PATH`; si no, el programa no arranca.
* Con `gcc` hay que enlazar la DLL, como arriba. `-lmotor_ffi` no sirve: encuentra `motor_ffi.lib`, la
  biblioteca estática para MSVC, y falla al enlazar.
* Con MSVC se enlaza `motor_ffi.dll.lib`, o `motor_ffi.lib` para una estática junto con `kernel32.lib`,
  `ntdll.lib`, `userenv.lib`, `ws2_32.lib` y `dbghelp.lib`.
* En Linux: `gcc -Icrates/motor-ffi/include crates/motor-ffi/examples/ejemplo.c -Ltarget/release -lmotor_ffi -o ejemplo`
  y `LD_LIBRARY_PATH=target/release ./ejemplo`.
* El ejemplo de Python busca la biblioteca en `target/release`. En tu programa, pon en `ctypes.CDLL` la
  ruta donde la instales.

```c
motor_solver *solver = motor_new();
motor_set_option(solver, "timeout_ms", "600000");
const int32_t clause[] = {1, -2};
motor_add_clause(solver, clause, 2);        /* o motor_load_file, motor_load_dimacs, motor_load_opb */
int code = motor_solve(solver);             /* MOTOR_SAT, MOTOR_UNSAT, MOTOR_OPTIMUM, MOTOR_UNKNOWN o MOTOR_ERROR */
puts(motor_result_json(solver));
motor_free(solver);
```

`motor_stop` se puede llamar desde otro hilo, y `motor_on_event` da el avance en JSON.

**Como proceso aparte**, para no enlazar nada: `--json` deja en la salida estándar una sola línea JSON y
los códigos de salida son los de [Uso](#uso). Las líneas `c` van a la salida de errores. `status` es `SAT`,
`UNSAT`, `OPTIMUM` o `UNKNOWN`; si hay un error, la línea es `{"status":"ERROR","error":"..."}`.

```powershell
.\target\release\motor solve problema.cnf --json
```

```json
{"status":"UNSAT","model":null,"cost":null,"model_checked":false,"proof_checked":true,"reason":null,"detail":null,"proof":null,"elapsed_ms":6160.601,"stats":{"variables":153,"clauses":6120,"passes":1,"conflicts":26089,"decisions":33273,"restarts":25,"symmetry_generators":59,"parity_equations":0,"congruence_equivalences":0,"congruence_units":0,"cubes_generated":0,"cubes_unsat":0,"trace_bytes":6779869,"trace_truncated":false}}
```

`reason` es `timeout`, `cancelled` o `rejected`, y entonces `detail` explica el rechazo.

**Cuándo se atiende una parada.** En cada conflicto o decisión de la búsqueda y entre automorfismos al
detectar simetrías; una comprobación con VeriPB se corta matando el proceso. La lectura de la fórmula, su
preparación y la eliminación gaussiana no se interrumpen: con R(5,5) en K43 (casi dos millones de
cláusulas) y un límite de 2 s, el comando terminó a los 3,2 s.

**Qué necesita la máquina donde se use**

* La biblioteca o `motor.exe`, sin más ficheros: `motor solve` no descarga nada ni usa GPU.
* En Windows, el Visual C++ Redistributable (`VCRUNTIME140.dll`), salvo que ya esté instalado.
* VeriPB en el `PATH` para certificar, que es lo predeterminado. Sin él, la llamada falla con un mensaje que
  lo dice; con `certify` a 0 funciona sin él, pero un `Unsat` no queda comprobado.
* Disco en la carpeta temporal: la traza puede ocupar varios GiB y se corta a los 64 GiB. Al cortarse el
  resultado es `Unknown` con `reason` `rejected`; el tope se cambia con `proof_cap`.

**Dónde se ha probado.** En Windows 11 con Rust MSVC: una copia recién clonada, enlazada desde un proyecto
externo en Rust, por `path` y por `git` (con un repositorio local), desde C y C++ con `gcc`, desde Python
con `ctypes` y como proceso aparte, con VeriPB y sin él. **Sin probar:** Linux, macOS, el compilador `cl` de
MSVC y la biblioteca estática. Java, Go y demás deberían funcionar con su FFI hacia la interfaz en C, pero
no se han probado.

## Límites

Tiempos medidos en este equipo (16 hilos, RTX 4060 Laptop).

| Caso | Resultado |
|---|---|
| R(4,4) en K18 | UNSAT certificado en 6 s en serie, 15 s con `--workers 4` |
| R(3,6) en K18 | UNSAT certificado en 1,4 s |
| R(4,5) en K24 | SAT sin certificado: mediana de 13 s en 8 semillas, entre 1 s y 157 s |
| 3-SAT aleatorio en el umbral, 350 variables | UNSAT con `--workers 4`: 34 s sin certificado, 574 s con él (traza de 741 MB); medido antes de las fases objetivo |
| R(3,7) en K23 | No termina en 300 s; [`sat/`](sat/README.md) sí, pero su partición en clases es un argumento en papel |
| R(4,5) en K25 | No termina en 300 s |
| R(5,5) en K43 | Fuera de alcance |

**Frente a Kissat y CaDiCaL** ([tablas, método y sesgos](results/comparacion_solvers.md)):

| Instancias resueltas | Kissat | CaDiCaL | `motor` | `motor --workers 4` |
|---|---:|---:|---:|---:|
| SAT Competition 2023: muestra de 60, 120 s | 30 | 21 | 27 | 28 |
| De ellas, SAT | 12 | 13 | 8 | 7 |
| Familias con estructura: 25, 300 s | 21 | 19 | 25 | 25 |

* Gana donde hay simetría, paridad o circuitos equivalentes: palomar, Tseitin, Ramsey y varias instancias
  de la competición, a menudo en décimas de segundo. Con R(4,4) en K18 Kissat y CaDiCaL no terminan en 300 s.
* Pierde en lo general: Kissat resuelve más, y `motor` encuentra 8 SAT donde Kissat encuentra 12 y CaDiCaL
  13. Sin estructura que aprovechar puede no terminar en 120 s donde Kissat tarda 0,9 s.
* Con prueba, VeriPB aceptó los 35 UNSAT de `motor` en menos de 600 s; CaDiCaL y drat-trim certificaron 17
  y agotaron ese tiempo en los otros 18. En cambio, si la búsqueda es larga VeriPB es mucho más lento: 230 s
  donde CaDiCaL y drat-trim tardan 6 s, y 446 s donde tardan 31 s.
* Cuatro hilos resuelven solo una más que uno y no de forma estable: el 3-SAT aleatorio de 300 variables baja
  de 72 s a 5,5 s, pero R(4,5) en K24 sube de 20 s a 90 s y `29bab0de` pasa de 23 s a no terminar.

* **No resuelve cualquier problema.** Ningún programa puede: hay problemas matemáticos indecidibles, y
  muchos decidibles son exponenciales. `motor` trata problemas finitos sobre variables booleanas.
* **R(5,5) en K43:** las estimaciones de [`sat/README.md`](sat/README.md) dan al menos 10^13 s (unos
  500 000 años-CPU) solo para la clase 18 cortando a 60 aristas, y 10^44–10^59 s por clase cuando todas
  las hojas se resuelven. Además, el detector de simetrías solo encuentra 1 generador en esa fórmula.
* **Certificar cuesta más que buscar**, a veces más de 10 veces lo que tarda la búsqueda.
* **`--use-gpu`** solo refuta cubos que caen por propagación unitaria, y esos apenas le cuestan nada a la
  CPU. En las fórmulas probadas no refutó ninguna sonda de un literal: no acelera la búsqueda.
* **Sin hacer:** no hay entradas SMT ni de enteros, ni conexión con asistentes de prueba como Lean.

## Motor GPU para K43

`motor k43` es una búsqueda en profundidad en CUDA, fijada a 43 vértices y sin aprendizaje de cláusulas.
La red de `neuro_ramsey_policy.onnx`, servida con TensorRT, solo reordena las aristas: no descarta ninguna
rama, así que no afecta a la corrección del resultado. **No genera prueba:** un SAT se comprueba en CPU y
un UNSAT depende de la corrección del kernel. `build.rs` elige CUDA 12.6 con VS2022 o 13.x con VS2026.

```powershell
cargo build --release --features neuro-trt
.\target\release\motor k43 --self-test
```

`gpu_ramsey43` y `ramsey_bench` siguen existiendo como alias de `motor k43` y `motor bench`, porque el
lanzador y los comandos de `results/` los usan por su nombre. El asedio se lanza con:

```cmd
.\iniciar_neuro_ramsey_k43.bat
```

`--features neuro-mock` sustituye la red por un puente determinista para probar la GPU, y `--features
neuro-dataset` guarda muestras de la búsqueda con `--dataset`.

`motor bench` lleva esa red dentro del binario, así que funciona desde cualquier carpeta; `--model` solo
hace falta para probar otra.

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| [`src/solver.rs`](src/solver.rs) | La API: resuelve, comprueba y devuelve el resultado |
| [`src/bin/motor.rs`](src/bin/motor.rs) | El comando: traduce sus opciones a esa API e imprime |
| [`crates/motor-ffi`](crates/motor-ffi) | Interfaz en C, su cabecera y ejemplos en C y Python |
| `src/hybrid*.rs` | Solver: simetría, paridad, congruencia, CDCL, enjambre y trazas |
| [`src/verifier.rs`](src/verifier.rs) | Lanza VeriPB y lee su veredicto |
| [`src/opb.rs`](src/opb.rs) | Formato `.opb` y traductor a CNF |
| `src/k43.rs`, `src/ramsey43_*`, `src/*.cu*` | Motor GPU de K43 |
| [`src/ramsey_ref.rs`](src/ramsey_ref.rs), [`src/ramsey_bench.rs`](src/ramsey_bench.rs) | Referencia en CPU de la búsqueda GPU y su banco ([resultados](results/validacion_ramsey.md)) |
| `src/bin/certify.rs`, `src/bin/parity_stress.rs` | Herramientas de medición: escribir una traza sin comprobarla y forzar la capa de paridad |
| [`crates/chronos-vfs`](crates/chronos-vfs) | Copia de Chronos-VFS, en el commit `6bdbec0` de su repositorio, con sus licencias |
| [`sat/`](sat/README.md) | Prototipo con CaDiCaL y pruebas LRAT/DRAT, y estimación de coste para Ramsey |
| `centinela_k43.py`, `visualizador_fragmentos.py` | Vigilan el asedio de K43: auditan los modelos y muestran instantáneas del trail |
| `scripts/` | Lanzador de K43 y reproducción de las bases R(5,5,42) de McKay-Radziszowski |
| [`archivo/`](archivo/README.md) | Experimentos anteriores, que no se compilan |

## Pruebas

```powershell
cargo test --release --workspace
```

Con VeriPB en el `PATH`, añade los casos que pasan por el verificador. Comparan cada veredicto con la
fuerza bruta y exigen que se rechacen las pruebas falsas:

```powershell
cargo test --release --workspace -- --include-ignored
```

Con GPU y el SDK de TensorRT:

```powershell
cargo test --release --features neuro-mock,neuro-trt -- --include-ignored
```

## ✅ Validación: los 656 grafos Ramsey(5,5,42) conocidos no extienden a K43
Reproduce el resultado de McKay–Radziszowski con datos del propio problema. Cada grafo de
[`r55_42some.g6`](https://users.cecs.anu.edu.au/~bdm/data/ramsey.html) (328 grafos de B. McKay
y sus 328 complementos) ocupa los vértices 1..42 de K43 (arista = rojo, no arista = azul) y la
estrella del vértice 0 (42 aristas) queda libre:

```powershell
python -I scripts\r55_42.py fetch   # descarga 47 888 bytes y verifica SHA-256 en data\r55\
cargo build --release --locked --features neuro-trt --bin gpu_ramsey43 --target-dir target\neuro-trt
python -I scripts\r55_42.py all     # convert, audit, gpu, cpu, summary
```

* `audit` compila `scripts/r55_check_bases.rs`, que lee cada base con `parse_partial_dimacs` y exige
  `check_reference = Open` (sin K5 en la parte fija) con sólo la estrella del vértice 0 libre.
* `gpu` ejecuta `gpu_ramsey43 --base <g>.sat --lookahead --explicit-trail --epochs 0` por base.
* `cpu` es independiente del motor: reduce la extensión a 42 variables (color de cada arista de 0;
  ningún K4 rojo todo rojo, ningún K4 azul todo azul) y la resuelve con un DPLL propio. Como control
  positivo, el mismo DPLL debe extender el grafo sin el vértice 42 y el modelo se verifica sin K5.

Resultado (RTX 4060 Laptop, CUDA 13.2): **656/656 `s UNSATISFIABLE`** en GPU (25 min de motor,
mediana 2,0 s y 1,15 M transiciones por grafo; 75 min de pared con el arranque CUDA de cada proceso),
656/656 UNSAT en el DPLL de CPU (≤51 nodos por grafo) y 656/656 controles positivos SAT, sin
desacuerdos. Detalle en [`results/r55_42/summary.md`](results/r55_42/summary.md) y los CSV por grafo.
Los complementos son redundantes por simetría de colores; se ejecutan como comprobación de consistencia.
Esto valida el motor, no demuestra R(5,5)=43: puede haber grafos (5,5,42) no conocidos.

## Licencia

Todo el repositorio está bajo la [licencia Apache 2.0](LICENSE): código, documentación, resultados y la
red `neuro_ramsey_policy.onnx`. Permite usarlo, modificarlo, venderlo y redistribuirlo. A cambio pide
conservar la licencia y el fichero [`NOTICE`](NOTICE) e indicar qué ficheros se han cambiado. Incluye una
licencia de patentes, que termina para quien demande alegando que el proyecto infringe una patente; no
da derecho a usar el nombre del proyecto como marca, y se ofrece sin garantía de ningún tipo.

[`crates/chronos-vfs`](crates/chronos-vfs) mantiene su propia licencia, MIT o Apache 2.0. VeriPB, CaDiCaL,
drat-trim, CUDA, TensorRT y los grafos de B. McKay no forman parte del repositorio y tienen sus propias
condiciones: [`NOTICE`](NOTICE) detalla esto y las dependencias que entran en los binarios.
