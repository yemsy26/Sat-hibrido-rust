// Nucleo del comando `motor`: lectura DIMACS, solver CDCL hibrido, generadores de formulas,
// comprobacion externa de las pruebas y motores de Ramsey. Para integrarlo en otro programa,
// `solver` es la API; lo demas es interno. `deny` y no `forbid`: gpu_dispatcher.rs permite
// `unsafe` en una sola funcion, la que lanza el nucleo CUDA.
#![deny(unsafe_code)]

pub mod dimacs;
pub mod gen_erdos;
pub mod gen_ramsey;
#[allow(dead_code)]
pub mod gpu_dispatcher;
#[path = "hybrid.rs"]
pub mod hybrid;
#[cfg(feature = "gpu")]
pub mod k43;
#[cfg(feature = "gpu")]
#[path = "ramsey43_gpu.rs"]
pub mod ramsey43_gpu;
pub mod opb;
pub mod ramsey_bench;
pub mod ramsey_ref;
pub mod ramsey_symmetry;
pub mod solver;
pub mod symmetry_breaker;
pub mod trace_bridge;
pub mod verifier;
