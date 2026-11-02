// Interfaz en C de `motor_core::solver`; el contrato esta en include/motor.h. El `unsafe` de esta
// frontera vive aqui y el nucleo sigue sin el. Ningun panico cruza hacia C.
#![allow(non_camel_case_types, clippy::missing_safety_doc)]

use motor_core::dimacs::Formula;
use motor_core::solver::{self, Options, Problem, Solution, Status, StopToken};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

const MOTOR_API_VERSION: c_int = 1;
const MOTOR_ERROR: c_int = -1;
const MOTOR_OK: c_int = 0;

pub type motor_event_fn = Option<unsafe extern "C" fn(*const c_char, *mut c_void)>;

struct State {
    clauses: Formula,
    loaded: Option<Problem>,
    options: Options,
    solution: Option<Solution>,
    json: CString,
    error: CString,
}

pub struct motor_solver {
    // Fuera del candado: motor_stop tiene que poder entrar mientras motor_solve lo sostiene.
    stop: StopToken,
    state: Mutex<State>,
}

// C responde de que el puntero y su aviso sirvan desde otro hilo: asi lo dice motor.h.
struct Callback(unsafe extern "C" fn(*const c_char, *mut c_void), *mut c_void);
unsafe impl Send for Callback {}
unsafe impl Sync for Callback {}

fn c_string(text: &str) -> CString {
    CString::new(text.replace('\0', " ")).expect("sin ceros")
}

fn guarded<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or(fallback)
}

fn lock(solver: &motor_solver) -> MutexGuard<'_, State> {
    solver.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

// Entrada comun: comprueba el puntero, toma el estado y guarda el error de `body`.
unsafe fn with_state(solver: *mut motor_solver, body: impl FnOnce(&mut State) -> Result<c_int, String>) -> c_int {
    guarded(MOTOR_ERROR, || {
        let Some(solver) = (unsafe { solver.as_ref() }) else { return MOTOR_ERROR };
        let mut state = lock(solver);
        state.error = CString::default();
        match body(&mut state) {
            Ok(code) => code,
            Err(error) => {
                state.error = c_string(&error);
                MOTOR_ERROR
            }
        }
    })
}

unsafe fn text<'a>(pointer: *const c_char) -> Result<&'a str, String> {
    if pointer.is_null() {
        return Err("Puntero nulo".into());
    }
    unsafe { CStr::from_ptr(pointer) }.to_str().map_err(|_| "El texto no es UTF-8".to_owned())
}

#[no_mangle]
pub extern "C" fn motor_api_version() -> c_int {
    MOTOR_API_VERSION
}

#[no_mangle]
pub extern "C" fn motor_new() -> *mut motor_solver {
    guarded(std::ptr::null_mut(), || {
        Box::into_raw(Box::new(motor_solver {
            stop: StopToken::default(),
            state: Mutex::new(State {
                clauses: Formula::new(), loaded: None, options: Options::default(), solution: None,
                json: c_string("null"), error: CString::default(),
            }),
        }))
    })
}

#[no_mangle]
pub unsafe extern "C" fn motor_free(solver: *mut motor_solver) {
    if !solver.is_null() {
        guarded((), || drop(unsafe { Box::from_raw(solver) }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn motor_add_clause(solver: *mut motor_solver, literals: *const i32, count: usize) -> c_int {
    unsafe {
        with_state(solver, |state| {
            if state.loaded.is_some() {
                return Err("Ya hay un problema cargado: motor_add_clause no se puede mezclar con motor_load_*".into());
            }
            if literals.is_null() && count != 0 {
                return Err("Puntero nulo".into());
            }
            let clause = if count == 0 { &[][..] } else { std::slice::from_raw_parts(literals, count) };
            if let Some(&bad) = clause.iter().find(|&&literal| literal == 0 || literal == i32::MIN) {
                return Err(format!("Literal no valido: {bad}"));
            }
            state.clauses.push(clause.to_vec());
            Ok(MOTOR_OK)
        })
    }
}

fn load(state: &mut State, problem: Result<Problem, solver::SolveError>) -> Result<c_int, String> {
    state.loaded = Some(problem.map_err(|error| error.to_string())?);
    state.clauses.clear();
    Ok(MOTOR_OK)
}

#[no_mangle]
pub unsafe extern "C" fn motor_load_file(solver: *mut motor_solver, path: *const c_char) -> c_int {
    unsafe { with_state(solver, |state| load(state, Problem::from_file(text(path)?))) }
}

#[no_mangle]
pub unsafe extern "C" fn motor_load_dimacs(solver: *mut motor_solver, source: *const c_char) -> c_int {
    unsafe { with_state(solver, |state| load(state, Problem::dimacs(text(source)?))) }
}

#[no_mangle]
pub unsafe extern "C" fn motor_load_opb(solver: *mut motor_solver, source: *const c_char) -> c_int {
    unsafe { with_state(solver, |state| load(state, Problem::opb(text(source)?))) }
}

fn set_option(options: &mut Options, name: &str, value: &str) -> Result<(), String> {
    let number = || value.parse::<u64>().map_err(|_| format!("{name}: numero no valido: {value}"));
    let switch = || match value {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(format!("{name}: vale 0 o 1, no {value}")),
    };
    match name {
        "certify" => options.certify = switch()?,
        "timeout_ms" => options.timeout = Some(number()?).filter(|&limit| limit != 0).map(Duration::from_millis),
        "workers" => options.workers = Some(number()? as usize).filter(|&workers| workers != 0),
        "cubes" => options.cubes = number()?.max(1) as usize,
        "share" => options.share = switch()?,
        "symmetry" => options.symmetry = switch()?,
        "parity" => options.parity = switch()?,
        "congruence" => options.congruence = switch()?,
        "flips" => options.flips = number()? as usize,
        "interval" => options.interval = number()? as usize,
        "seed" => options.seed = Some(number()?),
        "proof" => options.proof = Some(PathBuf::from(value)).filter(|_| !value.is_empty()),
        "cnf" => options.cnf = Some(PathBuf::from(value)).filter(|_| !value.is_empty()),
        "proof_cap_gib" => options.proof_cap = Some(number()?.saturating_mul(1 << 30)),
        "progress_ms" => options.progress_interval = Some(number()?).filter(|&every| every != 0).map(Duration::from_millis),
        _ => return Err(format!("Opcion desconocida: {name}")),
    }
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn motor_set_option(solver: *mut motor_solver, name: *const c_char, value: *const c_char) -> c_int {
    unsafe {
        with_state(solver, |state| {
            set_option(&mut state.options, text(name)?, text(value)?)?;
            Ok(MOTOR_OK)
        })
    }
}

#[no_mangle]
pub unsafe extern "C" fn motor_on_event(solver: *mut motor_solver, callback: motor_event_fn, user: *mut c_void) -> c_int {
    unsafe {
        with_state(solver, |state| {
            state.options.on_event = callback.map(|function| {
                let callback = Callback(function, user);
                let handler: solver::EventHandler = Arc::new(move |event: &solver::Event| {
                    let json = c_string(&event.to_json());
                    let Callback(function, user) = &callback;
                    function(json.as_ptr(), *user);
                });
                handler
            });
            Ok(MOTOR_OK)
        })
    }
}

#[no_mangle]
pub unsafe extern "C" fn motor_solve(solver: *mut motor_solver) -> c_int {
    let Some(token) = (unsafe { solver.as_ref() }).map(|solver| solver.stop.clone()) else { return MOTOR_ERROR };
    let code = unsafe {
        with_state(solver, |state| {
            state.solution = None;
            state.json = c_string("null");
            let owned;
            let problem = match &state.loaded {
                Some(problem) => problem,
                None => {
                    owned = Problem::cnf(state.clauses.clone()).map_err(|error| error.to_string())?;
                    &owned
                }
            };
            let options = Options { stop: Some(token.clone()), ..state.options.clone() };
            let solution = solver::solve(problem, &options).map_err(|error| error.to_string())?;
            let code = match solution.status {
                Status::Sat => 10,
                Status::Unsat => 20,
                Status::Optimum => 30,
                Status::Unknown => 0,
            };
            state.json = c_string(&solution.to_json());
            state.solution = Some(solution);
            Ok(code)
        })
    };
    token.clear();
    code
}

#[no_mangle]
pub unsafe extern "C" fn motor_stop(solver: *mut motor_solver) {
    if let Some(solver) = unsafe { solver.as_ref() } {
        solver.stop.stop();
    }
}

#[no_mangle]
pub unsafe extern "C" fn motor_model(solver: *mut motor_solver, out: *mut i32, capacity: usize) -> usize {
    guarded(0, || {
        let Some(solver) = (unsafe { solver.as_ref() }) else { return 0 };
        let state = lock(solver);
        let Some(model) = state.solution.as_ref().and_then(|solution| solution.model.as_ref()) else { return 0 };
        if !out.is_null() {
            unsafe { std::ptr::copy_nonoverlapping(model.as_ptr(), out, model.len().min(capacity)) };
        }
        model.len()
    })
}

#[no_mangle]
pub unsafe extern "C" fn motor_cost(solver: *mut motor_solver, out: *mut i64) -> c_int {
    guarded(0, || {
        let Some(solver) = (unsafe { solver.as_ref() }) else { return 0 };
        let state = lock(solver);
        let cost = state.solution.as_ref().and_then(|solution| solution.cost).and_then(|cost| i64::try_from(cost).ok());
        match cost {
            Some(cost) if !out.is_null() => {
                unsafe { *out = cost };
                1
            }
            _ => 0,
        }
    })
}

const EMPTY: &CStr = c"";

#[no_mangle]
pub unsafe extern "C" fn motor_result_json(solver: *mut motor_solver) -> *const c_char {
    guarded(EMPTY.as_ptr(), || {
        let Some(solver) = (unsafe { solver.as_ref() }) else { return EMPTY.as_ptr() };
        lock(solver).json.as_ptr()
    })
}

#[no_mangle]
pub unsafe extern "C" fn motor_last_error(solver: *mut motor_solver) -> *const c_char {
    guarded(EMPTY.as_ptr(), || {
        let Some(solver) = (unsafe { solver.as_ref() }) else { return EMPTY.as_ptr() };
        lock(solver).error.as_ptr()
    })
}
