// La interfaz en C tal como la ve C: punteros, codigos y cadenas.
use motor_ffi::*;
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::atomic::{AtomicUsize, Ordering};

const SAT: i32 = 10;
const UNSAT: i32 = 20;
const OPTIMUM: i32 = 30;

fn set(solver: *mut motor_solver, name: &str, value: &str) {
    let (name, value) = (CString::new(name).unwrap(), CString::new(value).unwrap());
    assert_eq!(unsafe { motor_set_option(solver, name.as_ptr(), value.as_ptr()) }, 0);
}

fn json(solver: *mut motor_solver) -> String {
    unsafe { CStr::from_ptr(motor_result_json(solver)) }.to_str().unwrap().to_owned()
}

fn unchecked() -> *mut motor_solver {
    let solver = motor_new();
    assert!(!solver.is_null());
    set(solver, "certify", "0");
    solver
}

#[test]
fn clauses_are_solved_and_the_model_is_copied_out() {
    let solver = unchecked();
    unsafe {
        assert_eq!(motor_add_clause(solver, [1, 2].as_ptr(), 2), 0);
        assert_eq!(motor_add_clause(solver, [-1].as_ptr(), 1), 0);
        assert_eq!(motor_solve(solver), SAT);
        assert_eq!(motor_model(solver, std::ptr::null_mut(), 0), 2);
        let mut model = [0_i32; 2];
        assert_eq!(motor_model(solver, model.as_mut_ptr(), 2), 2);
        assert_eq!(model, [-1, 2]);
        assert!(json(solver).contains("\"status\":\"SAT\""));
        // Se sigue pudiendo anadir y volver a resolver.
        assert_eq!(motor_add_clause(solver, [-2].as_ptr(), 1), 0);
        assert_eq!(motor_solve(solver), UNSAT);
        assert_eq!(motor_model(solver, std::ptr::null_mut(), 0), 0);
        assert!(json(solver).contains("\"proof_checked\":false"));
        motor_free(solver);
    }
}

#[test]
fn an_objective_gives_the_optimum_and_its_cost() {
    let solver = unchecked();
    let source = CString::new("min: +3 x1 +2 x2 +4 x3 ;\n+1 x1 +1 x2 +1 x3 >= 2 ;\n").unwrap();
    let mut cost = 0_i64;
    unsafe {
        assert_eq!(motor_load_opb(solver, source.as_ptr()), 0);
        assert_eq!(motor_solve(solver), OPTIMUM);
        assert_eq!(motor_cost(solver, &mut cost), 1);
        assert_eq!(cost, 5);
        motor_free(solver);
    }
}

#[test]
fn bad_calls_leave_a_message_and_do_not_crash() {
    let solver = motor_new();
    let (bad, value) = (CString::new("nada").unwrap(), CString::new("1").unwrap());
    let last_error = || unsafe { CStr::from_ptr(motor_last_error(solver)) }.to_str().unwrap().to_owned();
    unsafe {
        assert_eq!(motor_set_option(solver, bad.as_ptr(), value.as_ptr()), -1);
        assert!(last_error().contains("nada"));
        assert_eq!(motor_add_clause(solver, [0].as_ptr(), 1), -1);
        assert_eq!(motor_load_dimacs(solver, CString::new("1 x 0").unwrap().as_ptr()), -1);
        assert_eq!(motor_load_file(solver, CString::new("no-existe.cnf").unwrap().as_ptr()), -1);
        assert_eq!(motor_load_file(solver, std::ptr::null()), -1);
        assert_eq!(motor_solve(std::ptr::null_mut()), -1);
        motor_stop(std::ptr::null_mut());
        motor_free(std::ptr::null_mut());
        assert_eq!(motor_model(std::ptr::null_mut(), std::ptr::null_mut(), 0), 0);
        assert!(CStr::from_ptr(motor_result_json(std::ptr::null_mut())).to_bytes().is_empty());
        motor_free(solver);
    }
}

// Palomar sin simetria ni paridad: no termina a tiempo.
fn pigeonhole(solver: *mut motor_solver, holes: i32) {
    let variable = |pigeon: i32, hole: i32| pigeon * holes + hole + 1;
    for pigeon in 0..=holes {
        let clause: Vec<i32> = (0..holes).map(|hole| variable(pigeon, hole)).collect();
        assert_eq!(unsafe { motor_add_clause(solver, clause.as_ptr(), clause.len()) }, 0);
    }
    for hole in 0..holes {
        for first in 0..=holes {
            for second in first + 1..=holes {
                let clause = [-variable(first, hole), -variable(second, hole)];
                assert_eq!(unsafe { motor_add_clause(solver, clause.as_ptr(), 2) }, 0);
            }
        }
    }
    set(solver, "symmetry", "0");
    set(solver, "parity", "0");
}

static EVENTS: AtomicUsize = AtomicUsize::new(0);
unsafe extern "C" fn count_event(event: *const c_char, user: *mut c_void) {
    assert!(unsafe { CStr::from_ptr(event) }.to_bytes().starts_with(b"{\"event\":"));
    assert_eq!(user as usize, 7);
    EVENTS.fetch_add(1, Ordering::Relaxed);
}

#[test]
fn a_time_limit_is_never_an_unsat_and_events_arrive() {
    let solver = unchecked();
    pigeonhole(solver, 15);
    set(solver, "timeout_ms", "300");
    set(solver, "progress_ms", "50");
    unsafe {
        assert_eq!(motor_on_event(solver, Some(count_event), 7 as *mut c_void), 0);
        assert_eq!(motor_solve(solver), 0);
        assert!(json(solver).contains("\"reason\":\"timeout\""), "{}", json(solver));
        motor_free(solver);
    }
    assert!(EVENTS.load(Ordering::Relaxed) > 0);
}

#[test]
fn stop_works_from_another_thread_while_solve_is_running() {
    let solver = unchecked();
    pigeonhole(solver, 15);
    set(solver, "workers", "2");
    let address = solver as usize;
    let stopper = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(300));
        unsafe { motor_stop(address as *mut motor_solver) };
    });
    unsafe {
        assert_eq!(motor_solve(solver), 0);
        stopper.join().unwrap();
        assert!(json(solver).contains("\"reason\":\"cancelled\""), "{}", json(solver));
        // La parada no se queda pegada a la siguiente resolucion.
        set(solver, "timeout_ms", "100");
        assert_eq!(motor_solve(solver), 0);
        assert!(json(solver).contains("\"reason\":\"timeout\""), "{}", json(solver));
        motor_free(solver);
    }
}

#[test]
#[ignore = "requiere veripb"]
fn a_certified_unsat_says_so() {
    let solver = motor_new();
    unsafe {
        for clause in [[1, 2], [-1, 2], [1, -2], [-1, -2]] {
            assert_eq!(motor_add_clause(solver, clause.as_ptr(), 2), 0);
        }
        assert_eq!(motor_solve(solver), UNSAT);
        assert!(json(solver).contains("\"proof_checked\":true"), "{}", json(solver));
        motor_free(solver);
    }
}
