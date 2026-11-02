//! Alias de `motor k43`: el lanzador del asedio y scripts/r55_42.py detectan el proceso por este nombre.
fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(motor_core::k43::run(&arguments));
}
