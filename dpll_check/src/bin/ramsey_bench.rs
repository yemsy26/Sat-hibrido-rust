//! Alias de `motor bench`.
fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    motor_core::ramsey_bench::run(&arguments);
}
