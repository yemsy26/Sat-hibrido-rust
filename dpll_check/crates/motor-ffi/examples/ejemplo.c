/* Una formula con solucion, una sin ella y un problema de optimizacion, desde C.
 * Se pasa "--certify" para comprobar con VeriPB, que debe estar en el PATH.
 *
 *   cargo build --release -p motor-ffi
 *   gcc -Icrates/motor-ffi/include crates/motor-ffi/examples/ejemplo.c target/release/motor_ffi.dll -o ejemplo.exe
 *
 * La DLL tiene que estar junto al ejecutable o en el PATH. Linux y MSVC: ver el README. */
#include "motor.h"
#include <stdio.h>
#include <string.h>

static void on_event(const char *event_json, void *user) {
    (void)user;
    printf("  aviso: %s\n", event_json);
}

int main(int argc, char **argv) {
    const char *certify = argc > 1 && strcmp(argv[1], "--certify") == 0 ? "1" : "0";
    motor_solver *solver = motor_new();
    if (!solver) return 1;
    motor_set_option(solver, "certify", certify);
    motor_on_event(solver, on_event, NULL);

    const int32_t first[] = {1, 2}, second[] = {-1};
    motor_add_clause(solver, first, 2);
    motor_add_clause(solver, second, 1);
    int code = motor_solve(solver);
    int32_t model[2];
    size_t size = motor_model(solver, model, 2);
    printf("sat: codigo %d, modelo de %zu: %d %d\n", code, size, model[0], model[1]);
    if (code != MOTOR_SAT || model[0] != -1 || model[1] != 2) return 2;

    const int32_t third[] = {-2};
    motor_add_clause(solver, third, 1);
    code = motor_solve(solver);
    printf("unsat: codigo %d\n%s\n", code, motor_result_json(solver));
    if (code != MOTOR_UNSAT) return 3;

    motor_load_opb(solver, "min: +3 x1 +2 x2 +4 x3 ;\n+1 x1 +1 x2 +1 x3 >= 2 ;\n");
    code = motor_solve(solver);
    int64_t cost = 0;
    motor_cost(solver, &cost);
    printf("optimo: codigo %d, coste %lld\n", code, (long long)cost);
    if (code != MOTOR_OPTIMUM || cost != 5) return 4;

    if (motor_set_option(solver, "nada", "1") != MOTOR_ERROR) return 5;
    printf("error: %s\n", motor_last_error(solver));
    motor_free(solver);
    return 0;
}
