/* Interfaz en C del motor. Enlaza con motor_ffi (cargo build --release -p motor-ffi); la orden para
 * cada compilador esta en el README, seccion "Integrarlo en otro programa".
 *
 * Uso: motor_new, cargar el problema, motor_set_option, motor_solve, leer el resultado y
 * motor_free. Un mismo motor se puede resolver varias veces. Salvo motor_stop, las funciones
 * de un mismo motor se esperan entre si: no se pueden llamar desde el aviso de motor_on_event.
 *
 * Garantia: MOTOR_SAT y MOTOR_OPTIMUM traen un modelo evaluado contra el problema. MOTOR_UNSAT y
 * MOTOR_OPTIMUM los ha aceptado VeriPB, salvo con la opcion "certify" a 0. Una parada o una prueba
 * rechazada nunca devuelven MOTOR_UNSAT. En un OPB, VeriPB comprueba la CNF traducida.
 */
#ifndef MOTOR_H
#define MOTOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MOTOR_API_VERSION 1

/* Resultado de motor_solve. Las demas funciones devuelven MOTOR_OK o MOTOR_ERROR. */
#define MOTOR_ERROR (-1)
#define MOTOR_OK 0
#define MOTOR_UNKNOWN 0 /* Sin respuesta: el motivo esta en "reason" de motor_result_json. */
#define MOTOR_SAT 10    /* En un OPB con objetivo y "reason" no nulo: mejor solucion, sin probar que sea optima. */
#define MOTOR_UNSAT 20
#define MOTOR_OPTIMUM 30

typedef struct motor_solver motor_solver;

/* event_json: {"event":"progress",...} o {"event":"improved","cost":N}. Vale durante la llamada,
 * que llega desde otro hilo; mientras no vuelve, no se atiende el limite de tiempo. */
typedef void (*motor_event_fn)(const char *event_json, void *user);

int motor_api_version(void);

/* Devuelve NULL solo si falla la reserva. */
motor_solver *motor_new(void);
void motor_free(motor_solver *solver);

/* Clausula con literales DIMACS: 3 es la variable 3 y -3 su negacion; sin el 0 final. */
int motor_add_clause(motor_solver *solver, const int32_t *literals, size_t count);
/* Sustituyen el problema: un fichero (OPB si termina en .opb, DIMACS si no) o un texto. */
int motor_load_file(motor_solver *solver, const char *path);
int motor_load_dimacs(motor_solver *solver, const char *text);
int motor_load_opb(motor_solver *solver, const char *text);

/* Opciones, siempre como texto:
 *   certify 0|1 (1)     comprobar con VeriPB; sin VeriPB, motor_solve falla antes de buscar
 *   timeout_ms N        0 sin limite
 *   workers N           0 un hilo; N, enjambre de N hilos
 *   cubes N             cubos por hilo
 *   share 0|1           compartir clausulas entre hilos
 *   symmetry 0|1        ruptura de simetrias
 *   parity 0|1          capa de paridad
 *   congruence 0|1      equivalencias y cierre por congruencia
 *   flips N             flips por variable del oraculo
 *   interval N          conflictos entre re-igniciones; 0 apaga el oraculo
 *   seed N              semilla del oraculo
 *   proof F             conserva la traza; con .zst, comprimida
 *   cnf F               en un OPB, conserva la ultima CNF traducida
 *   proof_cap_gib N     tope de la traza
 *   progress_ms N       cada cuanto llega el avance a motor_on_event; 0 solo al parar
 */
int motor_set_option(motor_solver *solver, const char *name, const char *value);
/* callback NULL deja de avisar. */
int motor_on_event(motor_solver *solver, motor_event_fn callback, void *user);

/* Bloquea hasta terminar. Devuelve MOTOR_SAT, MOTOR_UNSAT, MOTOR_OPTIMUM, MOTOR_UNKNOWN o MOTOR_ERROR. */
int motor_solve(motor_solver *solver);
/* Se puede llamar desde cualquier hilo. Sin resolucion en marcha, afecta a la siguiente, que aun
 * asi devuelve su respuesta si la encuentra antes de ver la parada. */
void motor_stop(motor_solver *solver);

/* Copia hasta capacity literales, uno por variable, y devuelve cuantos tiene el modelo. Con
 * capacity 0 solo pregunta el tamano. Sin modelo devuelve 0. */
size_t motor_model(motor_solver *solver, int32_t *out, size_t capacity);
/* 1 si hay coste y cabe en 64 bits; 0 si no. */
int motor_cost(motor_solver *solver, int64_t *out);
/* Las dos cadenas valen hasta la siguiente llamada sobre el motor. Nunca devuelven NULL. */
const char *motor_result_json(motor_solver *solver);
const char *motor_last_error(motor_solver *solver);

#ifdef __cplusplus
}
#endif
#endif
