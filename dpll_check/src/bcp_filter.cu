// Mejora 3, fase 2: filtro de cubos por propagacion unitaria. Un hilo por cubo.
//
// gpu_dispatcher.rs incrusta este fichero y lo compila con nvrtc al abrir la tarjeta. El
// nucleo no decide por busqueda: carga el cubo, repite pasadas densas sobre todas las
// clausulas y se queda con lo que la propagacion unitaria sola demuestra.
//
// Un literal es 2 * (variable - 1), mas 1 si es negativo. El estado de un cubo son dos bits
// por variable (0 sin asignar, 1 cierta, 2 falsa) en su fila de `assignment`. Cada hilo
// escribe solo en su fila y en su veredicto.
//
// La propagacion unitaria da el mismo resultado en cualquier orden: los veredictos no
// dependen de como se repartan los hilos ni de cuantas pasadas quepan en un lanzamiento.

#define UNKNOWN 0u
#define UNSAT 1u
#define SAT 2u
// Sin terminar: se agotaron las pasadas de este lanzamiento. El estado queda en VRAM y el
// siguiente lanzamiento, con fresh = 0, sigue donde se quedo.
#define PENDING 3u

// Valor de un literal: 0 sin asignar, 1 cierto, 2 falso.
__device__ __forceinline__ unsigned value_of(const unsigned* state, unsigned literal) {
    unsigned variable = literal >> 1;
    unsigned value = (state[variable >> 4] >> ((variable & 15u) << 1)) & 3u;
    // Un literal negativo cambia cierto por falso.
    return (value != 0u && (literal & 1u)) ? 3u - value : value;
}

// Hace cierto un literal cuya variable esta sin asignar.
__device__ __forceinline__ void make_true(unsigned* state, unsigned literal) {
    unsigned variable = literal >> 1;
    state[variable >> 4] |= (1u + (literal & 1u)) << ((variable & 15u) << 1);
}

// Los cubos first .. first + count - 1 del lote, uno por hilo, con `rounds` pasadas como
// mucho. `words` son las palabras de 32 bits del estado de un cubo. `phase` son las pasadas
// que ya llevan estos cubos: las pares recorren las clausulas hacia delante y las impares
// hacia atras, para que una cadena de implicaciones escrita al reves no cueste una pasada
// por eslabon.
extern "C" __global__ void bcp_filter(
    const unsigned* clause_starts, const unsigned* clause_literals, unsigned clauses,
    const unsigned* cube_starts, const unsigned* cube_literals,
    unsigned* assignment, unsigned words, unsigned* verdicts,
    unsigned first, unsigned count, unsigned fresh, unsigned rounds, unsigned phase)
{
    unsigned thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread >= count) return;
    unsigned cube = first + thread;
    unsigned* state = assignment + (unsigned long long)cube * words;
    if (fresh) {
        for (unsigned word = 0; word < words; ++word) state[word] = 0u;
        // a. El cubo: sus literales son ciertos. Uno que contradice a otro lo refuta.
        for (unsigned index = cube_starts[cube]; index < cube_starts[cube + 1]; ++index) {
            unsigned literal = cube_literals[index];
            unsigned value = value_of(state, literal);
            if (value == 2u) { verdicts[cube] = UNSAT; return; }
            if (value == 0u) make_true(state, literal);
        }
    } else if (verdicts[cube] != PENDING) {
        return;
    }
    // b. Pasadas densas hasta que una no cambie nada.
    for (unsigned round = 0; round < rounds; ++round) {
        bool changed = false;
        // Alguna clausula sigue abierta: dos literales libres antes de ninguno cierto.
        bool open = false;
        bool backwards = ((round + phase) & 1u) != 0u;
        for (unsigned step = 0; step < clauses; ++step) {
            unsigned clause = backwards ? clauses - 1u - step : step;
            unsigned end = clause_starts[clause + 1];
            unsigned free_literal = 0u;
            unsigned free_count = 0u;
            bool satisfied = false;
            for (unsigned index = clause_starts[clause]; index < end; ++index) {
                unsigned literal = clause_literals[index];
                unsigned value = value_of(state, literal);
                if (value == 1u) { satisfied = true; break; }
                if (value == 0u) {
                    // El mismo literal repetido cuenta una vez; con dos distintos la
                    // clausula ni propaga ni falla, y no hace falta leer mas.
                    if (free_count == 0u) { free_literal = literal; free_count = 1u; }
                    else if (literal != free_literal) { free_count = 2u; break; }
                }
            }
            if (satisfied) continue;
            // Todos los literales falsos: el cubo es insatisfacible.
            if (free_count == 0u) { verdicts[cube] = UNSAT; return; }
            // Todos falsos menos uno: ese es cierto.
            if (free_count == 1u) { make_true(state, free_literal); changed = true; }
            else open = true;
        }
        // c. Punto fijo sin contradiccion. Si ademas no queda ninguna clausula abierta,
        //    todas tienen un literal cierto y el estado es un modelo.
        if (!changed) { verdicts[cube] = open ? UNKNOWN : SAT; return; }
    }
    verdicts[cube] = PENDING;
}
