// K43 experimental: cada warp mantiene las 86 filas de dos colores en registros.
// Las aristas desconocidas no pertenecen a ningun color: azul nunca es ~rojo.
// BAD es una refutacion por un K5; OPEN no demuestra extensibilidad del fragmento.
// COMPLETE_SAFE solo se emite tras comprobar todas las 903 aristas y ambos colores.
// Lanzar exactamente 256 hilos por bloque, con blockDim.y = blockDim.z = 1.

typedef unsigned long long Ramsey43Bits;
static const unsigned RAMSEY43_FULL_WARP = 0xffffffffu;
static const Ramsey43Bits RAMSEY43_VERTICES = (1ull << 43) - 1ull;
static const Ramsey43Bits RAMSEY43_WITNESS_PRESENT = 1ull << 63;
static const unsigned RAMSEY43_OPEN = 0u;
static const unsigned RAMSEY43_BAD = 1u;
static const unsigned RAMSEY43_COMPLETE_SAFE = 2u;

struct Ramsey43WarpGraph {
    Ramsey43Bits red0;
    Ramsey43Bits blue0;
    Ramsey43Bits red1;
    Ramsey43Bits blue1;
};

// Todas las llamadas se ejecutan con 32 lanes activos. El indice puede variar
// por lane; se intercambian siempre ambas bancas antes de seleccionar la fila.
__device__ __forceinline__ Ramsey43Bits ramsey43_row(
    Ramsey43Bits row0, Ramsey43Bits row1, unsigned vertex)
{
    unsigned source = vertex & 31u;
    Ramsey43Bits low = __shfl_sync(RAMSEY43_FULL_WARP, row0, source);
    Ramsey43Bits high = __shfl_sync(RAMSEY43_FULL_WARP, row1, source);
    return vertex < 32u ? low : high;
}

__device__ __forceinline__ Ramsey43Bits ramsey43_after(unsigned vertex)
{
    return RAMSEY43_VERTICES & ~((1ull << (vertex + 1u)) - 1ull);
}

// Testigo: bit63 = presente; vertices en bits 0..29 (cinco campos de 6 bits).
// El llamador añade bit30 = color (0 rojo, 1 azul). No hay arrays por hilo.
__device__ __forceinline__ Ramsey43Bits ramsey43_find_k5(
    Ramsey43Bits row0, Ramsey43Bits row1)
{
    unsigned lane = threadIdx.x & 31u;
    for (unsigned a = 0; a < 39u; ++a) {
        Ramsey43Bits neighbors_a = ramsey43_row(row0, row1, a);
        // Distribucion de b entre lanes; el segundo paso cubre los vertices >32.
        for (unsigned pass = 0; pass < 2u; ++pass) {
            unsigned b = a + 1u + lane + 32u * pass;
            bool valid_b = b < 40u;
            unsigned safe_b = valid_b ? b : 0u;
            Ramsey43Bits neighbors_b = ramsey43_row(row0, row1, safe_b);
            Ramsey43Bits candidates_c = valid_b && ((neighbors_a >> safe_b) & 1ull)
                ? neighbors_a & neighbors_b & ramsey43_after(safe_b) : 0ull;
            while (__any_sync(RAMSEY43_FULL_WARP, candidates_c != 0ull)) {
                bool valid_c = candidates_c != 0ull;
                unsigned c = valid_c ? (unsigned)(__ffsll((long long)candidates_c) - 1) : 0u;
                Ramsey43Bits neighbors_c = ramsey43_row(row0, row1, c);
                Ramsey43Bits common_abc = neighbors_a & neighbors_b & neighbors_c;
                Ramsey43Bits candidates_d = valid_c ? common_abc & ramsey43_after(c) : 0ull;
                while (__any_sync(RAMSEY43_FULL_WARP, candidates_d != 0ull)) {
                    bool valid_d = candidates_d != 0ull;
                    unsigned d = valid_d ? (unsigned)(__ffsll((long long)candidates_d) - 1) : 0u;
                    Ramsey43Bits neighbors_d = ramsey43_row(row0, row1, d);
                    Ramsey43Bits candidates_e = valid_d
                        ? common_abc & neighbors_d & ramsey43_after(d) : 0ull;
                    unsigned winners = __ballot_sync(RAMSEY43_FULL_WARP, candidates_e != 0ull);
                    if (winners != 0u) {
                        unsigned e = candidates_e != 0ull
                            ? (unsigned)(__ffsll((long long)candidates_e) - 1) : 0u;
                        Ramsey43Bits witness = RAMSEY43_WITNESS_PRESENT
                            | (Ramsey43Bits)a | ((Ramsey43Bits)safe_b << 6)
                            | ((Ramsey43Bits)c << 12) | ((Ramsey43Bits)d << 18)
                            | ((Ramsey43Bits)e << 24);
                        // Eleccion determinista del primer lane con un K5.
                        return __shfl_sync(RAMSEY43_FULL_WARP, witness, __ffs(winners) - 1);
                    }
                    candidates_d &= candidates_d - 1ull;
                }
                candidates_c &= candidates_c - 1ull;
            }
        }
    }
    return 0ull;
}

// Los dos colores y los 43 vertices estan repartidos: lane i posee fila i,
// y los primeros 11 lanes poseen ademas fila i+32. No se lee memoria global aqui.
__device__ __forceinline__ bool ramsey43_complete(Ramsey43WarpGraph graph)
{
    unsigned lane = threadIdx.x & 31u;
    Ramsey43Bits expected0 = RAMSEY43_VERTICES & ~(1ull << lane);
    bool complete0 = (graph.red0 | graph.blue0) == expected0
        && (graph.red0 & graph.blue0) == 0ull;
    bool complete1 = true;
    if (lane < 11u) {
        Ramsey43Bits expected1 = RAMSEY43_VERTICES & ~(1ull << (lane + 32u));
        complete1 = (graph.red1 | graph.blue1) == expected1
            && (graph.red1 & graph.blue1) == 0ull;
    }
    return __all_sync(RAMSEY43_FULL_WARP, complete0 && complete1);
}

// Entrada contigua: graphs[universo][color: 0 rojo/1 azul][vertice: 0..42].
// Contrato de entrada: filas simetricas, diagonal vacia y colores disjuntos;
// bits 43..63 a cero. El host debe validar ese contrato una vez por grafo base.
// witnesses puede ser nullptr. No escribir BAD como UNSAT de la formula global.
extern "C" __global__ void ramsey43_check(
    const Ramsey43Bits* graphs, unsigned count,
    unsigned* verdicts, Ramsey43Bits* witnesses)
{
    if (blockDim.x != 256u || blockDim.y != 1u || blockDim.z != 1u) return;
    unsigned lane = threadIdx.x & 31u;
    unsigned warp = threadIdx.x >> 5;
    unsigned universe = blockIdx.x * 8u + warp;
    if (universe >= count) return; // Condicion uniforme para el warp completo.
    __shared__ unsigned dead[8];
    if (lane == 0u) dead[warp] = 0u;
    __syncwarp(RAMSEY43_FULL_WARP);

    const Ramsey43Bits* rows = graphs + (Ramsey43Bits)universe * 86ull;
    Ramsey43WarpGraph graph;
    graph.red0 = rows[lane];
    graph.blue0 = rows[43u + lane];
    graph.red1 = lane < 11u ? rows[lane + 32u] : 0ull;
    graph.blue1 = lane < 11u ? rows[75u + lane] : 0ull;

    Ramsey43Bits witness = ramsey43_find_k5(graph.red0, graph.red1);
    if (lane == 0u && witness != 0ull) dead[warp] = 1u;
    __syncwarp(RAMSEY43_FULL_WARP);
    if (dead[warp] == 0u) {
        witness = ramsey43_find_k5(graph.blue0, graph.blue1);
        if (witness != 0ull) witness |= 1ull << 30;
        if (lane == 0u && witness != 0ull) dead[warp] = 1u;
    }
    __syncwarp(RAMSEY43_FULL_WARP);
    bool complete = ramsey43_complete(graph);
    if (lane == 0u) {
        verdicts[universe] = dead[warp] != 0u ? RAMSEY43_BAD
            : complete ? RAMSEY43_COMPLETE_SAFE : RAMSEY43_OPEN;
        if (witnesses != 0) witnesses[universe] = witness;
    }
}

// Busqueda experimental: orden comun de aristas, dos alternativas exactas y
// continuacion persistente. El presupuesto solo suspende; nunca prueba UNSAT.
static const unsigned RAMSEY43_LIVE = 0u;
static const unsigned RAMSEY43_PRUNED = 1u;
static const unsigned RAMSEY43_SAT = 2u;
static const unsigned RAMSEY43_EXHAUSTED = 3u;
static const unsigned RAMSEY43_ERROR = 4u;

__device__ __forceinline__ unsigned ramsey43_mix(unsigned x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

__device__ __forceinline__ void ramsey43_order(
    unsigned seed, unsigned& start, unsigned& stride)
{
    start = seed == 0u ? 0u : ramsey43_mix(seed) % 903u;
    stride = seed == 0u ? 1u : 1u + ramsey43_mix(seed ^ 0x9e3779b9u) % 902u;
    // 903 = 3*7*43: la coprimalidad hace de p -> arista una biyeccion.
    while (stride % 3u == 0u || stride % 7u == 0u || stride % 43u == 0u) {
        stride = stride == 902u ? 1u : stride + 1u;
    }
}

__device__ __forceinline__ void ramsey43_edge(
    unsigned position, unsigned start, unsigned stride, unsigned& a, unsigned& b)
{
    unsigned index = (start + stride * position) % 903u;
    unsigned low = 0u, high = 42u;
    while (low + 1u < high) {
        unsigned middle = (low + high) >> 1;
        unsigned first = middle * (85u - middle) / 2u;
        if (first <= index) low = middle;
        else high = middle;
    }
    a = low;
    b = a + 1u + index - a * (85u - a) / 2u;
}

__device__ __forceinline__ bool ramsey43_fixed(
    Ramsey43WarpGraph base, unsigned a, unsigned b)
{
    return ((ramsey43_row(base.red0 | base.blue0, base.red1 | base.blue1, a) >> b) & 1ull) != 0ull;
}

// Actualiza las dos filas propietarias. colour=2 borra ambos colores.
__device__ __forceinline__ void ramsey43_set_edge(
    Ramsey43WarpGraph& graph, unsigned a, unsigned b, unsigned colour)
{
    unsigned lane = threadIdx.x & 31u;
    if ((a & 31u) == lane) {
        Ramsey43Bits mask = 1ull << b;
        if (a < 32u) {
            graph.red0 = (graph.red0 & ~mask) | (colour == 0u ? mask : 0ull);
            graph.blue0 = (graph.blue0 & ~mask) | (colour == 1u ? mask : 0ull);
        } else {
            graph.red1 = (graph.red1 & ~mask) | (colour == 0u ? mask : 0ull);
            graph.blue1 = (graph.blue1 & ~mask) | (colour == 1u ? mask : 0ull);
        }
    }
    if ((b & 31u) == lane) {
        Ramsey43Bits mask = 1ull << a;
        if (b < 32u) {
            graph.red0 = (graph.red0 & ~mask) | (colour == 0u ? mask : 0ull);
            graph.blue0 = (graph.blue0 & ~mask) | (colour == 1u ? mask : 0ull);
        } else {
            graph.red1 = (graph.red1 & ~mask) | (colour == 0u ? mask : 0ull);
            graph.blue1 = (graph.blue1 & ~mask) | (colour == 1u ? mask : 0ull);
        }
    }
}

// El padre no tiene K5. Si aparece uno al anadir (a,b), necesariamente contiene
// esa arista: basta encontrar un triangulo dentro de N(a) & N(b) de su color.
__device__ __forceinline__ bool ramsey43_new_k5(
    Ramsey43WarpGraph graph, unsigned a, unsigned b, unsigned colour)
{
    unsigned lane = threadIdx.x & 31u;
    Ramsey43Bits row0 = colour == 0u ? graph.red0 : graph.blue0;
    Ramsey43Bits row1 = colour == 0u ? graph.red1 : graph.blue1;
    Ramsey43Bits common = ramsey43_row(row0, row1, a) & ramsey43_row(row0, row1, b);
    for (unsigned pass = 0u; pass < 2u; ++pass) {
        unsigned c = lane + 32u * pass;
        unsigned safe_c = c < 43u ? c : 0u;
        Ramsey43Bits row_c = ramsey43_row(row0, row1, safe_c);
        Ramsey43Bits candidates_d = c < 43u && ((common >> safe_c) & 1ull)
            ? common & row_c & ramsey43_after(safe_c) : 0ull;
        while (__any_sync(RAMSEY43_FULL_WARP, candidates_d != 0ull)) {
            bool valid_d = candidates_d != 0ull;
            unsigned d = valid_d ? (unsigned)(__ffsll((long long)candidates_d) - 1) : 0u;
            Ramsey43Bits row_d = ramsey43_row(row0, row1, d);
            Ramsey43Bits candidates_e = valid_d
                ? common & row_c & row_d & ramsey43_after(d) : 0ull;
            if (__any_sync(RAMSEY43_FULL_WARP, candidates_e != 0ull)) return true;
            candidates_d &= candidates_d - 1ull;
        }
    }
    return false;
}

// Padre viable, (a,b) libre e indices uniformes para los 32 lanes activos.
// Retorno: bit0 = rojo permitido; bit1 = azul permitido; 0 = conflicto.
// El triangulo en N(a)&N(b) aporta las nueve aristas del K5 menos (a,b).
__device__ __forceinline__ unsigned ramsey43_allowed_colours(
    Ramsey43WarpGraph graph, unsigned a, unsigned b)
{
    Ramsey43Bits ra = ramsey43_row(graph.red0, graph.red1, a);
    Ramsey43Bits rb = ramsey43_row(graph.red0, graph.red1, b);
    Ramsey43Bits ba = ramsey43_row(graph.blue0, graph.blue1, a);
    Ramsey43Bits bb = ramsey43_row(graph.blue0, graph.blue1, b);
    // R(4,5)=25: todo modelo K43 tiene grado de cada color <=24.
    bool red_blocked = __popcll(ra) >= 24 || __popcll(rb) >= 24;
    bool blue_blocked = __popcll(ba) >= 24 || __popcll(bb) >= 24;
    // Estos predicados son uniformes; ninguna colectiva pierde participantes.
    if (!red_blocked && __popcll(ra & rb) >= 3)
        red_blocked = ramsey43_new_k5(graph, a, b, 0u);
    if (!blue_blocked && __popcll(ba & bb) >= 3)
        blue_blocked = ramsey43_new_k5(graph, a, b, 1u);
    return (red_blocked ? 0u : 1u) | (blue_blocked ? 0u : 2u);
}

__device__ __forceinline__ bool ramsey43_degrees_viable(Ramsey43WarpGraph graph)
{
    unsigned lane = threadIdx.x & 31u;
    bool invalid = __popcll(graph.red0) > 24 || __popcll(graph.blue0) > 24;
    if (lane < 11u)
        invalid = invalid || __popcll(graph.red1) > 24 || __popcll(graph.blue1) > 24;
    return !__any_sync(RAMSEY43_FULL_WARP, invalid);
}

// Sonda del mismo predicado usado por DFS; solo para validacion del host.
// graphs respeta el contrato de ramsey43_check; edges = {a,b} por grafo,
// con 0<=a<b<43 y arista libre. Cada resultado es uniforme por warp.
extern "C" __global__ void ramsey43_lookahead_check(
    const Ramsey43Bits* graphs, unsigned count,
    const unsigned* edges, unsigned* allowed)
{
    if (blockDim.x != 256u || blockDim.y != 1u || blockDim.z != 1u) return;
    unsigned lane = threadIdx.x & 31u;
    unsigned universe = blockIdx.x * 8u + (threadIdx.x >> 5);
    if (universe >= count) return;
    const Ramsey43Bits* rows = graphs + (Ramsey43Bits)universe * 86ull;
    Ramsey43WarpGraph graph;
    graph.red0 = rows[lane];
    graph.blue0 = rows[43u + lane];
    graph.red1 = lane < 11u ? rows[lane + 32u] : 0ull;
    graph.blue1 = lane < 11u ? rows[75u + lane] : 0ull;
    unsigned a = __shfl_sync(RAMSEY43_FULL_WARP,
        lane == 0u ? edges[(Ramsey43Bits)universe * 2ull] : 0u, 0);
    unsigned b = __shfl_sync(RAMSEY43_FULL_WARP,
        lane == 0u ? edges[(Ramsey43Bits)universe * 2ull + 1ull] : 0u, 0);
    unsigned mask = ramsey43_allowed_colours(graph, a, b);
    if (lane == 0u) allowed[universe] = mask;
}

__device__ __forceinline__ unsigned ramsey43_first_colour(
    unsigned seed, unsigned universe, unsigned position)
{
    return ramsey43_mix(seed ^ (universe * 0x9e3779b9u) ^ (position * 0x85ebca6bu)) & 1u;
}

__device__ __forceinline__ void ramsey43_store_rows(
    Ramsey43Bits* rows, Ramsey43WarpGraph graph)
{
    unsigned lane = threadIdx.x & 31u;
    rows[lane] = graph.red0;
    rows[43u + lane] = graph.blue0;
    if (lane < 11u) {
        rows[lane + 32u] = graph.red1;
        rows[75u + lane] = graph.blue1;
    }
}

// Arena: rows[U][86], alt[U][32], control[U][5]. Control contiene
// {iniciado, profundidad, suelo_prefijo, estado, retrocediendo}.
// counters base = {transiciones, podas, universos_terminados}; smart añade
// {decisiones, asignaciones_forzadas, pruebas_lookahead, conflictos_lookahead}.
// found: 0 sin modelo, 1 reclamado, 2 modelo publicado. Solo un ganador escribe.
// LookAhead=false se especializa al motor original: sin lecturas/contadores extra.
template <bool LookAhead>
__device__ __forceinline__ void ramsey43_search_body(
    const Ramsey43Bits* base, Ramsey43Bits* state_rows, unsigned* state_alt,
    unsigned* control, unsigned universe_count, unsigned first, unsigned count,
    unsigned prefix_bits, unsigned order_seed, unsigned steps,
    unsigned* found, Ramsey43Bits* model, Ramsey43Bits* counters)
{
    if (blockDim.x != 256u || blockDim.y != 1u || blockDim.z != 1u) return;
    unsigned lane = threadIdx.x & 31u;
    unsigned warp = threadIdx.x >> 5;
    unsigned local = blockIdx.x * 8u + warp;
    if (local >= count || first >= universe_count || local >= universe_count - first) return;
    unsigned universe = first + local;
    unsigned published = lane == 0u ? atomicAdd(found, 0u) : 0u;
    published = __shfl_sync(RAMSEY43_FULL_WARP, published, 0);
    if (published != 0u) return;

    unsigned* ctl = control + (Ramsey43Bits)universe * 5ull;
    unsigned initialized = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[0] : 0u, 0);
    unsigned depth = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[1] : 0u, 0);
    unsigned floor = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[2] : 0u, 0);
    unsigned status = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[3] : 0u, 0);
    unsigned backtrack = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[4] : 0u, 0);
    if (initialized != 0u && status != RAMSEY43_LIVE) return;
    __shared__ unsigned dead[8];
    if (lane == 0u) dead[warp] = 0u;
    __syncwarp(RAMSEY43_FULL_WARP);

    Ramsey43WarpGraph original;
    original.red0 = base[lane];
    original.blue0 = base[43u + lane];
    original.red1 = lane < 11u ? base[lane + 32u] : 0ull;
    original.blue1 = lane < 11u ? base[75u + lane] : 0ull;
    Ramsey43Bits* rows = state_rows + (Ramsey43Bits)universe * 86ull;
    unsigned* alternatives = state_alt + (Ramsey43Bits)universe * 32ull;
    Ramsey43WarpGraph graph = original;
    unsigned alt_word = 0u;
    unsigned start, stride;
    ramsey43_order(order_seed, start, stride);
    Ramsey43Bits transitions = 0ull, prunes = 0ull;
    // Cada rafaga acepta <=65536 pasos + <=20 bits de prefijo: los cuatro
    // acumuladores locales caben en 32 bits; solo los totales son de 64 bits.
    unsigned decisions = 0u, forced = 0u;
    unsigned lookahead_checks = 0u, lookahead_conflicts = 0u;

    if (initialized == 0u) {
        initialized = 1u;
        depth = floor = backtrack = 0u;
        status = RAMSEY43_LIVE;
        if (prefix_bits > 20u || universe_count != (1u << prefix_bits)) {
            status = RAMSEY43_ERROR;
        } else {
            Ramsey43Bits witness = ramsey43_find_k5(graph.red0, graph.red1);
            if (witness == 0ull) witness = ramsey43_find_k5(graph.blue0, graph.blue1);
            bool invalid_degrees = LookAhead && !ramsey43_degrees_viable(graph);
            if (lane == 0u) dead[warp] = witness != 0ull || invalid_degrees;
            __syncwarp(RAMSEY43_FULL_WARP);
            if (dead[warp] != 0u) {
                status = RAMSEY43_PRUNED;
                ++prunes;
                if (LookAhead && invalid_degrees) ++lookahead_conflicts;
            }
        }
        unsigned assigned = 0u;
        while (status == RAMSEY43_LIVE && assigned < prefix_bits && depth < 903u) {
            unsigned a, b;
            ramsey43_edge(depth, start, stride, a, b);
            ++depth;
            if (ramsey43_fixed(original, a, b)) continue;
            unsigned colour = (universe >> assigned) & 1u;
            ++assigned;
            if (LookAhead) {
                // El prefijo conserva su correspondencia estatica bit -> arista.
                // Una inferencia no omite bits ni cambia la familia de cubos.
                unsigned allowed = ramsey43_allowed_colours(graph, a, b);
                ++lookahead_checks;
                if ((allowed & (1u << colour)) == 0u) {
                    status = RAMSEY43_PRUNED;
                    ++prunes;
                    ++lookahead_conflicts;
                    break;
                }
                ramsey43_set_edge(graph, a, b, colour);
                continue;
            }
            ramsey43_set_edge(graph, a, b, colour);
            bool conflict = ramsey43_new_k5(graph, a, b, colour);
            if (lane == 0u) dead[warp] = conflict;
            __syncwarp(RAMSEY43_FULL_WARP);
            if (dead[warp] != 0u) {
                status = RAMSEY43_PRUNED;
                ++prunes;
            }
        }
        floor = depth;
        if (status == RAMSEY43_LIVE && assigned != prefix_bits) status = RAMSEY43_ERROR;
    } else {
        graph.red0 = rows[lane];
        graph.blue0 = rows[43u + lane];
        graph.red1 = lane < 11u ? rows[lane + 32u] : 0ull;
        graph.blue1 = lane < 11u ? rows[75u + lane] : 0ull;
        alt_word = alternatives[lane];
        if (depth > 903u || floor > depth || backtrack > 1u) status = RAMSEY43_ERROR;
    }

    for (unsigned step = 0u; step < steps && status == RAMSEY43_LIVE; ++step) {
        ++transitions;
        if (backtrack == 0u && depth == 903u) {
            status = ramsey43_complete(graph) ? RAMSEY43_SAT : RAMSEY43_ERROR;
            break;
        }
        if (backtrack != 0u && depth <= floor) {
            status = RAMSEY43_EXHAUSTED;
            break;
        }
        if (backtrack != 0u) --depth;
        unsigned position = depth;
        unsigned a, b;
        ramsey43_edge(position, start, stride, a, b);
        if (ramsey43_fixed(original, a, b)) {
            if (backtrack == 0u) ++depth;
            continue;
        }
        unsigned owner = position & 31u;
        unsigned mask = 1u << (position >> 5);
        unsigned colour = ramsey43_first_colour(order_seed, universe, position);
        // Trail implicito reversible: el orden fijo reconstruye cada arista,
        // depth delimita la pila y alt_word registra alternativas consumidas.
        // Una fuerza ocupa una entrada, pero no crea una decision binaria.
        // Estas tres piezas se guardan entre rafagas; no hay arrays por hilo.
        if (backtrack != 0u) {
            unsigned tried = (__shfl_sync(RAMSEY43_FULL_WARP, alt_word, owner) & mask) != 0u;
            ramsey43_set_edge(graph, a, b, 2u);
            if (tried != 0u) {
                if (lane == owner) alt_word &= ~mask;
                continue;
            }
            if (lane == owner) alt_word |= mask;
            colour ^= 1u;
        }
        if (LookAhead) {
            unsigned allowed = ramsey43_allowed_colours(graph, a, b);
            ++lookahead_checks;
            if (backtrack != 0u) {
                if ((allowed & (1u << colour)) == 0u) {
                    ++prunes;
                    ++lookahead_conflicts;
                    // La arista sigue libre; el siguiente paso elimina su padre.
                    continue;
                }
                ++decisions;
            } else {
                if (allowed == 0u) {
                    ++prunes;
                    ++lookahead_conflicts;
                    backtrack = 1u;
                    continue;
                }
                if (allowed == 1u || allowed == 2u) {
                    colour = allowed == 1u ? 0u : 1u;
                    if (lane == owner) alt_word |= mask;
                    ++forced;
                } else {
                    if (lane == owner) alt_word &= ~mask;
                    ++decisions;
                }
            }
            ramsey43_set_edge(graph, a, b, colour);
            ++depth;
            backtrack = 0u;
        } else {
            if (backtrack == 0u && lane == owner) alt_word &= ~mask;
            ramsey43_set_edge(graph, a, b, colour);
            ++depth;
            bool conflict = ramsey43_new_k5(graph, a, b, colour);
            if (lane == 0u) dead[warp] = conflict;
            __syncwarp(RAMSEY43_FULL_WARP);
            backtrack = dead[warp] != 0u ? 1u : 0u;
            if (backtrack != 0u) ++prunes;
        }
    }

    if (status == RAMSEY43_SAT) {
        unsigned winner = lane == 0u ? atomicCAS(found, 0u, 1u) == 0u : 0u;
        winner = __shfl_sync(RAMSEY43_FULL_WARP, winner, 0);
        if (winner != 0u) {
            ramsey43_store_rows(model, graph);
            // Cada lane publica sus propias escrituras antes de marcar ready.
            __threadfence();
            __syncwarp(RAMSEY43_FULL_WARP);
            if (lane == 0u) atomicExch(found, 2u);
        }
    }
    ramsey43_store_rows(rows, graph);
    alternatives[lane] = alt_word;
    if (lane == 0u) {
        ctl[0] = initialized;
        ctl[1] = depth;
        ctl[2] = floor;
        ctl[3] = status;
        ctl[4] = backtrack;
        atomicAdd(counters, transitions);
        atomicAdd(counters + 1, prunes);
        if (LookAhead) {
            atomicAdd(counters + 3, (Ramsey43Bits)decisions);
            atomicAdd(counters + 4, (Ramsey43Bits)forced);
            atomicAdd(counters + 5, (Ramsey43Bits)lookahead_checks);
            atomicAdd(counters + 6, (Ramsey43Bits)lookahead_conflicts);
        }
        if (status == RAMSEY43_PRUNED || status == RAMSEY43_SAT || status == RAMSEY43_EXHAUSTED) {
            atomicAdd(counters + 2, 1ull);
        }
    }
}

// ABI original preservada: los kernels se eligen en el host, no por un flag
// dentro del bucle. El modo base necesita unicamente sus tres contadores.
extern "C" __global__ void ramsey43_search(
    const Ramsey43Bits* base, Ramsey43Bits* state_rows, unsigned* state_alt,
    unsigned* control, unsigned universe_count, unsigned first, unsigned count,
    unsigned prefix_bits, unsigned order_seed, unsigned steps,
    unsigned* found, Ramsey43Bits* model, Ramsey43Bits* counters)
{
    ramsey43_search_body<false>(base, state_rows, state_alt, control,
        universe_count, first, count, prefix_bits, order_seed, steps,
        found, model, counters);
}

// Mantener tres bloques de 256 hilos por SM sin imponer maxrregcount global.
extern "C" __global__ __launch_bounds__(256, 3) void ramsey43_search_smart(
    const Ramsey43Bits* base, Ramsey43Bits* state_rows, unsigned* state_alt,
    unsigned* control, unsigned universe_count, unsigned first, unsigned count,
    unsigned prefix_bits, unsigned order_seed, unsigned steps,
    unsigned* found, Ramsey43Bits* model, Ramsey43Bits* counters)
{
    ramsey43_search_body<true>(base, state_rows, state_alt, control,
        universe_count, first, count, prefix_bits, order_seed, steps,
        found, model, counters);
}

// Neuro-Ramsey Fase 1: diario explicito opcional; los kernels anteriores conservan
// su arena y su ABI. Cada frame describe una asignacion mutable real, no una
// posicion del selector. Base y prefijo quedan protegidos fuera de este diario.
typedef unsigned short Ramsey43Frame;
static_assert(sizeof(Ramsey43Frame) == 2, "Ramsey43Frame requiere 16 bits");
static const unsigned RAMSEY43_FRAME_EDGE = 0x03ffu;
static const unsigned RAMSEY43_FRAME_COLOUR = 0x0400u;
static const unsigned RAMSEY43_FRAME_ALT = 0x0800u;
static const unsigned RAMSEY43_FRAME_FORCED = 0x1000u;
static const unsigned RAMSEY43_FRAME_RESERVED = 0xe000u;

__device__ __forceinline__ Ramsey43Frame ramsey43_frame(
    unsigned edge, unsigned colour, bool alternate_consumed, bool forced)
{
    return (Ramsey43Frame)(edge | (colour << 10)
        | (alternate_consumed ? RAMSEY43_FRAME_ALT : 0u)
        | (forced ? RAMSEY43_FRAME_FORCED : 0u));
}

// Primera arista libre lexicografica. Cada lane examina sus propias filas,
// por tanto no hay escaneo de 903 aristas ni arrays locales indexados por hilo.
// El resultado (903 si no queda ninguna) es uniforme para todo el warp.
// Un selector neuronal futuro puede sustituir esta funcion sin cambiar pop().
__device__ __forceinline__ unsigned ramsey43_pick_free_edge(Ramsey43WarpGraph graph)
{
    unsigned lane = threadIdx.x & 31u;
    Ramsey43Bits unknown0 = ~(graph.red0 | graph.blue0) & ramsey43_after(lane);
    unsigned candidate = 903u;
    if (unknown0 != 0ull) {
        unsigned b = (unsigned)(__ffsll((long long)unknown0) - 1);
        candidate = lane * (85u - lane) / 2u + b - lane - 1u;
    }
    if (lane < 11u) {
        unsigned a = lane + 32u;
        Ramsey43Bits unknown1 = ~(graph.red1 | graph.blue1) & ramsey43_after(a);
        if (unknown1 != 0ull) {
            unsigned b = (unsigned)(__ffsll((long long)unknown1) - 1);
            unsigned other = a * (85u - a) / 2u + b - a - 1u;
            candidate = other < candidate ? other : candidate;
        }
    }
    for (unsigned delta = 16u; delta != 0u; delta >>= 1) {
        unsigned other = __shfl_down_sync(RAMSEY43_FULL_WARP, candidate, delta);
        candidate = other < candidate ? other : candidate;
    }
    return __shfl_sync(RAMSEY43_FULL_WARP, candidate, 0);
}

// rows[U][86], journal[U][max(capacity,1)], control[U][5]. Control:
// {iniciado, profundidad_mutable, root_floor=0, estado, retrocediendo}.
// Solo lane0 lee/escribe frames; shfl distribuye cada lectura a los 32 lanes.
// Todos los frames vivos corresponden a aristas distintas y asignadas. No se
// recalculan ni el ID ni la primera fase al retroceder, incluso entre rafagas.
// Solo las variantes Dataset=true contabilizan decisiones o escriben muestras.
// Hash por universo/ordinal evita sincronizar un contador global por rama.
__device__ __forceinline__ void ramsey43_capture_parent(
    Ramsey43WarpGraph graph, unsigned universe, unsigned depth, unsigned edge,
    unsigned colour, unsigned seed, Ramsey43Bits& sequence,
    unsigned sample_every, unsigned capacity, Ramsey43Bits* sample_rows,
    unsigned* sample_meta, unsigned* sample_control)
{
    if (sequence == ~0ull) return;
    ++sequence;
    Ramsey43Bits hash = sequence ^ ((Ramsey43Bits)universe << 32) ^ (Ramsey43Bits)seed;
    hash += 0x9e3779b97f4a7c15ull;
    hash = (hash ^ (hash >> 30)) * 0xbf58476d1ce4e5b9ull;
    hash = (hash ^ (hash >> 27)) * 0x94d049bb133111ebull;
    hash ^= hash >> 31;
    if (sample_every == 0u || hash % sample_every != 0ull) return;
    unsigned lane = threadIdx.x & 31u;
    unsigned slot = lane == 0u ? atomicAdd(sample_control, 1u) : 0u;
    slot = __shfl_sync(RAMSEY43_FULL_WARP, slot, 0);
    if (slot >= capacity) {
        if (lane == 0u) atomicAdd(sample_control + 1, 1u);
        return;
    }
    ramsey43_store_rows(sample_rows + (Ramsey43Bits)slot * 86ull, graph);
    if (lane == 0u) {
        unsigned* meta = sample_meta + (Ramsey43Bits)slot * 8ull;
        meta[0] = universe;
        meta[1] = (unsigned)sequence;
        meta[2] = (unsigned)(sequence >> 32);
        meta[3] = depth;
        meta[4] = edge;
        meta[5] = colour;
        meta[6] = meta[7] = 0u;
    }
}

template <bool LookAhead, bool Neuro = false, bool Dataset = false>
__device__ __forceinline__ void ramsey43_search_journal_body(
    const Ramsey43Bits* base, Ramsey43Bits* state_rows, Ramsey43Frame* state_journal,
    unsigned* control, unsigned universe_count, unsigned first, unsigned count,
    unsigned prefix_bits, unsigned order_seed, unsigned steps, unsigned journal_capacity,
    unsigned* found, Ramsey43Bits* model, Ramsey43Bits* counters,
    unsigned* policy_control = 0, Ramsey43Bits* policy_generation = 0,
    unsigned policy_interval = 0u, Ramsey43Bits* policy_counters = 0,
    Ramsey43Bits* sample_sequence = 0, Ramsey43Bits* sample_rows = 0,
    unsigned* sample_meta = 0, unsigned* sample_control = 0,
    unsigned sample_every = 0u, unsigned sample_capacity = 0u)
{
    if (blockDim.x != 256u || blockDim.y != 1u || blockDim.z != 1u) return;
    unsigned lane = threadIdx.x & 31u;
    unsigned local = blockIdx.x * 8u + (threadIdx.x >> 5);
    if (local >= count || first >= universe_count || local >= universe_count - first) return;
    unsigned universe = first + local;
    unsigned published = lane == 0u ? atomicAdd(found, 0u) : 0u;
    if (__shfl_sync(RAMSEY43_FULL_WARP, published, 0) != 0u) return;

    unsigned* ctl = control + (Ramsey43Bits)universe * 5ull;
    unsigned initialized = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[0] : 0u, 0);
    unsigned depth = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[1] : 0u, 0);
    unsigned floor = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[2] : 0u, 0);
    unsigned status = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[3] : 0u, 0);
    unsigned backtrack = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[4] : 0u, 0);
    if (initialized != 0u && status != RAMSEY43_LIVE) return;

    // El modo opcional suspende, no espera dentro de un warp residente.
    // policy[U][3] = {decisiones_desde_consulta, estado 0/1/2, eleccion}.
    // La generacion de 64 bits identifica el snapshot congelado; la CPU solo
    // coordina eventos. Las variantes Neuro=false no leen estos punteros.
    unsigned* policy = 0;
    unsigned since_policy = 0u, policy_state = 0u;
    if (Neuro) {
        policy = policy_control + (Ramsey43Bits)universe * 3ull;
        if (initialized != 0u) {
            since_policy = __shfl_sync(RAMSEY43_FULL_WARP,
                lane == 0u ? policy[0] : 0u, 0);
            policy_state = __shfl_sync(RAMSEY43_FULL_WARP,
                lane == 0u ? policy[1] : 0u, 0);
            if (policy_state == 1u) return; // WAIT: filas y journal inmutables.
            if (policy_state > 2u || (policy_state == 2u && backtrack != 0u)
                || policy_interval == 0u || since_policy > policy_interval)
                status = RAMSEY43_ERROR;
        }
    }

    Ramsey43Bits* rows = state_rows + (Ramsey43Bits)universe * 86ull;
    unsigned stride = journal_capacity == 0u ? 1u : journal_capacity;
    Ramsey43Frame* journal = state_journal + (Ramsey43Bits)universe * stride;
    Ramsey43WarpGraph graph;
    Ramsey43Bits transitions = 0ull, prunes = 0ull;
    unsigned decisions = 0u, forced = 0u, checks = 0u, conflicts = 0u;
    Ramsey43Bits sequence = Dataset && initialized != 0u ? sample_sequence[universe] : 0ull;

    if (initialized == 0u) {
        graph.red0 = base[lane];
        graph.blue0 = base[43u + lane];
        graph.red1 = lane < 11u ? base[lane + 32u] : 0ull;
        graph.blue1 = lane < 11u ? base[75u + lane] : 0ull;
        initialized = 1u;
        depth = floor = backtrack = 0u;
        status = RAMSEY43_LIVE;
        if (Neuro) {
            since_policy = policy_interval; // Primera consulta en la primera eleccion libre.
            policy_state = 0u;
            if (lane == 0u) {
                policy[2] = 0xffffu;
                policy_generation[universe] = 0ull;
            }
        }
        if (prefix_bits > 20u || universe_count != (1u << prefix_bits)
            || journal_capacity > 903u || (Neuro && policy_interval == 0u)) {
            status = RAMSEY43_ERROR;
        } else {
            Ramsey43Bits witness = ramsey43_find_k5(graph.red0, graph.red1);
            if (witness == 0ull) witness = ramsey43_find_k5(graph.blue0, graph.blue1);
            bool invalid_degrees = LookAhead && !ramsey43_degrees_viable(graph);
            if (witness != 0ull || invalid_degrees) {
                status = RAMSEY43_PRUNED;
                ++prunes;
                if (LookAhead && invalid_degrees) ++conflicts;
            }
        }
        // Solo la particion inicial usa el orden seeded previo. Cada bit sigue
        // representando la misma arista del cubo, aunque el suffix cambie orden.
        unsigned assigned = 0u, start, permutation_stride;
        ramsey43_order(order_seed, start, permutation_stride);
        for (unsigned position = 0u; status == RAMSEY43_LIVE
            && assigned < prefix_bits && position < 903u; ++position) {
            unsigned a, b;
            ramsey43_edge(position, start, permutation_stride, a, b);
            if (ramsey43_fixed(graph, a, b)) continue;
            unsigned colour = (universe >> assigned) & 1u;
            ++assigned;
            bool rejected;
            if (LookAhead) {
                unsigned allowed = ramsey43_allowed_colours(graph, a, b);
                ++checks;
                rejected = (allowed & (1u << colour)) == 0u;
            } else {
                rejected = ramsey43_new_k5(graph, a, b, colour);
            }
            if (rejected) {
                status = RAMSEY43_PRUNED;
                ++prunes;
                if (LookAhead) ++conflicts;
                break;
            }
            ramsey43_set_edge(graph, a, b, colour);
        }
        if (status == RAMSEY43_LIVE && assigned != prefix_bits) status = RAMSEY43_ERROR;
    } else {
        graph.red0 = rows[lane];
        graph.blue0 = rows[43u + lane];
        graph.red1 = lane < 11u ? rows[lane + 32u] : 0ull;
        graph.blue1 = lane < 11u ? rows[75u + lane] : 0ull;
        if (depth > journal_capacity || journal_capacity > 903u
            || floor != 0u || backtrack > 1u) status = RAMSEY43_ERROR;
    }

    for (unsigned step = 0u; step < steps && status == RAMSEY43_LIVE; ++step) {
        ++transitions;
        unsigned edge, colour;
        bool alternate_consumed = false, is_forced = false;
        if (backtrack != 0u) {
            if (depth == 0u) {
                status = RAMSEY43_EXHAUSTED;
                break;
            }
            --depth;
            unsigned frame = lane == 0u ? journal[depth] : 0u;
            frame = __shfl_sync(RAMSEY43_FULL_WARP, frame, 0);
            edge = frame & RAMSEY43_FRAME_EDGE;
            colour = (frame & RAMSEY43_FRAME_COLOUR) != 0u ? 1u : 0u;
            alternate_consumed = (frame & RAMSEY43_FRAME_ALT) != 0u;
            is_forced = (frame & RAMSEY43_FRAME_FORCED) != 0u;
            if (edge >= 903u || (frame & RAMSEY43_FRAME_RESERVED) != 0u
                || (is_forced && !alternate_consumed)) {
                status = RAMSEY43_ERROR;
                break;
            }
            unsigned a, b;
            ramsey43_edge(edge, 0u, 1u, a, b);
            Ramsey43Bits current = colour == 0u
                ? ramsey43_row(graph.red0, graph.red1, a)
                : ramsey43_row(graph.blue0, graph.blue1, a);
            if (((current >> b) & 1ull) == 0ull) {
                status = RAMSEY43_ERROR;
                break;
            }
            ramsey43_set_edge(graph, a, b, 2u);
            if (is_forced || alternate_consumed) continue;
            // El color complementario procede del frame, no del selector ni
            // del hash de fase; el padre ya esta restaurado y libre de K5.
            colour ^= 1u;
            alternate_consumed = true;
            bool rejected;
            if (LookAhead) {
                unsigned allowed = ramsey43_allowed_colours(graph, a, b);
                ++checks;
                rejected = (allowed & (1u << colour)) == 0u;
            } else {
                rejected = ramsey43_new_k5(graph, a, b, colour);
            }
            if (rejected) {
                ++prunes;
                if (LookAhead) ++conflicts;
                continue;
            }
            if (LookAhead) ++decisions;
            if (Neuro && since_policy < policy_interval) ++since_policy;
            if (Dataset) ramsey43_capture_parent(graph, universe, depth, edge, colour,
                order_seed, sequence, sample_every, sample_capacity,
                sample_rows, sample_meta, sample_control);
            ramsey43_set_edge(graph, a, b, colour);
        } else {
            edge = ramsey43_pick_free_edge(graph);
            if (edge == 903u) {
                bool valid = ramsey43_complete(graph);
                if (valid) valid = ramsey43_find_k5(graph.red0, graph.red1) == 0ull;
                if (valid) valid = ramsey43_find_k5(graph.blue0, graph.blue1) == 0ull;
                status = valid ? RAMSEY43_SAT : RAMSEY43_ERROR;
                break;
            }
            if (depth >= journal_capacity) {
                status = RAMSEY43_ERROR;
                break;
            }
            unsigned a, b;
            unsigned predicted_colour = 2u;
            if (Neuro && policy_state == 2u) {
                unsigned choice = __shfl_sync(RAMSEY43_FULL_WARP,
                    lane == 0u ? policy[2] : 0xffffu, 0);
                // READY se consume una sola vez. El selector GPU ya verifico
                // nonce, profundidad y candidatos; aqui se revalida libertad.
                policy_state = 0u;
                since_policy = 0u;
                unsigned chosen_edge = choice & RAMSEY43_FRAME_EDGE;
                if ((choice & ~0x07ffu) == 0u && chosen_edge < 903u) {
                    unsigned ca, cb;
                    ramsey43_edge(chosen_edge, 0u, 1u, ca, cb);
                    if (!ramsey43_fixed(graph, ca, cb)) {
                        edge = chosen_edge;
                        predicted_colour = (choice >> 10) & 1u;
                    }
                }
            }
            ramsey43_edge(edge, 0u, 1u, a, b);
            colour = predicted_colour < 2u ? predicted_colour
                : ramsey43_first_colour(order_seed, universe, edge);
            if (LookAhead) {
                unsigned allowed = ramsey43_allowed_colours(graph, a, b);
                ++checks;
                if (allowed == 0u) {
                    ++prunes;
                    ++conflicts;
                    backtrack = 1u;
                    continue;
                }
                is_forced = allowed == 1u || allowed == 2u;
                if (Neuro && !is_forced && since_policy >= policy_interval) {
                    Ramsey43Bits nonce = lane == 0u ? policy_generation[universe] : 0ull;
                    nonce = __shfl_sync(RAMSEY43_FULL_WARP, nonce, 0);
                    if (nonce == ~0ull) {
                        status = RAMSEY43_ERROR;
                        break;
                    }
                    if (lane == 0u) {
                        policy_generation[universe] = nonce + 1ull;
                        atomicAdd(policy_counters, 1ull);
                    }
                    policy_state = 1u;
                    // El checkpoint sigue viable: aun no se asigna la arista.
                    // Otro kernel del mismo flujo empaqueta X desde estas filas
                    // antes de features_ready; no hay spinning ni barrera CTA.
                    break;
                }
                if (is_forced) {
                    colour = allowed == 1u ? 0u : 1u;
                    alternate_consumed = true;
                    ++forced;
                } else {
                    ++decisions;
                    if (Neuro && since_policy < policy_interval) ++since_policy;
                }
            } else if (ramsey43_new_k5(graph, a, b, colour)) {
                // Rechazar antes de escribir mantiene tambien los checkpoints
                // base libres de K5. El segundo color se prueba exactamente.
                ++prunes;
                colour ^= 1u;
                alternate_consumed = true;
                if (ramsey43_new_k5(graph, a, b, colour)) {
                    ++prunes;
                    backtrack = 1u;
                    continue;
                }
            }
            if (Dataset && !is_forced) ramsey43_capture_parent(graph, universe,
                depth, edge, colour, order_seed, sequence, sample_every,
                sample_capacity, sample_rows, sample_meta, sample_control);
            ramsey43_set_edge(graph, a, b, colour);
        }
        if (lane == 0u) journal[depth] = ramsey43_frame(
            edge, colour, alternate_consumed, is_forced);
        ++depth;
        backtrack = 0u;
    }

    if (status == RAMSEY43_SAT) {
        unsigned winner = lane == 0u ? atomicCAS(found, 0u, 1u) == 0u : 0u;
        if (__shfl_sync(RAMSEY43_FULL_WARP, winner, 0) != 0u) {
            ramsey43_store_rows(model, graph);
            __threadfence();
            __syncwarp(RAMSEY43_FULL_WARP);
            if (lane == 0u) atomicExch(found, 2u);
        }
    }
    ramsey43_store_rows(rows, graph);
    if (lane == 0u) {
        if (Dataset) sample_sequence[universe] = sequence;
        if (Neuro) {
            policy[0] = since_policy;
            policy[1] = policy_state;
        }
        ctl[0] = initialized;
        ctl[1] = depth;
        ctl[2] = 0u;
        ctl[3] = status;
        ctl[4] = backtrack;
        atomicAdd(counters, transitions);
        atomicAdd(counters + 1, prunes);
        if (LookAhead) {
            atomicAdd(counters + 3, (Ramsey43Bits)decisions);
            atomicAdd(counters + 4, (Ramsey43Bits)forced);
            atomicAdd(counters + 5, (Ramsey43Bits)checks);
            atomicAdd(counters + 6, (Ramsey43Bits)conflicts);
        }
        if (status == RAMSEY43_PRUNED || status == RAMSEY43_SAT || status == RAMSEY43_EXHAUSTED)
            atomicAdd(counters + 2, 1ull);
    }
}

extern "C" __global__ __launch_bounds__(256, 3) void ramsey43_search_journal(
    const Ramsey43Bits* base, Ramsey43Bits* state_rows, Ramsey43Frame* state_journal,
    unsigned* control, unsigned universe_count, unsigned first, unsigned count,
    unsigned prefix_bits, unsigned order_seed, unsigned steps, unsigned journal_capacity,
    unsigned* found, Ramsey43Bits* model, Ramsey43Bits* counters)
{
    ramsey43_search_journal_body<false>(base, state_rows, state_journal, control,
        universe_count, first, count, prefix_bits, order_seed, steps, journal_capacity,
        found, model, counters);
}

extern "C" __global__ __launch_bounds__(256, 3) void ramsey43_search_journal_smart(
    const Ramsey43Bits* base, Ramsey43Bits* state_rows, Ramsey43Frame* state_journal,
    unsigned* control, unsigned universe_count, unsigned first, unsigned count,
    unsigned prefix_bits, unsigned order_seed, unsigned steps, unsigned journal_capacity,
    unsigned* found, Ramsey43Bits* model, Ramsey43Bits* counters)
{
    ramsey43_search_journal_body<true>(base, state_rows, state_journal, control,
        universe_count, first, count, prefix_bits, order_seed, steps, journal_capacity,
        found, model, counters);
}

// Neuro-Ramsey Fase 2: misma legalidad y mismo journal; solo cambia la eleccion
// del siguiente frame libre. X/Y se preparan y consumen en kernels separados
// para no retener temporales FP16/FP32 junto a los registros del DFS.
extern "C" __global__ __launch_bounds__(256, 3) void ramsey43_search_neuro_smart(
    const Ramsey43Bits* base, Ramsey43Bits* state_rows, Ramsey43Frame* state_journal,
    unsigned* control, unsigned universe_count, unsigned first, unsigned count,
    unsigned prefix_bits, unsigned order_seed, unsigned steps, unsigned journal_capacity,
    unsigned* found, Ramsey43Bits* model, Ramsey43Bits* counters,
    unsigned* policy_control, Ramsey43Bits* policy_generation,
    unsigned policy_interval, Ramsey43Bits* policy_counters)
{
    ramsey43_search_journal_body<true, true>(base, state_rows, state_journal, control,
        universe_count, first, count, prefix_bits, order_seed, steps, journal_capacity,
        found, model, counters, policy_control, policy_generation, policy_interval, policy_counters);
}

// NVRTC recibe los dos textos concatenados desde Rust; NVCC usa el include.
extern "C" __global__ __launch_bounds__(256, 3) void ramsey43_search_dataset_smart(
    const Ramsey43Bits* base, Ramsey43Bits* state_rows, Ramsey43Frame* state_journal,
    unsigned* control, unsigned universe_count, unsigned first, unsigned count,
    unsigned prefix_bits, unsigned order_seed, unsigned steps, unsigned journal_capacity,
    unsigned* found, Ramsey43Bits* model, Ramsey43Bits* counters,
    Ramsey43Bits* sample_sequence, Ramsey43Bits* sample_rows,
    unsigned* sample_meta, unsigned* sample_control,
    unsigned sample_every, unsigned sample_capacity)
{
    ramsey43_search_journal_body<true, false, true>(base, state_rows, state_journal,
        control, universe_count, first, count, prefix_bits, order_seed, steps,
        journal_capacity, found, model, counters, 0, 0, 0u, 0,
        sample_sequence, sample_rows, sample_meta, sample_control, sample_every, sample_capacity);
}

#ifndef RAMSEY43_COMBINED_SOURCE
#include "ramsey43_features.cuh"
#include "ramsey43_dataset.cuh"
#endif
