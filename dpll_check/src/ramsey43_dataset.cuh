// Neuro-Ramsey Fase 3: muestreo congelado y sondas independientes del DFS vivo.
// Se incluye despues de native.cu y features.cuh; no requiere headers CUDA.
// Cada sonda conserva el padre y el color candidato como prefijo inmutable.
// El presupuesto solo produce CENSORED: nunca demuestra UNSAT por agotarse.
#ifndef RAMSEY43_DATASET_CUH
#define RAMSEY43_DATASET_CUH

static const unsigned RAMSEY43_DATASET_PADDING = 0u;
static const unsigned RAMSEY43_DATASET_IMMEDIATE_K5 = 1u;
static const unsigned RAMSEY43_DATASET_REFUTED = 2u;
static const unsigned RAMSEY43_DATASET_CENSORED = 3u;
static const unsigned RAMSEY43_DATASET_SAT = 4u;
static const unsigned RAMSEY43_DATASET_INVALID = 5u;
static const unsigned RAMSEY43_DATASET_JOURNAL = 903u;

// El capturador admite padres con asignaciones arbitrarias, no solo prefijos
// lexicograficos. Comprobar el contrato completo evita que una sonda publique
// refutaciones o features a partir de filas asimetricas o corruptas. Todos los
// lanes participan en cada row(), aun cuando solo 11 poseen el banco alto.
__device__ __forceinline__ bool ramsey43_dataset_canonical(Ramsey43WarpGraph graph)
{
    unsigned lane = threadIdx.x & 31u;
    bool valid = ((graph.red0 | graph.blue0) & ~RAMSEY43_VERTICES) == 0ull
        && (graph.red0 & graph.blue0) == 0ull
        && ((graph.red0 | graph.blue0) & (1ull << lane)) == 0ull;
    if (lane < 11u) {
        valid = valid && ((graph.red1 | graph.blue1) & ~RAMSEY43_VERTICES) == 0ull
            && (graph.red1 & graph.blue1) == 0ull
            && ((graph.red1 | graph.blue1) & (1ull << (lane + 32u))) == 0ull;
    } else {
        valid = valid && graph.red1 == 0ull && graph.blue1 == 0ull;
    }
    for (unsigned vertex = 0u; vertex < 43u; ++vertex) {
        Ramsey43Bits red = ramsey43_row(graph.red0, graph.red1, vertex);
        Ramsey43Bits blue = ramsey43_row(graph.blue0, graph.blue1, vertex);
        valid = valid && ((graph.red0 >> vertex) & 1ull) == ((red >> lane) & 1ull)
            && ((graph.blue0 >> vertex) & 1ull) == ((blue >> lane) & 1ull);
        if (lane < 11u) {
            valid = valid
                && ((graph.red1 >> vertex) & 1ull) == ((red >> (lane + 32u)) & 1ull)
                && ((graph.blue1 >> vertex) & 1ull) == ((blue >> (lane + 32u)) & 1ull);
        }
    }
    return __all_sync(RAMSEY43_FULL_WARP, valid);
}

// Entrada: rows[capacity][86] son snapshots compactos e inmutables; meta[...][8]
// = {uid, sequence_lo, sequence_hi, mutable_depth, chosen_edge, phase, 0, 0}.
// No consulta policy del
// motor activo. La captura en el mismo stream debe terminar antes de lanzar.
// Salida: X FP16 bits [capacity][32][32], IDs canonicos [capacity][32], valid
// [capacity]. Las aristas libres se enumeran lexicograficamente sin reasignar
// el padre: cada fila X describe exactamente su candidato en el mismo estado.
extern "C" __global__ void ramsey43_dataset_pack(
    const Ramsey43Bits* rows, const unsigned* meta,
    unsigned count, unsigned capacity, unsigned short* x,
    unsigned short* ids, unsigned* valid)
{
    if (blockDim.x != 256u || blockDim.y != 1u || blockDim.z != 1u) return;
    unsigned lane = threadIdx.x & 31u;
    unsigned slot = blockIdx.x * 8u + (threadIdx.x >> 5);
    if (slot >= capacity) return;
    Ramsey43Bits x_base = (Ramsey43Bits)slot * 1024ull;
    Ramsey43Bits id_base = (Ramsey43Bits)slot * 32ull;
    for (unsigned item = lane; item < 1024u; item += 32u) x[x_base + item] = 0u;
    ids[id_base + lane] = 903u;
    if (lane == 0u) valid[slot] = 0u;
    __syncwarp(RAMSEY43_FULL_WARP);
    if (slot >= count) return;

    Ramsey43Bits m = (Ramsey43Bits)slot * 8ull;
    unsigned uid = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? meta[m] : 0u, 0);
    unsigned depth = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? meta[m + 3ull] : 0u, 0);
    if (uid == RAMSEY43_POLICY_INVALID_UID || depth > 903u) return;
    const Ramsey43Bits* source = rows + (Ramsey43Bits)slot * 86ull;
    Ramsey43WarpGraph graph;
    graph.red0 = source[lane];
    graph.blue0 = source[43u + lane];
    graph.red1 = lane < 11u ? source[lane + 32u] : 0ull;
    graph.blue1 = lane < 11u ? source[75u + lane] : 0ull;
    if (!ramsey43_dataset_canonical(graph) || !ramsey43_degrees_viable(graph)) return;
    if (ramsey43_find_k5(graph.red0, graph.red1) != 0ull) return;
    if (ramsey43_find_k5(graph.blue0, graph.blue1) != 0ull) return;

    unsigned red0 = (unsigned)__popcll(graph.red0);
    unsigned blue0 = (unsigned)__popcll(graph.blue0);
    unsigned red1 = (unsigned)__popcll(graph.red1);
    unsigned blue1 = (unsigned)__popcll(graph.blue1);
    Ramsey43FeatureGlobals globals;
    globals.red_edges = ramsey43_neuro_sum(red0 + red1) / 2u;
    globals.blue_edges = ramsey43_neuro_sum(blue0 + blue1) / 2u;
    globals.unknown_edges = 903u - globals.red_edges - globals.blue_edges;
    globals.min_red = ramsey43_neuro_warp_min(lane < 11u ? ramsey43_neuro_min(red0, red1) : red0);
    globals.max_red = ramsey43_neuro_warp_max(lane < 11u ? ramsey43_neuro_max(red0, red1) : red0);
    globals.min_blue = ramsey43_neuro_warp_min(lane < 11u ? ramsey43_neuro_min(blue0, blue1) : blue0);
    globals.max_blue = ramsey43_neuro_warp_max(lane < 11u ? ramsey43_neuro_max(blue0, blue1) : blue0);
    Ramsey43Bits remaining0 = ~(graph.red0 | graph.blue0) & ramsey43_after(lane);
    Ramsey43Bits remaining1 = lane < 11u
        ? ~(graph.red1 | graph.blue1) & ramsey43_after(lane + 32u) : 0ull;
    for (unsigned candidate = 0u; candidate < 32u; ++candidate) {
        unsigned local_edge = ramsey43_neuro_local_candidate(remaining0, remaining1, lane);
        unsigned edge = ramsey43_neuro_warp_min(local_edge);
        if (edge == 903u) break;
        if (lane == 0u) ids[id_base + candidate] = (unsigned short)edge;
        ramsey43_write_edge_features(graph, edge, globals, depth,
            x + x_base + (Ramsey43Bits)candidate * 32ull);
        if (local_edge == edge) {
            if (remaining0 != 0ull) remaining0 &= remaining0 - 1ull;
            else remaining1 &= remaining1 - 1ull;
        }
    }
    if (lane == 0u) valid[slot] = 1u;
}

// Cada warp visita una rama fija (candidato,color) del snapshot: 64 warps por
// padre, distribuibles entre CTAs sin barreras CTA ni arrays locales por hilo.
// journal[count*64][903] u16 es temporal GLOBAL y cada warp escribe su slice.
// costs/status tienen forma [count][32][2], color rojo primero.
// Coste = comprobacion inicial del color + transiciones DFS (descenso o pop).
// REFUTED publica ese coste exacto solo tras agotar todo el subarbol; CENSORED
// publica horizon como cota inferior, no como un coste de refutacion conocido.
// SAT significa que se observo una asignacion completa sin K5, NO refutacion.
// Esta sonda no publica al found/model del motor y no altera rows ni su journal.
extern "C" __global__ __launch_bounds__(256, 3) void ramsey43_dataset_probe(
    const Ramsey43Bits* rows, const unsigned short* ids, const unsigned* valid,
    unsigned count, unsigned horizon, Ramsey43Frame* journal,
    unsigned* costs, unsigned* status)
{
    if (blockDim.x != 256u || blockDim.y != 1u || blockDim.z != 1u) return;
    unsigned lane = threadIdx.x & 31u;
    Ramsey43Bits probe = (Ramsey43Bits)blockIdx.x * 8ull + (threadIdx.x >> 5);
    if (probe >= (Ramsey43Bits)count * 64ull) return;
    unsigned slot = (unsigned)(probe / 64ull);
    unsigned candidate = (unsigned)((probe / 2ull) & 31ull);
    unsigned fixed_colour = (unsigned)(probe & 1ull);
    if (lane == 0u) {
        costs[probe] = 0u;
        status[probe] = RAMSEY43_DATASET_PADDING;
    }
    unsigned parent_valid = __shfl_sync(RAMSEY43_FULL_WARP,
        lane == 0u ? valid[slot] : 0u, 0);
    unsigned edge = __shfl_sync(RAMSEY43_FULL_WARP,
        lane == 0u ? ids[(Ramsey43Bits)slot * 32ull + candidate] : 0u, 0);
    if (edge == 903u) return;
    if (parent_valid != 1u || edge >= 903u || horizon < 32u || horizon > 4096u) {
        if (lane == 0u) status[probe] = RAMSEY43_DATASET_INVALID;
        return;
    }
    const Ramsey43Bits* source = rows + (Ramsey43Bits)slot * 86ull;
    Ramsey43WarpGraph graph;
    graph.red0 = source[lane];
    graph.blue0 = source[43u + lane];
    graph.red1 = lane < 11u ? source[lane + 32u] : 0ull;
    graph.blue1 = lane < 11u ? source[75u + lane] : 0ull;
    unsigned a, b;
    ramsey43_edge(edge, 0u, 1u, a, b);
    if (ramsey43_fixed(graph, a, b)) {
        if (lane == 0u) status[probe] = RAMSEY43_DATASET_INVALID;
        return;
    }
    bool direct_k5 = ramsey43_new_k5(graph, a, b, fixed_colour);
    unsigned allowed = ramsey43_allowed_colours(graph, a, b);
    unsigned work = 1u;
    unsigned result = RAMSEY43_DATASET_CENSORED;
    if (direct_k5) {
        result = RAMSEY43_DATASET_IMMEDIATE_K5;
    } else if ((allowed & (1u << fixed_colour)) == 0u) {
        result = RAMSEY43_DATASET_REFUTED;
    } else {
        // La primera decision no pertenece al journal: se conserva en todos
        // los padres restaurados y nunca se explora el color complementario.
        ramsey43_set_edge(graph, a, b, fixed_colour);
        Ramsey43Frame* trail = journal + probe * (Ramsey43Bits)RAMSEY43_DATASET_JOURNAL;
        unsigned depth = 0u, backtrack = 0u;
        while (work < horizon) {
            ++work;
            unsigned colour;
            bool alternate_consumed = false, is_forced = false;
            if (backtrack != 0u) {
                if (depth == 0u) {
                    result = RAMSEY43_DATASET_REFUTED;
                    break;
                }
                --depth;
                unsigned frame = __shfl_sync(RAMSEY43_FULL_WARP,
                    lane == 0u ? trail[depth] : 0u, 0);
                edge = frame & RAMSEY43_FRAME_EDGE;
                colour = (frame & RAMSEY43_FRAME_COLOUR) != 0u ? 1u : 0u;
                alternate_consumed = (frame & RAMSEY43_FRAME_ALT) != 0u;
                is_forced = (frame & RAMSEY43_FRAME_FORCED) != 0u;
                if (edge >= 903u || (frame & RAMSEY43_FRAME_RESERVED) != 0u
                    || (is_forced && !alternate_consumed)) {
                    result = RAMSEY43_DATASET_INVALID;
                    break;
                }
                ramsey43_edge(edge, 0u, 1u, a, b);
                ramsey43_set_edge(graph, a, b, 2u);
                if (is_forced || alternate_consumed) continue;
                colour ^= 1u;
                alternate_consumed = true;
                allowed = ramsey43_allowed_colours(graph, a, b);
                if ((allowed & (1u << colour)) == 0u) continue;
            } else {
                edge = ramsey43_pick_free_edge(graph);
                if (edge == 903u) {
                    bool safe = ramsey43_complete(graph);
                    if (safe) safe = ramsey43_find_k5(graph.red0, graph.red1) == 0ull;
                    if (safe) safe = ramsey43_find_k5(graph.blue0, graph.blue1) == 0ull;
                    result = safe ? RAMSEY43_DATASET_SAT : RAMSEY43_DATASET_INVALID;
                    break;
                }
                if (depth >= RAMSEY43_DATASET_JOURNAL) {
                    result = RAMSEY43_DATASET_INVALID;
                    break;
                }
                ramsey43_edge(edge, 0u, 1u, a, b);
                allowed = ramsey43_allowed_colours(graph, a, b);
                if (allowed == 0u) {
                    backtrack = 1u;
                    continue;
                }
                is_forced = allowed == 1u || allowed == 2u;
                colour = allowed == 2u ? 1u : 0u;
                alternate_consumed = is_forced;
            }
            ramsey43_set_edge(graph, a, b, colour);
            if (lane == 0u) trail[depth] = ramsey43_frame(
                edge, colour, alternate_consumed, is_forced);
            ++depth;
            backtrack = 0u;
        }
    }
    if (lane == 0u) {
        costs[probe] = work;
        status[probe] = result;
    }
}

#endif
