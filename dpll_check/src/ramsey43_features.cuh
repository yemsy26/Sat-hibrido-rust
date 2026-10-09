// Neuro-Ramsey Fase 2: snapshots y consumo separados del DFS. Este header se
// incluye despues de ramsey43_native.cu o se concatena al SOURCE de NVRTC.
// No usa headers del toolkit, arrays por hilo ni barreras entre warps.
#ifndef RAMSEY43_FEATURES_CUH
#define RAMSEY43_FEATURES_CUH

static const unsigned RAMSEY43_POLICY_IDLE = 0u;
static const unsigned RAMSEY43_POLICY_WAITING = 1u;
static const unsigned RAMSEY43_POLICY_READY = 2u;
static const unsigned RAMSEY43_POLICY_FALLBACK = 0xffffu;
static const unsigned RAMSEY43_POLICY_INVALID_UID = 0xffffffffu;

// Conversion IEEE binary32 positivo -> binary16, round-to-nearest ties-to-even.
// Las features son 0 o valores finitos de [1/903,1]. La rama subnormal permite
// tambien validar la conversion de forma independiente sin headers CUDA.
__device__ __forceinline__ unsigned short ramsey43_positive_half_rn(float value)
{
    unsigned bits = __float_as_uint(value);
    unsigned exponent = (bits >> 23) & 0xffu;
    unsigned mantissa = bits & 0x7fffffu;
    if (exponent == 0u) return 0u;
    int half_exponent = (int)exponent - 127 + 15;
    if (half_exponent >= 31) return (unsigned short)0x7c00u;
    if (half_exponent <= 0) {
        if (half_exponent < -10) return 0u;
        unsigned significand = 0x800000u | mantissa;
        unsigned shift = (unsigned)(14 - half_exponent);
        unsigned rounded = significand >> shift;
        unsigned remainder = significand & ((1u << shift) - 1u);
        unsigned halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
        return (unsigned short)rounded;
    }
    unsigned rounded = ((unsigned)half_exponent << 10) | (mantissa >> 13);
    unsigned remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) ++rounded;
    return (unsigned short)rounded;
}

__device__ __forceinline__ unsigned ramsey43_neuro_min(unsigned a, unsigned b)
{
    return a < b ? a : b;
}

__device__ __forceinline__ unsigned ramsey43_neuro_max(unsigned a, unsigned b)
{
    return a > b ? a : b;
}

// Todos los lanes participan; cada resultado se difunde desde lane0.
__device__ __forceinline__ unsigned ramsey43_neuro_sum(unsigned value)
{
    for (unsigned delta = 16u; delta != 0u; delta >>= 1)
        value += __shfl_down_sync(RAMSEY43_FULL_WARP, value, delta);
    return __shfl_sync(RAMSEY43_FULL_WARP, value, 0);
}

__device__ __forceinline__ unsigned ramsey43_neuro_warp_min(unsigned value)
{
    for (unsigned delta = 16u; delta != 0u; delta >>= 1)
        value = ramsey43_neuro_min(value, __shfl_down_sync(RAMSEY43_FULL_WARP, value, delta));
    return __shfl_sync(RAMSEY43_FULL_WARP, value, 0);
}

__device__ __forceinline__ unsigned ramsey43_neuro_warp_max(unsigned value)
{
    for (unsigned delta = 16u; delta != 0u; delta >>= 1)
        value = ramsey43_neuro_max(value, __shfl_down_sync(RAMSEY43_FULL_WARP, value, delta));
    return __shfl_sync(RAMSEY43_FULL_WARP, value, 0);
}

// Candidato propio del lane, en orden canonico lexicografico. Como la fila
// baja precede siempre a la alta, solo se consulta la alta si la baja esta vacia.
__device__ __forceinline__ unsigned ramsey43_neuro_local_candidate(
    Ramsey43Bits remaining0, Ramsey43Bits remaining1, unsigned lane)
{
    if (remaining0 != 0ull) {
        unsigned b = (unsigned)(__ffsll((long long)remaining0) - 1);
        return lane * (85u - lane) / 2u + b - lane - 1u;
    }
    if (lane < 11u && remaining1 != 0ull) {
        unsigned a = lane + 32u;
        unsigned b = (unsigned)(__ffsll((long long)remaining1) - 1);
        return a * (85u - a) / 2u + b - a - 1u;
    }
    return 903u;
}

struct Ramsey43FeatureGlobals {
    unsigned red_edges;
    unsigned blue_edges;
    unsigned unknown_edges;
    unsigned min_red;
    unsigned max_red;
    unsigned min_blue;
    unsigned max_blue;
};

// Extractor device por arista: exactamente 32 lanes participan en cada row().
// La salida es una fila coalescida de 32 half. Solo usa escalares y los cuatro
// bancos de adyacencia del grafo; globals se calcula una vez por snapshot.
__device__ __forceinline__ void ramsey43_write_edge_features(
    Ramsey43WarpGraph graph, unsigned edge, Ramsey43FeatureGlobals globals,
    unsigned depth, unsigned short* x_row)
{
    unsigned lane = threadIdx.x & 31u;
    unsigned a, b;
    ramsey43_edge(edge, 0u, 1u, a, b);
    Ramsey43Bits ra = ramsey43_row(graph.red0, graph.red1, a);
    Ramsey43Bits rb = ramsey43_row(graph.red0, graph.red1, b);
    Ramsey43Bits ba = ramsey43_row(graph.blue0, graph.blue1, a);
    Ramsey43Bits bb = ramsey43_row(graph.blue0, graph.blue1, b);
    Ramsey43Bits ua = RAMSEY43_VERTICES & ~(1ull << a) & ~(ra | ba);
    Ramsey43Bits ub = RAMSEY43_VERTICES & ~(1ull << b) & ~(rb | bb);
    unsigned da_red = (unsigned)__popcll(ra), db_red = (unsigned)__popcll(rb);
    unsigned da_blue = (unsigned)__popcll(ba), db_blue = (unsigned)__popcll(bb);
    unsigned da_unknown = (unsigned)__popcll(ua), db_unknown = (unsigned)__popcll(ub);
    float feature = 0.0f;
    switch (lane) {
        case 0u: feature = (float)globals.red_edges / 903.0f; break;
        case 1u: feature = (float)globals.blue_edges / 903.0f; break;
        case 2u: feature = (float)globals.unknown_edges / 903.0f; break;
        case 3u: feature = (float)depth / 903.0f; break;
        case 4u: feature = (float)globals.min_red / 42.0f; break;
        case 5u: feature = (float)globals.max_red / 42.0f; break;
        case 6u: feature = (float)globals.min_blue / 42.0f; break;
        case 7u: feature = (float)globals.max_blue / 42.0f; break;
        case 8u: feature = (float)ramsey43_neuro_min(da_red, db_red) / 42.0f; break;
        case 9u: feature = (float)ramsey43_neuro_max(da_red, db_red) / 42.0f; break;
        case 10u: feature = (float)ramsey43_neuro_min(da_blue, db_blue) / 42.0f; break;
        case 11u: feature = (float)ramsey43_neuro_max(da_blue, db_blue) / 42.0f; break;
        case 12u: feature = (float)ramsey43_neuro_min(da_unknown, db_unknown) / 42.0f; break;
        case 13u: feature = (float)ramsey43_neuro_max(da_unknown, db_unknown) / 42.0f; break;
        case 14u: feature = (float)__popcll(ra & rb) / 42.0f; break;
        case 15u: feature = (float)__popcll(ba & bb) / 42.0f; break;
        case 16u: feature = (float)__popcll(ua & ub) / 42.0f; break;
        case 17u: feature = (float)(__popcll(ra & bb) + __popcll(ba & rb)) / 84.0f; break;
        case 18u: feature = (float)(__popcll(ra & ub) + __popcll(ua & rb)) / 84.0f; break;
        case 19u: feature = (float)(__popcll(ba & ub) + __popcll(ua & bb)) / 84.0f; break;
        default: break;
    }
    x_row[lane] = ramsey43_positive_half_rn(feature);
}

// X[batch_capacity][32][32] contiene bits IEEE half, no un tipo CUDA half.
// ids[batch_capacity][32], meta[batch_capacity][4] = {UID, gen_lo, gen_hi, depth}.
// Lanzar ceil(batch_capacity/8) bloques de 256 para inicializar tambien padding.
// El DFS ya ha marcado WAITING y queda suspendido mientras vive su snapshot.
extern "C" __global__ void ramsey43_pack_features(
    const Ramsey43Bits* rows, const unsigned* control, const unsigned* policy,
    const Ramsey43Bits* generation, unsigned universe_count, unsigned first,
    unsigned count, unsigned batch_capacity, unsigned short* x,
    unsigned short* ids, unsigned* meta)
{
    if (blockDim.x != 256u || blockDim.y != 1u || blockDim.z != 1u) return;
    unsigned lane = threadIdx.x & 31u;
    unsigned slot = blockIdx.x * 8u + (threadIdx.x >> 5);
    if (slot >= batch_capacity) return;
    Ramsey43Bits x_base = (Ramsey43Bits)slot * 1024ull;
    Ramsey43Bits id_base = (Ramsey43Bits)slot * 32ull;
    for (unsigned item = lane; item < 1024u; item += 32u) x[x_base + item] = 0u;
    ids[id_base + lane] = 903u;
    if (lane == 0u) {
        Ramsey43Bits m = (Ramsey43Bits)slot * 4ull;
        meta[m] = RAMSEY43_POLICY_INVALID_UID;
        meta[m + 1ull] = meta[m + 2ull] = meta[m + 3ull] = 0u;
    }
    // ids fue inicializado por todos los lanes y despues lo escribe lane0.
    __syncwarp(RAMSEY43_FULL_WARP);
    if (slot >= count || first >= universe_count || slot >= universe_count - first) return;
    unsigned universe = first + slot;
    const unsigned* ctl = control + (Ramsey43Bits)universe * 5ull;
    const unsigned* pol = policy + (Ramsey43Bits)universe * 3ull;
    unsigned state = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? pol[1] : 0u, 0);
    unsigned initialized = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[0] : 0u, 0);
    unsigned depth = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[1] : 0u, 0);
    unsigned floor = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[2] : 0u, 0);
    unsigned status = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[3] : 0u, 0);
    unsigned backtrack = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[4] : 0u, 0);
    if (state != RAMSEY43_POLICY_WAITING || initialized == 0u
        || status != RAMSEY43_LIVE || backtrack != 0u || floor != 0u || depth > 903u) return;

    const Ramsey43Bits* source = rows + (Ramsey43Bits)universe * 86ull;
    Ramsey43WarpGraph graph;
    graph.red0 = source[lane];
    graph.blue0 = source[43u + lane];
    graph.red1 = lane < 11u ? source[lane + 32u] : 0ull;
    graph.blue1 = lane < 11u ? source[75u + lane] : 0ull;
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

    // Cada lane escribe una feature; todos los row() y las reducciones se
    // ejecutan antes de la seleccion por indice de feature, con 32 lanes vivos.
    for (unsigned candidate_index = 0u; candidate_index < 32u; ++candidate_index) {
        unsigned local_candidate = ramsey43_neuro_local_candidate(remaining0, remaining1, lane);
        unsigned edge = ramsey43_neuro_warp_min(local_candidate);
        if (edge == 903u) break;
        if (lane == 0u) ids[id_base + candidate_index] = (unsigned short)edge;
        ramsey43_write_edge_features(graph, edge, globals, depth,
            x + x_base + (Ramsey43Bits)candidate_index * 32ull);
        if (local_candidate == edge) {
            if (remaining0 != 0ull) remaining0 &= remaining0 - 1ull;
            else remaining1 &= remaining1 - 1ull;
        }
    }
    if (lane == 0u) {
        Ramsey43Bits snapshot_generation = generation[universe];
        Ramsey43Bits m = (Ramsey43Bits)slot * 4ull;
        meta[m] = universe;
        meta[m + 1ull] = (unsigned)snapshot_generation;
        meta[m + 2ull] = (unsigned)(snapshot_generation >> 32);
        meta[m + 3ull] = depth;
    }
}

__device__ __forceinline__ bool ramsey43_neuro_finite(float value)
{
    return (__float_as_uint(value) & 0x7fffffffu) < 0x7f800000u;
}

// Orden total de candidatos finitos: coste, ID canonico, color (rojo primero).
__device__ __forceinline__ bool ramsey43_neuro_better(
    float score, unsigned choice, unsigned valid,
    float best_score, unsigned best_choice, unsigned best_valid)
{
    if (valid == 0u) return false;
    if (best_valid == 0u || score < best_score) return true;
    if (score > best_score) return false;
    unsigned edge = choice & RAMSEY43_FRAME_EDGE;
    unsigned best_edge = best_choice & RAMSEY43_FRAME_EDGE;
    return edge < best_edge || (edge == best_edge && choice < best_choice);
}

// Y[count][32][2] FP32, IDs/meta del pack. No fuerza ni poda: la eleccion READY
// se valida otra vez con el oraculo exacto dentro del siguiente DFS.
// counters = {queries (DFS), responses, fallbacks}; este kernel escribe los dos ultimos.
extern "C" __global__ void ramsey43_apply_scores(
    const Ramsey43Bits* rows, const unsigned* control, unsigned* policy,
    const Ramsey43Bits* generation, unsigned universe_count, unsigned first,
    unsigned count, const unsigned short* ids, const unsigned* meta,
    const float* scores, Ramsey43Bits* policy_counters)
{
    if (blockDim.x != 256u || blockDim.y != 1u || blockDim.z != 1u) return;
    unsigned lane = threadIdx.x & 31u;
    unsigned slot = blockIdx.x * 8u + (threadIdx.x >> 5);
    if (slot >= count || first >= universe_count || slot >= universe_count - first) return;
    unsigned universe = first + slot;
    unsigned* pol = policy + (Ramsey43Bits)universe * 3ull;
    unsigned state = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? pol[1] : 0u, 0);
    if (state != RAMSEY43_POLICY_WAITING) return;
    const unsigned* ctl = control + (Ramsey43Bits)universe * 5ull;
    Ramsey43Bits m = (Ramsey43Bits)slot * 4ull;
    unsigned uid = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? meta[m] : 0u, 0);
    unsigned low = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? meta[m + 1ull] : 0u, 0);
    unsigned high = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? meta[m + 2ull] : 0u, 0);
    unsigned request_depth = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? meta[m + 3ull] : 0u, 0);
    unsigned initialized = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[0] : 0u, 0);
    unsigned depth = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[1] : 0u, 0);
    unsigned floor = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[2] : 0u, 0);
    unsigned status = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[3] : 0u, 0);
    unsigned backtrack = __shfl_sync(RAMSEY43_FULL_WARP, lane == 0u ? ctl[4] : 0u, 0);
    Ramsey43Bits current_generation = __shfl_sync(RAMSEY43_FULL_WARP,
        lane == 0u ? generation[universe] : 0ull, 0);
    Ramsey43Bits request_generation = (Ramsey43Bits)low | ((Ramsey43Bits)high << 32);
    unsigned choice = RAMSEY43_POLICY_FALLBACK;
    bool valid_snapshot = uid == universe && request_generation == current_generation
        && request_depth == depth && initialized != 0u && depth <= 903u
        && floor == 0u && status == RAMSEY43_LIVE && backtrack == 0u;
    if (valid_snapshot) {
        const Ramsey43Bits* source = rows + (Ramsey43Bits)universe * 86ull;
        Ramsey43WarpGraph graph;
        graph.red0 = source[lane];
        graph.blue0 = source[43u + lane];
        graph.red1 = lane < 11u ? source[lane + 32u] : 0ull;
        graph.blue1 = lane < 11u ? source[75u + lane] : 0ull;
        unsigned edge = ids[(Ramsey43Bits)slot * 32ull + lane];
        bool padding = edge == 903u;
        bool invalid = !padding && edge >= 903u;
        // row() tolera endpoints distintos por lane, pero todos participan.
        unsigned a, b;
        ramsey43_edge(edge < 903u ? edge : 0u, 0u, 1u, a, b);
        Ramsey43Bits occupied = ramsey43_row(graph.red0 | graph.blue0,
            graph.red1 | graph.blue1, a);
        if (!padding && ((occupied >> b) & 1ull) != 0ull) invalid = true;
        for (unsigned other_lane = 0u; other_lane < 32u; ++other_lane) {
            unsigned other = __shfl_sync(RAMSEY43_FULL_WARP, edge, other_lane);
            if (!padding && other_lane < lane && other == edge) invalid = true;
        }
        Ramsey43Bits score_base = ((Ramsey43Bits)slot * 32ull + lane) * 2ull;
        float red_score = scores[score_base];
        float blue_score = scores[score_base + 1ull];
        if (!padding && (!ramsey43_neuro_finite(red_score) || !ramsey43_neuro_finite(blue_score)))
            invalid = true;
        unsigned valid = !padding && !invalid ? 1u : 0u;
        unsigned colour = blue_score < red_score ? 1u : 0u;
        float best_score = colour == 0u ? red_score : blue_score;
        unsigned best_choice = edge | (colour << 10);
        bool all_valid = __all_sync(RAMSEY43_FULL_WARP, !invalid);
        for (unsigned delta = 16u; delta != 0u; delta >>= 1) {
            float other_score = __shfl_down_sync(RAMSEY43_FULL_WARP, best_score, delta);
            unsigned other_choice = __shfl_down_sync(RAMSEY43_FULL_WARP, best_choice, delta);
            unsigned other_valid = __shfl_down_sync(RAMSEY43_FULL_WARP, valid, delta);
            if (ramsey43_neuro_better(other_score, other_choice, other_valid,
                best_score, best_choice, valid)) {
                best_score = other_score;
                best_choice = other_choice;
                valid = other_valid;
            }
        }
        unsigned winner_valid = __shfl_sync(RAMSEY43_FULL_WARP, valid, 0);
        unsigned winner_choice = __shfl_sync(RAMSEY43_FULL_WARP, best_choice, 0);
        if (all_valid && winner_valid != 0u) choice = winner_choice;
    }
    if (lane == 0u) {
        pol[2] = choice;
        __threadfence();
        pol[1] = RAMSEY43_POLICY_READY;
        atomicAdd(policy_counters + 1, 1ull);
        if (choice == RAMSEY43_POLICY_FALLBACK) atomicAdd(policy_counters + 2, 1ull);
    }
}

#endif
