#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>

#include <algorithm>
#include <climits>
#include <cstdint>

struct HotCacheParams {
    int32_t* query;
    int32_t* slot_for_id;
    int32_t* id_of_slot;
    int64_t* usage;
    int64_t* step;
    int32_t* src;
    int32_t* dst;
    int64_t* num_copy;
    int32_t* scores;
    int64_t* score_epochs;
    uint8_t* hot;
    int32_t* hot_count;
    int64_t* lru_stats;
    int64_t* hot_stats;
    int queries;
    int slots;
    int id_base;
    int hot_limit;
    int reserved_slots;
    int decay_calls;
    int min_score;
};

__device__ inline int hot_score(const HotCacheParams& p, int id, int64_t epoch) {
    const int64_t age = epoch - p.score_epochs[id];
    return p.scores[id] >> (age < 31 ? static_cast<int>(age) : 31);
}

template <int kThreads>
__device__ inline int hot_cache_min(int64_t key, int slot, int64_t* keys, int* slots) {
    keys[threadIdx.x] = key;
    slots[threadIdx.x] = slot;
    __syncthreads();
    for (int width = kThreads / 2; width > 0; width /= 2) {
        if (threadIdx.x < width) {
            const int other = threadIdx.x + width;
            if (keys[other] < keys[threadIdx.x] ||
                (keys[other] == keys[threadIdx.x] && slots[other] < slots[threadIdx.x])) {
                keys[threadIdx.x] = keys[other];
                slots[threadIdx.x] = slots[other];
            }
        }
        __syncthreads();
    }
    return slots[0];
}

template <int kQueries, int kThreads = 256>
__global__ void adaptive_hot_ensure_kernel(const __grid_constant__ HotCacheParams p) {
    __shared__ int ids[kQueries];
    __shared__ int first[kQueries];
    __shared__ int missing[kQueries];
    __shared__ int rank[kQueries];
    __shared__ int64_t keys[kThreads];
    __shared__ int positions[kThreads];
    __shared__ int64_t tick;
    __shared__ int64_t epoch;
    __shared__ int active_count;
    __shared__ int miss_count;
    __shared__ int protected_count;
    __shared__ int demotions;
    __shared__ int promotions;
    __shared__ int protected_hits;
    if (threadIdx.x == 0) {
        tick = ++p.step[0];
        epoch = tick / p.decay_calls;
        active_count = miss_count = demotions = promotions = protected_hits = 0;
        protected_count = p.hot_count[0];
    }
    for (int i = threadIdx.x; i < p.queries; i += kThreads) {
        ids[i] = p.query[i] + p.id_base;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < p.queries; i += kThreads) {
        bool unique = true;
        for (int j = 0; j < i; ++j) {
            unique = unique && ids[j] != ids[i];
        }
        first[i] = unique;
        const int slot = p.slot_for_id[ids[i]];
        missing[i] = unique && slot < 0;
        if (unique) {
            atomicAdd(&active_count, 1);
            if (slot < 0) {
                atomicAdd(&miss_count, 1);
            } else {
                p.usage[slot] = tick;
                if (p.hot[slot]) {
                    atomicAdd(&protected_hits, 1);
                }
            }
            const int score = hot_score(p, ids[i], epoch);
            p.scores[ids[i]] = score < INT_MAX ? score + 1 : INT_MAX;
            p.score_epochs[ids[i]] = epoch;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < p.queries; i += kThreads) {
        int r = 0;
        for (int j = 0; j < p.queries; ++j) {
            r += missing[j] && ids[j] < ids[i];
        }
        rank[i] = r;
        if (missing[i]) {
            p.src[r] = ids[i] - p.id_base;
        }
    }
    // Expired hot owners lose protection; their payload stays cached until an actual miss needs it.
    for (int slot = threadIdx.x; slot < p.slots; slot += kThreads) {
        if (p.hot[slot]) {
            const int id = p.id_of_slot[slot];
            if (id < 0 || hot_score(p, id, epoch) < p.min_score) {
                p.hot[slot] = 0;
                atomicSub(&protected_count, 1);
                atomicAdd(&demotions, 1);
            }
        }
    }
    __syncthreads();
    for (int m = 0; m < miss_count; ++m) {
        int best_slot = INT_MAX;
        int64_t best_age = LLONG_MAX;
        for (int slot = threadIdx.x; slot < p.slots; slot += kThreads) {
            const int64_t age = p.usage[slot];
            const int tie_slot = slot < p.reserved_slots ? p.slots + slot : slot;
            if (!p.hot[slot] && age != tick &&
                (age < best_age || (age == best_age && tie_slot < best_slot))) {
                best_age = age;
                best_slot = tie_slot;
            }
        }
        // Prefer stable slots on age ties so hot experts need no payload relocation out of prefill buffers.
        const int victim_key = hot_cache_min<kThreads>(best_age, best_slot, keys, positions);
        const int victim = victim_key >= p.slots ? victim_key - p.slots : victim_key;
        if (threadIdx.x == 0) {
            const int old = p.id_of_slot[victim];
            if (old >= 0 && p.slot_for_id[old] == victim) {
                p.slot_for_id[old] = -1;
            }
            const int id = p.src[m] + p.id_base;
            p.id_of_slot[victim] = id;
            p.slot_for_id[id] = victim;
            p.usage[victim] = tick;
            p.dst[m] = victim;
        }
        __syncthreads();
    }
    // Promotion only changes protection, so every expert keeps its original weight bytes and slot address.
    for (int i = 0; i < p.queries; ++i) {
        const int slot = p.slot_for_id[ids[i]];
        if (!first[i] || slot < p.reserved_slots || p.hot[slot] ||
            p.hot_limit == 0 || p.scores[ids[i]] < p.min_score) {
            continue;
        }
        if (protected_count >= p.hot_limit) {
            int best_slot = INT_MAX;
            int64_t best_score = LLONG_MAX;
            for (int s = threadIdx.x; s < p.slots; s += kThreads) {
                if (p.hot[s]) {
                    const int score = hot_score(p, p.id_of_slot[s], epoch);
                    if (score < best_score || (score == best_score && s < best_slot)) {
                        best_score = score;
                        best_slot = s;
                    }
                }
            }
            const int weakest = hot_cache_min<kThreads>(best_score, best_slot, keys, positions);
            if (p.scores[ids[i]] <= keys[0]) {
                continue;
            }
            if (threadIdx.x == 0) {
                p.hot[weakest] = 0;
                --protected_count;
                ++demotions;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            p.hot[slot] = 1;
            ++protected_count;
            ++promotions;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < p.queries; i += kThreads) {
        p.query[i] = p.slot_for_id[ids[i]];
    }
    if (threadIdx.x == 0) {
        p.num_copy[0] = miss_count;
        p.hot_count[0] = protected_count;
        if (p.lru_stats != nullptr) {
            p.lru_stats[0] += active_count;
            p.lru_stats[1] += miss_count;
            p.lru_stats[2] += 1;
        }
        p.hot_stats[0] += protected_hits;
        p.hot_stats[1] += promotions;
        p.hot_stats[2] += demotions;
        p.hot_stats[3] += 1;
    }
}

template <int kQueries>
struct AdaptiveHotEnsure {
    static void run(tvm::ffi::TensorView query, tvm::ffi::TensorView slot_for_id,
                    tvm::ffi::TensorView id_of_slot, tvm::ffi::TensorView usage,
                    tvm::ffi::TensorView step, tvm::ffi::TensorView src, tvm::ffi::TensorView dst,
                    tvm::ffi::TensorView num_copy, tvm::ffi::TensorView scores,
                    tvm::ffi::TensorView epochs, tvm::ffi::TensorView hot,
                    tvm::ffi::TensorView hot_count, tvm::ffi::TensorView stats,
                    tvm::ffi::TensorView hot_stats, int id_base, int hot_limit,
                    int reserved_slots, int decay_calls, int min_score, bool collect_stats) {
        using namespace host;
        auto device = SymbolicDevice{};
        auto q = SymbolicSize{"queries"};
        auto ids = SymbolicSize{"total experts"};
        auto slots = SymbolicSize{"cache slots"};
        auto plan = SymbolicSize{"copy plan"};
        auto i32 = SymbolicDType{};
        auto i64 = SymbolicDType{};
        auto byte = SymbolicDType{};
        TensorMatcher({q}).with_dtype<int32_t>(i32).with_device<kDLCUDA, kDLROCM>(device).verify(query);
        TensorMatcher({ids}).with_dtype<int32_t>(i32).with_device<kDLCUDA, kDLROCM>(device).verify(slot_for_id).verify(scores);
        TensorMatcher({ids}).with_dtype<int64_t>(i64).with_device<kDLCUDA, kDLROCM>(device).verify(epochs);
        TensorMatcher({slots}).with_dtype<int32_t>(i32).with_device<kDLCUDA, kDLROCM>(device).verify(id_of_slot);
        TensorMatcher({slots}).with_dtype<int64_t>(i64).with_device<kDLCUDA, kDLROCM>(device).verify(usage);
        TensorMatcher({slots}).with_dtype<uint8_t>(byte).with_device<kDLCUDA, kDLROCM>(device).verify(hot);
        TensorMatcher({plan}).with_dtype<int32_t>(i32).with_device<kDLCUDA, kDLROCM>(device).verify(src).verify(dst);
        TensorMatcher({1}).with_dtype<int64_t>(i64).with_device<kDLCUDA, kDLROCM>(device).verify(step).verify(num_copy);
        TensorMatcher({1}).with_dtype<int32_t>(i32).with_device<kDLCUDA, kDLROCM>(device).verify(hot_count);
        TensorMatcher({3}).with_dtype<int64_t>(i64).with_device<kDLCUDA, kDLROCM>(device).verify(stats);
        TensorMatcher({4}).with_dtype<int64_t>(i64).with_device<kDLCUDA, kDLROCM>(device).verify(hot_stats);
        RuntimeCheck(q.unwrap() <= kQueries, "adaptive-hot query exceeds kernel capacity");
        RuntimeCheck(hot_limit >= 0 && hot_limit < slots.unwrap() && decay_calls > 0 && min_score > 0,
                     "invalid adaptive-hot policy settings");
        RuntimeCheck(reserved_slots >= 0 && reserved_slots <= slots.unwrap(), "invalid prefill reservation");
        const auto p = HotCacheParams{
            static_cast<int32_t*>(query.data_ptr()), static_cast<int32_t*>(slot_for_id.data_ptr()),
            static_cast<int32_t*>(id_of_slot.data_ptr()), static_cast<int64_t*>(usage.data_ptr()),
            static_cast<int64_t*>(step.data_ptr()), static_cast<int32_t*>(src.data_ptr()),
            static_cast<int32_t*>(dst.data_ptr()), static_cast<int64_t*>(num_copy.data_ptr()),
            static_cast<int32_t*>(scores.data_ptr()), static_cast<int64_t*>(epochs.data_ptr()),
            static_cast<uint8_t*>(hot.data_ptr()), static_cast<int32_t*>(hot_count.data_ptr()),
            collect_stats ? static_cast<int64_t*>(stats.data_ptr()) : nullptr,
            static_cast<int64_t*>(hot_stats.data_ptr()), static_cast<int>(q.unwrap()),
            static_cast<int>(slots.unwrap()), id_base, hot_limit, reserved_slots, decay_calls, min_score,
        };
        LaunchKernel(1, 256, device.unwrap())(adaptive_hot_ensure_kernel<kQueries>, p);
    }
};
