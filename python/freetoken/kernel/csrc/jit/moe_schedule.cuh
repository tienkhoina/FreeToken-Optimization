#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>

#include <climits>
#include <cstdint>

__global__ void moe_cpu_doorbell_kernel(const int64_t* active, int64_t* done, int64_t* ready, int slot) {
    if (threadIdx.x == 0) {
        const bool work = *active > 0;
        *reinterpret_cast<volatile int64_t*>(done + slot) = work ? 0 : 1;
        __threadfence_system();
        *reinterpret_cast<volatile int64_t*>(ready + slot) = work ? 1 : 0;
        __threadfence_system();
    }
}

struct MoeCpuDoorbell {
    static void run(tvm::ffi::TensorView active, tvm::ffi::TensorView done, tvm::ffi::TensorView ready, int slot) {
        using namespace host;
        RuntimeCheck(active.ndim() == 1 && active.size(0) == 1 && active.device().device_type == kDLCUDA,
                     "CPU branch count must be a CUDA scalar tensor");
        RuntimeCheck(done.device().device_type == kDLCPU && ready.device().device_type == kDLCPU &&
                     done.size(0) == ready.size(0) && slot >= 0 && slot < done.size(0), "invalid CPU handshake buffers");
        int64_t* dp = nullptr;
        int64_t* rp = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dp), done.data_ptr(), 0));
        CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&rp), ready.data_ptr(), 0));
        LaunchKernel(1, 32, active.device())(moe_cpu_doorbell_kernel, static_cast<const int64_t*>(active.data_ptr()), dp, rp, slot);
    }
};

struct MoeScheduleParams {
    const int32_t* ids;
    const float* weights;
    int32_t* hit_ids;
    float* hit_weights;
    int32_t* fetch_ids;
    float* fetch_weights;
    int32_t* cpu_ids;
    int32_t* map;
    int32_t* owners;
    int64_t* usage;
    int64_t* step;
    int32_t* source;
    int32_t* destination;
    int64_t* copies;
    int64_t* counts;
    int64_t* stats;
    const int32_t* fraction;
    const int32_t* fetch_table;
    const int32_t* valid_tokens;
    int tokens;
    int top_k;
    int experts;
    int slots;
    int evict_limit;
    int base;
};

__global__ void moe_schedule_kernel(const __grid_constant__ MoeScheduleParams p) {
    extern __shared__ int scratch[];
    int* histogram = scratch;
    int* state = histogram + p.experts;
    int* targets = state + p.experts;
    __shared__ int64_t ages[256];
    __shared__ int positions[256];
    __shared__ int64_t tick;
    __shared__ int nhit, nmiss, nfetch;
    const int limit = p.valid_tokens == nullptr ? p.tokens : min(p.tokens, *p.valid_tokens);
    if (threadIdx.x == 0) {
        tick = ++p.step[0];
        nhit = nmiss = nfetch = 0;
    }
    for (int e = threadIdx.x; e < p.experts; e += blockDim.x) {
        histogram[e] = 0;
        state[e] = 0;
        targets[e] = p.slots;
    }
    __syncthreads();
    for (int r = threadIdx.x; r < limit * p.top_k; r += blockDim.x) {
        const int e = p.ids[r];
        if (e >= 0 && e < p.experts) atomicAdd(histogram + e, 1);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < p.experts; e += blockDim.x) {
        if (histogram[e] != 0) {
            const int slot = p.map[p.base + e];
            if (slot >= 0) {
                state[e] = 1;
                targets[e] = slot;
                p.usage[slot] = tick;
                atomicAdd(&nhit, 1);
            } else {
                atomicAdd(&nmiss, 1);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int fraction = max(0, min(65536, p.fraction[0]));
        nfetch = min(nmiss, static_cast<int>((static_cast<int64_t>(nmiss) * fraction + 32768) >> 16));
        if (limit == 1 && p.fetch_table[nmiss] >= 0) nfetch = min(nmiss, p.fetch_table[nmiss]);
        nfetch = min(nfetch, p.slots - nhit);
        int selected = 0;
        for (int e = 0; e < p.experts; ++e) {
            if (histogram[e] && state[e] == 0 && selected < nfetch) {
                state[e] = 2;
                p.source[selected++] = e;
            }
        }
    }
    __syncthreads();
    for (int m = 0; m < nfetch; ++m) {
        int candidate = INT_MAX;
        int64_t age = LLONG_MAX;
        for (int s = threadIdx.x; s < p.evict_limit; s += blockDim.x) {
            const int64_t u = p.usage[s];
            if (u != tick && (u < age || (u == age && s < candidate))) {
                age = u;
                candidate = s;
            }
        }
        ages[threadIdx.x] = age;
        positions[threadIdx.x] = candidate;
        __syncthreads();
        for (int width = 128; width > 0; width /= 2) {
            if (threadIdx.x < width) {
                const int other = threadIdx.x + width;
                if (ages[other] < ages[threadIdx.x] ||
                    (ages[other] == ages[threadIdx.x] && positions[other] < positions[threadIdx.x])) {
                    ages[threadIdx.x] = ages[other];
                    positions[threadIdx.x] = positions[other];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const int victim = positions[0];
            const int old = p.owners[victim];
            if (old >= 0 && p.map[old] == victim) p.map[old] = -1;
            const int expert = p.source[m];
            p.owners[victim] = p.base + expert;
            p.map[p.base + expert] = victim;
            p.usage[victim] = tick;
            p.destination[m] = victim;
            targets[expert] = victim;
        }
        __syncthreads();
    }
    for (int r = threadIdx.x; r < p.tokens * p.top_k; r += blockDim.x) {
        const int e = p.ids[r];
        const bool valid = r < limit * p.top_k && e >= 0 && e < p.experts;
        const int branch = valid ? state[e] : -1;
        const float w = valid ? p.weights[r] : 0.0f;
        // The immutable zero row is outside the cache ledger, so masked routes never read an evicted slot.
        p.hit_ids[r] = branch == 1 ? targets[e] : p.slots;
        p.fetch_ids[r] = branch == 2 ? targets[e] : p.slots;
        p.hit_weights[r] = branch == 1 ? w : 0.0f;
        p.fetch_weights[r] = branch == 2 ? w : 0.0f;
        p.cpu_ids[r] = branch == 0 ? e : -1;
    }
    if (threadIdx.x == 0) {
        p.copies[0] = nfetch;
        p.counts[0] = nhit;
        p.counts[1] = nmiss;
        p.counts[2] = nfetch;
        p.counts[3] = nmiss - nfetch;
        p.stats[0] += nhit;
        p.stats[1] += nmiss;
        p.stats[2] += nfetch;
        p.stats[3] += nmiss - nfetch;
    }
}

struct MoeSchedule {
    static void run(tvm::ffi::TensorView ids, tvm::ffi::TensorView weights,
                    tvm::ffi::TensorView hit_ids, tvm::ffi::TensorView hit_weights,
                    tvm::ffi::TensorView fetch_ids, tvm::ffi::TensorView fetch_weights,
                    tvm::ffi::TensorView cpu_ids, tvm::ffi::TensorView map,
                    tvm::ffi::TensorView owners, tvm::ffi::TensorView usage,
                    tvm::ffi::TensorView step, tvm::ffi::TensorView source,
                    tvm::ffi::TensorView destination, tvm::ffi::TensorView copies,
                    tvm::ffi::TensorView counts, tvm::ffi::TensorView stats, tvm::ffi::TensorView fraction, tvm::ffi::TensorView fetch_table,
                    tvm::ffi::TensorView valid_tokens, int experts, int base, int evict_limit) {
        using namespace host;
        RuntimeCheck(ids.ndim() == 2 && weights.ndim() == 2 && weights.size(0) == ids.size(0) &&
                     weights.size(1) == ids.size(1), "schedule routes must be [tokens, top_k]");
        auto device = SymbolicDevice{};
        auto tokens = SymbolicSize{"tokens"};
        auto topk = SymbolicSize{"top_k"};
        auto slots = SymbolicSize{"slots"};
        auto total = SymbolicSize{"expert map"};
        auto plan = SymbolicSize{"copy descriptors"};
        auto i32 = SymbolicDType{};
        auto i64 = SymbolicDType{};
        auto f32 = SymbolicDType{};
        TensorMatcher({tokens, topk}).with_dtype<int32_t>(i32).with_device<kDLCUDA>(device)
            .verify(ids).verify(hit_ids).verify(fetch_ids).verify(cpu_ids);
        TensorMatcher({tokens, topk}).with_dtype<float>(f32).with_device<kDLCUDA>(device)
            .verify(weights).verify(hit_weights).verify(fetch_weights);
        TensorMatcher({total}).with_dtype<int32_t>(i32).with_device<kDLCUDA>(device).verify(map);
        TensorMatcher({slots}).with_dtype<int32_t>(i32).with_device<kDLCUDA>(device).verify(owners);
        TensorMatcher({slots}).with_dtype<int64_t>(i64).with_device<kDLCUDA>(device).verify(usage);
        TensorMatcher({plan}).with_dtype<int32_t>(i32).with_device<kDLCUDA>(device).verify(source).verify(destination);
        TensorMatcher({1}).with_dtype<int64_t>(i64).with_device<kDLCUDA>(device).verify(step).verify(copies);
        TensorMatcher({4}).with_dtype<int64_t>(i64).with_device<kDLCUDA>(device).verify(counts).verify(stats);
        TensorMatcher({1}).with_dtype<int32_t>(i32).with_device<kDLCUDA>(device).verify(fraction).verify(valid_tokens);
        TensorMatcher({experts + 1}).with_dtype<int32_t>(i32).with_device<kDLCUDA>(device).verify(fetch_table);
        RuntimeCheck(experts > 0 && experts <= 4096 && base >= 0 && base + experts <= total.unwrap(), "invalid expert range");
        RuntimeCheck(plan.unwrap() >= experts && slots.unwrap() >= experts, "schedule needs a full layer of slots/descriptors");
        RuntimeCheck(evict_limit >= experts && evict_limit <= slots.unwrap(), "invalid schedule victim range");
        const auto p = MoeScheduleParams{
            static_cast<const int32_t*>(ids.data_ptr()), static_cast<const float*>(weights.data_ptr()),
            static_cast<int32_t*>(hit_ids.data_ptr()), static_cast<float*>(hit_weights.data_ptr()),
            static_cast<int32_t*>(fetch_ids.data_ptr()), static_cast<float*>(fetch_weights.data_ptr()),
            static_cast<int32_t*>(cpu_ids.data_ptr()), static_cast<int32_t*>(map.data_ptr()),
            static_cast<int32_t*>(owners.data_ptr()), static_cast<int64_t*>(usage.data_ptr()),
            static_cast<int64_t*>(step.data_ptr()), static_cast<int32_t*>(source.data_ptr()),
            static_cast<int32_t*>(destination.data_ptr()), static_cast<int64_t*>(copies.data_ptr()),
            static_cast<int64_t*>(counts.data_ptr()), static_cast<int64_t*>(stats.data_ptr()), static_cast<const int32_t*>(fraction.data_ptr()),
            static_cast<const int32_t*>(fetch_table.data_ptr()),
            static_cast<const int32_t*>(valid_tokens.data_ptr()), static_cast<int>(tokens.unwrap()),
            static_cast<int>(topk.unwrap()), experts, static_cast<int>(slots.unwrap()), evict_limit, base};
        LaunchKernel(1, 256, device.unwrap(), experts * 3 * sizeof(int))(moe_schedule_kernel, p);
    }
};
