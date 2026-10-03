#include "fast_index_copy.cuh"

template <typename IdType, std::size_t kThreads, std::size_t kBlocksPerBank, int kUnroll>
__global__ __launch_bounds__(kThreads) void fast_index_copy_scheduled(
    const __grid_constant__ MultiIndexCopyParams p
) {
    const int64_t rows = p.valid_length ? p.valid_length[0] : p.length;
    if (rows <= 0) {
        return;
    }
    int64_t sum_units = 0;
    for (int bank = 0; bank < p.num_banks; ++bank) {
        sum_units += p.feat_bytes[bank] >> 4;
    }
    if (sum_units == 0) {
        return;
    }
    const int64_t extra_blocks = gridDim.x - p.num_banks;
    int64_t prefix_units = 0;
    int64_t begin = 0;
    for (int bank = 0; bank < p.num_banks; ++bank) {
        const int64_t units = p.feat_bytes[bank] >> 4;
        prefix_units += units;
        // Reserve one block per bank and apportion the remaining blocks by row bytes.
        const int64_t end = bank + 1 + extra_blocks * prefix_units / sum_units;
        if (blockIdx.x >= begin && blockIdx.x < end) {
            const auto* src = reinterpret_cast<const uint4*>(p.src_ptrs[bank]);
            auto* dst = reinterpret_cast<uint4*>(p.dst_ptrs[bank]);
            const auto* si = static_cast<const IdType*>(p.src_indices);
            const auto* di = static_cast<const IdType*>(p.dst_indices);
            const int64_t bank_threads = (end - begin) * kThreads;
            const int64_t thread = (blockIdx.x - begin) * kThreads + threadIdx.x;
            if (units < bank_threads) {
                // Short rows need row-parallel work when the bank grid is wider than one row.
                const int64_t total = rows * units;
                for (int64_t offset = thread; offset < total; offset += bank_threads * kUnroll) {
#pragma unroll
                    for (int part = 0; part < kUnroll; ++part) {
                        const int64_t pos = offset + bank_threads * part;
                        if (pos < total) {
                            const int64_t row = pos / units;
                            const int64_t col = pos - row * units;
                            dst[static_cast<int64_t>(di[row]) * units + col] =
                                src[static_cast<int64_t>(si[row]) * units + col];
                        }
                    }
                }
                return;
            }
            for (int64_t row = 0; row < rows; ++row) {
                const auto* row_src = src + static_cast<int64_t>(si[row]) * units;
                auto* row_dst = dst + static_cast<int64_t>(di[row]) * units;
                for (int64_t offset = thread; offset < units; offset += bank_threads * kUnroll) {
#pragma unroll
                    for (int part = 0; part < kUnroll; ++part) {
                        const int64_t col = offset + bank_threads * part;
                        if (col < units) {
                            row_dst[col] = row_src[col];
                        }
                    }
                }
            }
            return;
        }
        begin = end;
    }
}

template <std::size_t kThreads, std::size_t kBlocksPerBank, int kUnroll, std::size_t kMinBlocks = 0>
struct ScheduledIndexCopyKernel {
    static_assert(kBlocksPerBank > 0 && kThreads > 0 && kUnroll > 0);
    static void run(tvm::ffi::TensorView dst_ptrs, tvm::ffi::TensorView src_ptrs,
                    tvm::ffi::TensorView feat_bytes, tvm::ffi::TensorView dst_indices,
                    tvm::ffi::TensorView src_indices,
                    tvm::ffi::Optional<tvm::ffi::TensorView> num_indices) {
        using namespace host;
        auto device = SymbolicDevice{};
        auto banks = SymbolicSize{"num_banks"};
        auto length = SymbolicSize{"indices length"};
        auto ptr_dtype = SymbolicDType{};
        auto indices_dtype = SymbolicDType{};
        auto count_dtype = SymbolicDType{};
        TensorMatcher({banks}).with_dtype<int64_t>(ptr_dtype)
            .with_device<kDLCUDA, kDLROCM>(device).verify(dst_ptrs).verify(src_ptrs).verify(feat_bytes);
        TensorMatcher({length}).with_dtype<int32_t, int64_t>(indices_dtype)
            .with_device<kDLCUDA, kDLROCM>(device).verify(dst_indices).verify(src_indices);
        const int64_t* valid = nullptr;
        if (num_indices.has_value()) {
            TensorMatcher({1}).with_dtype<int64_t>(count_dtype)
                .with_device<kDLCUDA, kDLROCM>(device).verify(num_indices.value());
            valid = static_cast<const int64_t*>(num_indices.value().data_ptr());
        }
        const auto p = MultiIndexCopyParams{
            static_cast<const int64_t*>(dst_ptrs.data_ptr()),
            static_cast<const int64_t*>(src_ptrs.data_ptr()),
            static_cast<const int64_t*>(feat_bytes.data_ptr()),
            dst_indices.data_ptr(), src_indices.data_ptr(), valid,
            static_cast<int64_t>(length.unwrap()), static_cast<int>(banks.unwrap()),
        };
        const auto kernel = indices_dtype.unwrap().bits == 32
            ? fast_index_copy_scheduled<int32_t, kThreads, kBlocksPerBank, kUnroll>
            : fast_index_copy_scheduled<int64_t, kThreads, kBlocksPerBank, kUnroll>;
        const auto blocks = std::max<std::size_t>(kBlocksPerBank * banks.unwrap(), kMinBlocks);
        LaunchKernel(blocks, kThreads, device.unwrap())(kernel, p);
    }
};
