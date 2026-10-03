#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cstdint>

namespace {

struct DecodeMetadataParams {
  const int32_t* page_table;
  const int32_t* requests;
  const int32_t* positions;
  const int64_t* swa_mapping;
  int32_t* indptr;
  int32_t* indices;
  int32_t* q_positions;
  int32_t* swa_indices;
  int width;
  int batch_size;
  int mapping_size;
};

__global__ void prepare_decode_metadata_kernel(const __grid_constant__ DecodeMetadataParams p) {
  const int row = blockIdx.y;
  const int length = p.requests[row * 2 + 1];
  __shared__ int offset;
  if (threadIdx.x == 0) {
    int sum = 0;
    for (int i = 0; i < row; ++i) sum += p.requests[i * 2 + 1];
    offset = sum;
    if (blockIdx.x == 0) {
      p.indptr[row] = sum;
      if (row == p.batch_size - 1) p.indptr[row + 1] = sum + length;
      p.q_positions[row] = p.positions[row];
    }
  }
  __syncthreads();
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column < length) {
    const int slot = p.page_table[p.requests[row * 2] * p.width + column];
    p.indices[offset + column] = slot;
    if (p.swa_mapping) {
      const int mapped_slot = slot < 0 ? p.mapping_size + slot : slot;
      p.swa_indices[offset + column] = static_cast<int32_t>(p.swa_mapping[mapped_slot]);
    }
  }
}

struct PrepareDecodeMetadata {
  static void run(tvm::ffi::TensorView page_table, tvm::ffi::TensorView requests,
                  tvm::ffi::TensorView positions, tvm::ffi::TensorView indptr,
                  tvm::ffi::TensorView indices, tvm::ffi::TensorView q_positions,
                  tvm::ffi::Optional<tvm::ffi::TensorView> swa_mapping,
                  tvm::ffi::Optional<tvm::ffi::TensorView> swa_indices, int max_length) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto width = SymbolicSize{"page table width"};
    auto batch = SymbolicSize{"batch"};
    auto capacity = SymbolicSize{"index capacity"};
    TensorMatcher({-1, width}).with_strides({width, 1}).with_dtype<int32_t>()
        .with_device<kDLCUDA>(device).verify(page_table);
    TensorMatcher({batch, 2}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(requests);
    TensorMatcher({batch}).with_dtype<int32_t>().with_device<kDLCUDA>(device)
        .verify(positions).verify(q_positions);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(indptr);
    TensorMatcher({capacity}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(indices);
    RuntimeCheck(batch.unwrap() > 0 && indptr.size(0) == batch.unwrap() + 1,
                 "decode indptr must have batch+1 elements");
    RuntimeCheck(max_length > 0 && max_length <= width.unwrap() &&
                 capacity.unwrap() >= batch.unwrap() * max_length,
                 "decode metadata capacity is too small");
    RuntimeCheck(swa_mapping.has_value() == swa_indices.has_value(), "SWA mapping needs output indices");
    if (swa_mapping.has_value()) {
      TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCUDA>(device).verify(swa_mapping.value());
      TensorMatcher({capacity}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(swa_indices.value());
    }
    const auto p = DecodeMetadataParams{
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<const int32_t*>(requests.data_ptr()),
        static_cast<const int32_t*>(positions.data_ptr()),
        swa_mapping.has_value() ? static_cast<const int64_t*>(swa_mapping.value().data_ptr()) : nullptr,
        static_cast<int32_t*>(indptr.data_ptr()), static_cast<int32_t*>(indices.data_ptr()),
        static_cast<int32_t*>(q_positions.data_ptr()),
        swa_indices.has_value() ? static_cast<int32_t*>(swa_indices.value().data_ptr()) : nullptr,
        static_cast<int>(width.unwrap()), static_cast<int>(batch.unwrap()),
        swa_mapping.has_value() ? static_cast<int>(swa_mapping.value().size(0)) : 0};
    LaunchKernel(dim3((max_length + 255) / 256, batch.unwrap()), 256, device.unwrap())
        (prepare_decode_metadata_kernel, p);
  }
};

}  // namespace
