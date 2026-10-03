#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cstdint>
#include <climits>
#include <cmath>

namespace {

__global__ void prepare_batch_kernel(const int32_t* desc, const int32_t* pages,
                                    const int32_t* tokens, int width, int real_rows,
                                    int32_t* ids, int32_t* loc, int32_t* positions,
                                    int64_t* input_rows, int64_t* input_cols,
                                    int64_t* output_rows, int64_t* output_cols,
                                    int32_t* linear, int32_t* decode_rows) {
  const int row = blockIdx.y;
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t* r = desc + row * 6;
  if (column < r[2] - r[1]) {
    const int at = r[5] + column;
    const int pos = r[1] + column;
    ids[at] = tokens[r[0] * width + pos];
    loc[at] = pages[r[0] * width + pos];
    positions[at] = pos;
    input_rows[at] = r[0];
    input_cols[at] = pos;
  }
  if (column == 0) {
    linear[row] = r[4];
    decode_rows[row * 2] = r[0];
    decode_rows[row * 2 + 1] = r[2];
    if (row < real_rows) {
      output_rows[row] = r[0];
      output_cols[row] = r[3];
    }
  }
}

struct PrepareBatch {
  static void run(tvm::ffi::TensorView desc, tvm::ffi::TensorView pages,
                  tvm::ffi::TensorView tokens, tvm::ffi::TensorView ids,
                  tvm::ffi::TensorView loc, tvm::ffi::TensorView positions,
                  tvm::ffi::TensorView in_rows, tvm::ffi::TensorView in_cols,
                  tvm::ffi::TensorView out_rows, tvm::ffi::TensorView out_cols,
                  tvm::ffi::TensorView linear, tvm::ffi::TensorView decode_rows,
                  int max_extend) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto rows = SymbolicSize{"rows"}; auto width = SymbolicSize{"width"};
    auto bs = SymbolicSize{"batch"}; auto n = SymbolicSize{"tokens"};
    auto real = SymbolicSize{"real rows"};
    TensorMatcher({rows, width}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(pages).verify(tokens);
    TensorMatcher({bs, 6}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(desc);
    TensorMatcher({n}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(ids).verify(loc).verify(positions);
    TensorMatcher({n}).with_dtype<int64_t>().with_device<kDLCUDA>(device).verify(in_rows).verify(in_cols);
    TensorMatcher({real}).with_dtype<int64_t>().with_device<kDLCUDA>(device).verify(out_rows).verify(out_cols);
    TensorMatcher({bs}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(linear);
    TensorMatcher({bs, 2}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(decode_rows);
    RuntimeCheck(bs.unwrap() > 0 && real.unwrap() <= bs.unwrap() && max_extend > 0, "invalid batch descriptor");
    LaunchKernel(dim3((max_extend + 255) / 256, bs.unwrap()), 256, device.unwrap())(
        prepare_batch_kernel, static_cast<const int32_t*>(desc.data_ptr()),
        static_cast<const int32_t*>(pages.data_ptr()), static_cast<const int32_t*>(tokens.data_ptr()),
        static_cast<int>(width.unwrap()), static_cast<int>(real.unwrap()),
        static_cast<int32_t*>(ids.data_ptr()), static_cast<int32_t*>(loc.data_ptr()),
        static_cast<int32_t*>(positions.data_ptr()), static_cast<int64_t*>(in_rows.data_ptr()),
        static_cast<int64_t*>(in_cols.data_ptr()), static_cast<int64_t*>(out_rows.data_ptr()),
        static_cast<int64_t*>(out_cols.data_ptr()), static_cast<int32_t*>(linear.data_ptr()),
        static_cast<int32_t*>(decode_rows.data_ptr()));
  }
};

struct PrefillMetadataParams {
  const int32_t* desc;
  const int32_t* pages;
  const int64_t* mapping;
  int32_t* cu_q;
  int32_t* indptr;
  int32_t* prefix;
  int32_t* indices;
  int32_t* q_to_req;
  int32_t* swa_indices;
  int width;
  int batch;
  int mapping_size;
};

__global__ void prefill_metadata_kernel(const __grid_constant__ PrefillMetadataParams p) {
  const int row = blockIdx.y;
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t* r = p.desc + row * 6;
  __shared__ int kv_offset;
  if (threadIdx.x == 0) {
    int sum = 0;
    for (int i = 0; i < row; ++i) sum += p.desc[i * 6 + 2];
    kv_offset = sum;
    if (blockIdx.x == 0) {
      p.cu_q[row] = r[5]; p.indptr[row] = sum; p.prefix[row] = r[1];
      if (row == p.batch - 1) {
        p.cu_q[row + 1] = r[5] + r[2] - r[1];
        p.indptr[row + 1] = sum + r[2];
      }
    }
  }
  __syncthreads();
  if (column < r[2]) {
    const int slot = p.pages[r[0] * p.width + column];
    p.indices[kv_offset + column] = slot;
    if (p.mapping) {
      const int at = slot < 0 ? p.mapping_size + slot : slot;
      p.swa_indices[kv_offset + column] = static_cast<int32_t>(p.mapping[at]);
    }
  }
  if (column < r[2] - r[1]) p.q_to_req[r[5] + column] = row;
}

struct PrefillMetadata {
  static void run(tvm::ffi::TensorView desc, tvm::ffi::TensorView pages,
                  tvm::ffi::TensorView cu_q, tvm::ffi::TensorView indptr,
                  tvm::ffi::TensorView prefix, tvm::ffi::TensorView indices,
                  tvm::ffi::TensorView q_to_req,
                  tvm::ffi::Optional<tvm::ffi::TensorView> mapping,
                  tvm::ffi::Optional<tvm::ffi::TensorView> swa_indices, int max_length) {
    using namespace host;
    auto device = SymbolicDevice{}; auto bs = SymbolicSize{"batch"};
    auto width = SymbolicSize{"width"}; auto n = SymbolicSize{"indices"};
    TensorMatcher({bs, 6}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(desc);
    TensorMatcher({-1, width}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(pages);
    TensorMatcher({bs.unwrap() + 1}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(cu_q).verify(indptr);
    TensorMatcher({bs}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(prefix);
    TensorMatcher({n}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(indices);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(q_to_req);
    RuntimeCheck(bs.unwrap() > 0 && max_length > 0 && max_length <= width.unwrap(), "invalid prefill metadata");
    RuntimeCheck(mapping.has_value() == swa_indices.has_value(), "SWA metadata needs output indices");
    if (mapping.has_value()) {
      TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCUDA>(device).verify(mapping.value());
      TensorMatcher({n}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(swa_indices.value());
    }
    const auto p = PrefillMetadataParams{
        static_cast<const int32_t*>(desc.data_ptr()), static_cast<const int32_t*>(pages.data_ptr()),
        mapping.has_value() ? static_cast<const int64_t*>(mapping.value().data_ptr()) : nullptr,
        static_cast<int32_t*>(cu_q.data_ptr()), static_cast<int32_t*>(indptr.data_ptr()),
        static_cast<int32_t*>(prefix.data_ptr()), static_cast<int32_t*>(indices.data_ptr()),
        static_cast<int32_t*>(q_to_req.data_ptr()),
        swa_indices.has_value() ? static_cast<int32_t*>(swa_indices.value().data_ptr()) : nullptr,
        static_cast<int>(width.unwrap()), static_cast<int>(bs.unwrap()),
        mapping.has_value() ? static_cast<int>(mapping.value().size(0)) : 0};
    LaunchKernel(dim3((max_length + 255) / 256, bs.unwrap()), 256, device.unwrap())(prefill_metadata_kernel, p);
  }
};

__global__ void qsa_metadata_kernel(const int32_t* desc, const int32_t* pages,
                                   int width, int page_size, int block_width, int bs,
                                   int32_t* cu_q, int32_t* token_to_req,
                                   int32_t* seq_lens, int32_t* ring_slots,
                                   int32_t* block_table, int32_t* last_indices) {
  const int row = blockIdx.y;
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t* r = desc + row * 6;
  if (column == 0) {
    cu_q[row] = r[5]; seq_lens[row] = r[2]; ring_slots[row] = r[0];
    last_indices[row] = r[5] + r[2] - r[1] - 1;
    if (row == bs - 1) cu_q[row + 1] = r[5] + r[2] - r[1];
  }
  if (column < r[2] - r[1]) token_to_req[r[5] + column] = row;
  if (column < block_width) {
    block_table[row * block_width + column] = pages[r[0] * width + column * page_size] / page_size;
  }
}

struct QsaMetadata {
  static void run(tvm::ffi::TensorView desc, tvm::ffi::TensorView pages,
                  tvm::ffi::TensorView cu_q, tvm::ffi::TensorView token_to_req,
                  tvm::ffi::TensorView seq_lens, tvm::ffi::TensorView ring_slots,
                  tvm::ffi::TensorView block_table, tvm::ffi::TensorView last_indices,
                  int page_size, int max_extend) {
    using namespace host;
    auto device = SymbolicDevice{}; auto bs = SymbolicSize{"batch"};
    auto width = SymbolicSize{"width"}; auto blocks = SymbolicSize{"blocks"};
    TensorMatcher({bs, 6}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(desc);
    TensorMatcher({-1, width}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(pages);
    TensorMatcher({bs.unwrap() + 1}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(cu_q);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(token_to_req);
    TensorMatcher({bs}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(seq_lens).verify(ring_slots).verify(last_indices);
    TensorMatcher({bs, blocks}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(block_table);
    RuntimeCheck(page_size > 0 && bs.unwrap() > 0 && max_extend > 0 &&
                 blocks.unwrap() == (width.unwrap() + page_size - 1) / page_size, "invalid QSA metadata geometry");
    const int n = std::max(max_extend, static_cast<int>(blocks.unwrap()));
    LaunchKernel(dim3((n + 255) / 256, bs.unwrap()), 256, device.unwrap())(
        qsa_metadata_kernel, static_cast<const int32_t*>(desc.data_ptr()),
        static_cast<const int32_t*>(pages.data_ptr()), static_cast<int>(width.unwrap()), page_size,
        static_cast<int>(blocks.unwrap()), static_cast<int>(bs.unwrap()), static_cast<int32_t*>(cu_q.data_ptr()),
        static_cast<int32_t*>(token_to_req.data_ptr()), static_cast<int32_t*>(seq_lens.data_ptr()),
        static_cast<int32_t*>(ring_slots.data_ptr()), static_cast<int32_t*>(block_table.data_ptr()),
        static_cast<int32_t*>(last_indices.data_ptr()));
  }
};

__global__ void copy_batch_inputs_kernel(const int32_t* ids, const int32_t* loc,
                                       const int32_t* pos, const int32_t* linear,
                                       int32_t* dst_ids, int32_t* dst_loc,
                                       int32_t* dst_pos, int32_t* dst_linear, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst_ids[i] = ids[i]; dst_loc[i] = loc[i];
    dst_pos[i] = pos[i]; dst_linear[i] = linear[i];
  }
}

struct CopyBatchInputs {
  static void run(tvm::ffi::TensorView ids, tvm::ffi::TensorView loc,
                  tvm::ffi::TensorView pos, tvm::ffi::TensorView linear,
                  tvm::ffi::TensorView dst_ids, tvm::ffi::TensorView dst_loc,
                  tvm::ffi::TensorView dst_pos, tvm::ffi::TensorView dst_linear) {
    using namespace host;
    auto device = SymbolicDevice{}; auto n = SymbolicSize{"batch"};
    TensorMatcher({n}).with_dtype<int32_t>().with_device<kDLCUDA>(device)
        .verify(ids).verify(loc).verify(pos).verify(linear)
        .verify(dst_ids).verify(dst_loc).verify(dst_pos).verify(dst_linear);
    LaunchKernel((n.unwrap() + 255) / 256, 256, device.unwrap())(
        copy_batch_inputs_kernel, static_cast<const int32_t*>(ids.data_ptr()),
        static_cast<const int32_t*>(loc.data_ptr()), static_cast<const int32_t*>(pos.data_ptr()),
        static_cast<const int32_t*>(linear.data_ptr()), static_cast<int32_t*>(dst_ids.data_ptr()),
        static_cast<int32_t*>(dst_loc.data_ptr()), static_cast<int32_t*>(dst_pos.data_ptr()),
        static_cast<int32_t*>(dst_linear.data_ptr()), static_cast<int>(n.unwrap()));
  }
};

__global__ void write_pages_kernel(int32_t* pages, const int32_t* allocated,
                                  const int32_t* desc, int width) {
  const int row = blockIdx.y;
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t* r = desc + row * 4;
  if (column < r[2]) pages[r[0] * width + r[1] + column] = allocated[r[3] + column];
}

struct WritePages {
  static void run(tvm::ffi::TensorView pages, tvm::ffi::TensorView allocated,
                  tvm::ffi::TensorView desc, int max_length) {
    using namespace host;
    auto device = SymbolicDevice{}; auto width = SymbolicSize{"width"}; auto bs = SymbolicSize{"rows"};
    TensorMatcher({-1, width}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(pages);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(allocated);
    TensorMatcher({bs, 4}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(desc);
    RuntimeCheck(max_length > 0 && bs.unwrap() > 0, "empty page write");
    LaunchKernel(dim3((max_length + 255) / 256, bs.unwrap()), 256, device.unwrap())(
        write_pages_kernel, static_cast<int32_t*>(pages.data_ptr()),
        static_cast<const int32_t*>(allocated.data_ptr()), static_cast<const int32_t*>(desc.data_ptr()),
        static_cast<int>(width.unwrap()));
  }
};

__global__ void alloc_swa_kernel(const int32_t* full, const int32_t* slots, int64_t* mapping, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) mapping[full[i]] = slots[i];
}

struct AllocSwa {
  static void run(tvm::ffi::TensorView full, tvm::ffi::TensorView slots, tvm::ffi::TensorView mapping) {
    using namespace host;
    auto device = SymbolicDevice{}; auto n = SymbolicSize{"slots"};
    TensorMatcher({n}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(full).verify(slots);
    TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCUDA>(device).verify(mapping);
    LaunchKernel((n.unwrap() + 255) / 256, 256, device.unwrap())(
        alloc_swa_kernel, static_cast<const int32_t*>(full.data_ptr()),
        static_cast<const int32_t*>(slots.data_ptr()), static_cast<int64_t*>(mapping.data_ptr()),
        static_cast<int>(n.unwrap()));
  }
};

__device__ bool argmax_better(float a, int ia, float b, int ib) {
  if (isnan(a)) return !isnan(b) || ia < ib;
  if (isnan(b)) return false;
  return a > b || (a == b && ia < ib);
}

template <bool kPartial>
__global__ void greedy_argmax_kernel(const float* logits, int32_t* output, float* partial_values,
                                    int32_t* partial_ids, int vocab, int stride, int splits) {
  __shared__ float values[256];
  __shared__ int indices[256];
  float best = -INFINITY; int at = INT_MAX;
  const int begin = kPartial ? blockIdx.y * 4096 : 0;
  const int end = kPartial ? min(vocab, begin + 4096) : vocab;
  for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
    const float v = logits[blockIdx.x * stride + i];
    if (argmax_better(v, i, best, at)) { best = v; at = i; }
  }
  values[threadIdx.x] = best; indices[threadIdx.x] = at;
  __syncthreads();
  for (int delta = 128; delta > 0; delta /= 2) {
    if (threadIdx.x < delta) {
      const int other = threadIdx.x + delta;
      if (argmax_better(values[other], indices[other], values[threadIdx.x], indices[threadIdx.x])) {
        values[threadIdx.x] = values[other]; indices[threadIdx.x] = indices[other];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if constexpr (kPartial) {
      const int at = blockIdx.x * splits + blockIdx.y;
      partial_values[at] = values[0]; partial_ids[at] = indices[0];
    } else {
      output[blockIdx.x] = indices[0];
    }
  }
}

__global__ void greedy_argmax_finish_kernel(const float* values, const int32_t* ids,
                                           int32_t* output, int splits) {
  __shared__ float best_values[256]; __shared__ int best_ids[256];
  float best = -INFINITY; int at = INT_MAX;
  for (int i = threadIdx.x; i < splits; i += blockDim.x) {
    const int offset = blockIdx.x * splits + i;
    if (argmax_better(values[offset], ids[offset], best, at)) { best = values[offset]; at = ids[offset]; }
  }
  best_values[threadIdx.x] = best; best_ids[threadIdx.x] = at;
  __syncthreads();
  for (int delta = 128; delta > 0; delta /= 2) {
    if (threadIdx.x < delta) {
      const int other = threadIdx.x + delta;
      if (argmax_better(best_values[other], best_ids[other], best_values[threadIdx.x], best_ids[threadIdx.x])) {
        best_values[threadIdx.x] = best_values[other]; best_ids[threadIdx.x] = best_ids[other];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) output[blockIdx.x] = best_ids[0];
}

struct GreedyArgmax {
  static void run(tvm::ffi::TensorView logits, tvm::ffi::TensorView output,
                  tvm::ffi::TensorView partial_values, tvm::ffi::TensorView partial_ids) {
    using namespace host;
    auto device = SymbolicDevice{}; auto bs = SymbolicSize{"batch"};
    auto vocab = SymbolicSize{"vocab"}; auto stride = SymbolicSize{"stride"};
    TensorMatcher({bs, vocab}).with_strides({stride, 1}).with_dtype<float>().with_device<kDLCUDA>(device).verify(logits);
    TensorMatcher({bs}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(output);
    RuntimeCheck(bs.unwrap() > 0 && vocab.unwrap() > 0, "empty sampling logits");
    const int splits = (vocab.unwrap() + 4095) / 4096;
    TensorMatcher({bs, splits}).with_dtype<float>().with_device<kDLCUDA>(device).verify(partial_values);
    TensorMatcher({bs, splits}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(partial_ids);
    auto* values = static_cast<float*>(partial_values.data_ptr());
    auto* ids = static_cast<int32_t*>(partial_ids.data_ptr());
    auto* out = static_cast<int32_t*>(output.data_ptr());
    if (splits == 1) {
      LaunchKernel(bs.unwrap(), 256, device.unwrap())(
          greedy_argmax_kernel<false>, static_cast<const float*>(logits.data_ptr()), out, values, ids,
          static_cast<int>(vocab.unwrap()), static_cast<int>(stride.unwrap()), splits);
    } else {
      LaunchKernel(dim3(bs.unwrap(), splits), 256, device.unwrap())(
          greedy_argmax_kernel<true>, static_cast<const float*>(logits.data_ptr()), out, values, ids,
          static_cast<int>(vocab.unwrap()), static_cast<int>(stride.unwrap()), splits);
      LaunchKernel(bs.unwrap(), 256, device.unwrap())(greedy_argmax_finish_kernel, values, ids, out, splits);
    }
  }
};

}  // namespace
