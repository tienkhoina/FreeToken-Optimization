#include <freetoken/tensor.h>
#include <cuda_runtime_api.h>
#include <freetoken/utils.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <algorithm>
#include <vector>

inline void layer_cuda_check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

struct LayerPrefill {
  static void run(tvm::ffi::TensorView graph_handles, tvm::ffi::TensorView buffers,
                  tvm::ffi::TensorView bank_copies, tvm::ffi::TensorView plan,
                  tvm::ffi::TensorView device_desc, tvm::ffi::TensorView ids,
                  tvm::ffi::TensorView loc, tvm::ffi::TensorView positions,
                  tvm::ffi::TensorView mrope, tvm::ffi::TensorView plane_a,
                  tvm::ffi::TensorView plane_b, tvm::ffi::TensorView ple_values,
                  int64_t main_handle, int64_t copy_handle, bool overlap, int ple_layer) {
    using namespace host;
    for (auto tensor : {graph_handles, buffers, bank_copies, plan}) {
      RuntimeCheck(tensor.device().device_type == kDLCPU && tensor.is_contiguous() &&
                   tensor.dtype().code == kDLInt && tensor.dtype().bits == 64,
                   "layer dispatch plans require contiguous CPU int64 storage");
    }
    RuntimeCheck(graph_handles.ndim() == 2 && buffers.ndim() == 2 && buffers.size(1) == 10,
                 "invalid graph/buffer geometry");
    RuntimeCheck(plan.ndim() == 2 && plan.size(1) == 3 && device_desc.ndim() == 2 &&
                 device_desc.size(0) == plan.size(0) && device_desc.size(1) == 8,
                 "invalid block plan geometry");
    RuntimeCheck(bank_copies.ndim() == 3 && bank_copies.size(2) == 3 &&
                 bank_copies.size(0) == graph_handles.size(0) - 1,
                 "expert copy plan must have one row per decoder layer");
    RuntimeCheck(plan.size(0) > 0 && buffers.size(0) == graph_handles.size(1) &&
                 graph_handles.size(1) > 0 && graph_handles.size(1) % 2 == 0,
                 "layer dispatch needs blocks and two staging windows per bucket");
    RuntimeCheck(plane_a.ndim() == 2 && plane_b.ndim() == 2 && plane_a.is_contiguous() &&
                 plane_b.is_contiguous() && plane_a.size(0) == plane_b.size(0) &&
                 plane_a.size(1) == plane_b.size(1) && plane_a.dtype() == plane_b.dtype(),
                 "activation planes differ");
    RuntimeCheck(ids.ndim() == 1 && loc.size(0) == ids.size(0) && positions.size(0) == ids.size(0) &&
                 plane_a.size(0) >= ids.size(0) && mrope.ndim() == 2 &&
                 mrope.size(0) == 3 && mrope.size(1) == ids.size(0), "input rows differ");
    const auto* graphs = static_cast<const int64_t*>(graph_handles.data_ptr());
    const auto* pointers = static_cast<const int64_t*>(buffers.data_ptr());
    const auto* copies = static_cast<const int64_t*>(bank_copies.data_ptr());
    const auto* blocks = static_cast<const int64_t*>(plan.data_ptr());
    const int layers = static_cast<int>(bank_copies.size(0));
    const int graph_columns = static_cast<int>(graph_handles.size(1));
    const int buckets = graph_columns / 2;
    const int banks = static_cast<int>(bank_copies.size(1));
    const int64_t n = ids.size(0);
    for (int64_t block = 0; block < plan.size(0); ++block) {
      const auto* p = blocks + block * 3;
      RuntimeCheck(p[0] >= 0 && p[1] > 0 && p[0] + p[1] <= n && p[2] >= 0 && p[2] < buckets,
                   "invalid block range or bucket");
      RuntimeCheck(p[1] <= pointers[p[2] * 20 + 9], "block exceeds captured capacity");
    }
    const auto main = reinterpret_cast<cudaStream_t>(main_handle);
    const auto copy_stream = reinterpret_cast<cudaStream_t>(copy_handle);
    const size_t row_bytes = plane_a.size(1) * (plane_a.dtype().bits / 8);
    auto* a = static_cast<char*>(plane_a.data_ptr());
    auto* b = static_cast<char*>(plane_b.data_ptr());
    cudaEvent_t ready[2], released[2], begin, input_ready[2], compute_done[2], stage_free[2];
    cudaStream_t activation_stream;
    layer_cuda_check(cudaStreamCreateWithFlags(&activation_stream, cudaStreamNonBlocking));
    for (int parity = 0; parity < 2; ++parity) {
      layer_cuda_check(cudaEventCreateWithFlags(&ready[parity], cudaEventDisableTiming));
      layer_cuda_check(cudaEventCreateWithFlags(&released[parity], cudaEventDisableTiming));
      layer_cuda_check(cudaEventCreateWithFlags(&input_ready[parity], cudaEventDisableTiming));
      layer_cuda_check(cudaEventCreateWithFlags(&compute_done[parity], cudaEventDisableTiming));
      layer_cuda_check(cudaEventCreateWithFlags(&stage_free[parity], cudaEventDisableTiming));
    }
    layer_cuda_check(cudaEventCreateWithFlags(&begin, cudaEventDisableTiming));
    auto stage = [&](int64_t block, bool stage_ple, int parity, cudaStream_t stream) {
      const auto* p = blocks + block * 3;
      RuntimeCheck(p[0] >= 0 && p[1] > 0 && p[0] + p[1] <= n && p[2] >= 0 && p[2] < buckets,
                   "block exceeds input or bucket capacity");
      const auto* ptr = pointers + (p[2] * 2 + parity) * 10;
      RuntimeCheck(p[1] <= ptr[9], "block exceeds bucket");
      const size_t bytes = p[1] * sizeof(int32_t);
      layer_cuda_check(cudaMemcpyAsync(reinterpret_cast<void*>(ptr[0]), static_cast<const int32_t*>(ids.data_ptr()) + p[0], bytes, cudaMemcpyDeviceToDevice, stream));
      layer_cuda_check(cudaMemcpyAsync(reinterpret_cast<void*>(ptr[1]), static_cast<const int32_t*>(loc.data_ptr()) + p[0], bytes, cudaMemcpyDeviceToDevice, stream));
      layer_cuda_check(cudaMemcpyAsync(reinterpret_cast<void*>(ptr[2]), static_cast<const int32_t*>(positions.data_ptr()) + p[0], bytes, cudaMemcpyDeviceToDevice, stream));
      for (int axis = 0; axis < 3; ++axis) {
        layer_cuda_check(cudaMemcpyAsync(reinterpret_cast<int32_t*>(ptr[3]) + axis * ptr[9],
                                  static_cast<const int32_t*>(mrope.data_ptr()) + axis * n + p[0],
                                  bytes, cudaMemcpyDeviceToDevice, stream));
      }
      layer_cuda_check(cudaMemcpyAsync(reinterpret_cast<void*>(ptr[6]), static_cast<const int32_t*>(device_desc.data_ptr()) + block * 8,
                                8 * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
      if (stage_ple && ptr[7] && ple_values.size(0)) {
        const auto bytes_per_row = ple_values.size(1) * (ple_values.dtype().bits / 8);
        layer_cuda_check(cudaMemcpyAsync(reinterpret_cast<void*>(ptr[7]),
                                  static_cast<const char*>(ple_values.data_ptr()) + p[0] * bytes_per_row,
                                  p[1] * bytes_per_row, cudaMemcpyDefault, stream));
      }
      return ptr;
    };
    auto launch = [&](int layer, int bucket) {
      auto exec = reinterpret_cast<cudaGraphExec_t>(graphs[layer * graph_columns + bucket]);
      RuntimeCheck(exec != nullptr, "uncaptured layer bucket");
      layer_cuda_check(cudaGraphLaunch(exec, main));
    };
    // Graph row zero is embedding only. Decode/model weights are untouched here.
    for (int64_t block = 0; block < plan.size(0); ++block) {
      const auto* p = blocks + block * 3;
      const auto* ptr = stage(block, false, 0, main);
      launch(0, static_cast<int>(p[2] * 2));
      layer_cuda_check(cudaMemcpyAsync(a + p[0] * row_bytes, reinterpret_cast<void*>(ptr[5]),
                                p[1] * row_bytes, cudaMemcpyDefault, main));
    }
    layer_cuda_check(cudaEventRecord(begin, main));
    layer_cuda_check(cudaStreamWaitEvent(copy_stream, begin, 0));
    auto prefetch = [&](int layer) {
      auto stream = overlap ? copy_stream : main;
      const int parity = layer % 2;
      if (layer >= 2 && overlap) layer_cuda_check(cudaStreamWaitEvent(stream, released[parity], 0));
      for (int bank = 0; bank < banks; ++bank) {
        const auto* entry = copies + (layer * banks + bank) * 3;
        layer_cuda_check(cudaMemcpyAsync(reinterpret_cast<void*>(entry[0]), reinterpret_cast<void*>(entry[1]),
                                  static_cast<size_t>(entry[2]), cudaMemcpyDefault, stream));
      }
      layer_cuda_check(cudaEventRecord(ready[parity], stream));
    };
    prefetch(0);
    for (int layer = 0; layer < layers; ++layer) {
      const int parity = layer % 2;
      if (overlap) layer_cuda_check(cudaStreamWaitEvent(main, ready[parity], 0));
      layer_cuda_check(cudaEventRecord(begin, main));
      layer_cuda_check(cudaStreamWaitEvent(activation_stream, begin, 0));
      auto stage_input = [&](int64_t block) {
        const int slot = block % 2;
        if (block >= 2) layer_cuda_check(cudaStreamWaitEvent(activation_stream, stage_free[slot], 0));
        const auto* p = blocks + block * 3;
        const auto* ptr = stage(block, layer == ple_layer, slot, activation_stream);
        layer_cuda_check(cudaMemcpyAsync(reinterpret_cast<void*>(ptr[4]), a + p[0] * row_bytes,
                                        p[1] * row_bytes, cudaMemcpyDefault, activation_stream));
        layer_cuda_check(cudaEventRecord(input_ready[slot], activation_stream));
      };
      stage_input(0);
      for (int64_t block = 0; block < plan.size(0); ++block) {
        const auto* p = blocks + block * 3;
        const int slot = block % 2;
        const auto* ptr = pointers + (p[2] * 2 + slot) * 10;
        layer_cuda_check(cudaStreamWaitEvent(main, input_ready[slot], 0));
        launch(layer + 1, static_cast<int>(p[2] * 2 + slot));
        layer_cuda_check(cudaEventRecord(compute_done[slot], main));
        if (block + 1 < plan.size(0)) stage_input(block + 1);
        // Feed current-layer compute before the large next-layer DMA. Otherwise
        // one H2D engine can copy the next bank before the first activation arrives.
        if (block == 0 && layer + 1 < layers) prefetch(layer + 1);
        layer_cuda_check(cudaStreamWaitEvent(activation_stream, compute_done[slot], 0));
        layer_cuda_check(cudaMemcpyAsync(b + p[0] * row_bytes, reinterpret_cast<void*>(ptr[5]),
                                        p[1] * row_bytes, cudaMemcpyDefault, activation_stream));
        layer_cuda_check(cudaEventRecord(stage_free[slot], activation_stream));
      }
      for (int slot = 0; slot < std::min<int64_t>(2, plan.size(0)); ++slot)
        layer_cuda_check(cudaStreamWaitEvent(main, stage_free[slot], 0));
      layer_cuda_check(cudaEventRecord(released[parity], main));
      std::swap(a, b);
    }
    for (int parity = 0; parity < 2; ++parity) {
      layer_cuda_check(cudaEventDestroy(ready[parity]));
      layer_cuda_check(cudaEventDestroy(released[parity]));
      layer_cuda_check(cudaEventDestroy(input_ready[parity]));
      layer_cuda_check(cudaEventDestroy(compute_done[parity]));
      layer_cuda_check(cudaEventDestroy(stage_free[parity]));
    }
    layer_cuda_check(cudaEventDestroy(begin));
    layer_cuda_check(cudaStreamDestroy(activation_stream));
  }
};
