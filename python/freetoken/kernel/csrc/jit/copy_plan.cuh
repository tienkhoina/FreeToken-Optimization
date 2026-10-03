#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstdint>
#include <vector>

class ByteCopyPlan : public tvm::ffi::Object {
public:
    ByteCopyPlan(tvm::ffi::TensorView dst, tvm::ffi::TensorView src,
                 tvm::ffi::TensorView sizes) {
        using namespace host;
        RuntimeCheck(dst.ndim() == 1 && src.ndim() == 1 && sizes.ndim() == 1,
                     "copy descriptors must be one-dimensional");
        RuntimeCheck(dst.size(0) == src.size(0) && dst.size(0) == sizes.size(0),
                     "copy descriptor lengths differ");
        for (const auto tensor : {dst, src, sizes}) {
            RuntimeCheck(tensor.device().device_type == kDLCPU && tensor.is_contiguous() &&
                         tensor.dtype().code == kDLInt && tensor.dtype().bits == 64,
                         "copy descriptors must be contiguous CPU int64 tensors");
        }
        const auto* dp = static_cast<const int64_t*>(dst.data_ptr());
        const auto* sp = static_cast<const int64_t*>(src.data_ptr());
        const auto* np = static_cast<const int64_t*>(sizes.data_ptr());
        for (int64_t i = 0; i < dst.size(0); ++i) {
            RuntimeCheck(np[i] >= 0, "copy length must be nonnegative");
            if (np[i] == 0) {
                continue;
            }
            RuntimeCheck(dp[i] != 0 && sp[i] != 0, "nonempty copy requires valid pointers");
            dst_.push_back(reinterpret_cast<void*>(dp[i]));
            src_.push_back(reinterpret_cast<const void*>(sp[i]));
            sizes_.push_back(static_cast<size_t>(np[i]));
        }
    }

    void launch(int64_t stream_handle) const {
        using namespace host;
        const auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
#if FREETOKEN_USE_ROCM
        RuntimeCheck(false, "ByteCopyPlan uses the CUDA runtime; ROCm callers use tensor copies");
#else
        // Independent async copies also handle mixed small/large banks without batch-copy heuristics.
        for (size_t i = 0; i < sizes_.size(); ++i) {
            CUDA_CHECK(cudaMemcpyAsync(dst_[i], src_[i], sizes_[i], cudaMemcpyDefault, stream));
        }
#endif
    }

    int64_t bytes() const {
        int64_t total = 0;
        for (const auto size : sizes_) {
            total += size;
        }
        return total;
    }

    TVM_FFI_DECLARE_OBJECT_INFO_FINAL("freetoken.ByteCopyPlan", ByteCopyPlan, tvm::ffi::Object);

private:
    std::vector<void*> dst_;
    std::vector<const void*> src_;
    std::vector<size_t> sizes_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ByteCopyPlan>()
        .def(refl::init<tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView>(), "__init__")
        .def("launch", &ByteCopyPlan::launch)
        .def("bytes", &ByteCopyPlan::bytes);
}
