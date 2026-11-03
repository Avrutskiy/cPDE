#pragma once
#include "cuda_runtime.h"
// TEST ONLY: serial semantics, not real CUDA/CUB execution.
namespace cub {
struct DeviceReduce {
    template<class Input, class Output, class Op, class T>
    static cudaError_t Reduce(void* storage, size_t& bytes, Input in, Output out,
                              int count, Op op, T initial, cudaStream_t = 0) {
        if (!storage) { bytes = 1; return cudaSuccess; }
        for (int i = 0; i < count; ++i) initial = op(initial, in[i]);
        *out = initial;
        return cudaSuccess;
    }
};
}
