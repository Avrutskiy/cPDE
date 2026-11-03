#pragma once
#include "cuda_runtime.h"
namespace cub {
struct DeviceReduce {
    template<class Input, class Output, class Op, class T>
    static cudaError_t Reduce(void* storage, size_t& bytes, Input in, Output out,
                              int count, Op op, T initial, cudaStream_t = 0) {
        if(count<0) return cudaErrorInvalidValue;
        if(!storage) {bytes=1;return cudaSuccess;}
        if(!out || (count && !in)) return cudaErrorInvalidValue;
        for(int i=0;i<count;++i) initial=op(initial,in[i]);
        *out=initial;
        return cudaSuccess;
    }
};
}
