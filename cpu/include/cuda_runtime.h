#pragma once
// Host implementation of the runtime subset used by the shared engine.
// This header is visible only to CPDE_CPU_BACKEND targets; it never loads CUDA.
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#define __global__
#define __device__
#define __host__
#define __shared__
using std::isfinite;
struct CpuLane { unsigned int x; };
inline CpuLane blockDim{1}, gridDim{1}, threadIdx{0}, blockIdx{0};
using cudaStream_t = int;
enum cudaError_t { cudaSuccess, cudaErrorMemoryAllocation, cudaErrorInvalidValue, cudaErrorInvalidDevice };
enum { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
       cudaMemcpyDeviceToDevice, cudaHostAllocDefault };
inline const char* cudaGetErrorString(cudaError_t status) {
    switch(status) {
    case cudaSuccess: return "CPU operation succeeded";
    case cudaErrorMemoryAllocation: return "CPU memory allocation failed";
    case cudaErrorInvalidDevice: return "CPU backend accepts only device=0";
    default: return "invalid CPU runtime argument";
    }
}
inline cudaError_t cudaGetDevice(int* device) {
    if(!device) return cudaErrorInvalidValue;
    *device=0; return cudaSuccess;
}
inline cudaError_t cudaSetDevice(int device) { return device==0 ? cudaSuccess : cudaErrorInvalidDevice; }
inline cudaError_t cudaMalloc(void** pointer, size_t bytes) {
    if(!pointer) return cudaErrorInvalidValue;
    *pointer=bytes ? std::malloc(bytes) : nullptr;
    return bytes && !*pointer ? cudaErrorMemoryAllocation : cudaSuccess;
}
inline cudaError_t cudaHostAlloc(void** pointer, size_t bytes, int) { return cudaMalloc(pointer,bytes); }
inline cudaError_t cudaFree(void* pointer) { std::free(pointer); return cudaSuccess; }
inline cudaError_t cudaFreeHost(void* pointer) { return cudaFree(pointer); }
inline cudaError_t cudaMemcpy(void* dest, const void* source, size_t bytes, int) {
    if(!bytes) return cudaSuccess;
    if(!dest || !source) return cudaErrorInvalidValue;
    std::memmove(dest,source,bytes); return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* dest, const void* source, size_t bytes, int kind, cudaStream_t) {
    return cudaMemcpy(dest,source,bytes,kind);
}
inline cudaError_t cudaMemsetAsync(void* dest, int value, size_t bytes, cudaStream_t) {
    if(!bytes) return cudaSuccess;
    if(!dest) return cudaErrorInvalidValue;
    std::memset(dest,value,bytes); return cudaSuccess;
}
inline cudaError_t cudaStreamCreate(cudaStream_t* stream) {
    if(!stream) return cudaErrorInvalidValue;
    *stream=0; return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
