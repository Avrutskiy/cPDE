#pragma once
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#define __global__
#define __device__
#define __host__
#define __shared__
using std::isfinite;
struct TestDim { unsigned int x=1; };
static TestDim blockDim, gridDim, threadIdx{0}, blockIdx{0};
using cudaStream_t=int;using cudaEvent_t=int;
enum cudaError_t{cudaSuccess};
enum{cudaMemcpyHostToHost,cudaMemcpyHostToDevice,cudaMemcpyDeviceToHost,cudaMemcpyDeviceToDevice,cudaHostAllocDefault,cudaEventWaitDefault};
inline const char* cudaGetErrorString(cudaError_t){return "test shim";}
inline cudaError_t cudaGetDevice(int* d){*d=0;return cudaSuccess;}
inline cudaError_t cudaSetDevice(int){return cudaSuccess;}
inline cudaError_t cudaMalloc(void** p,size_t n){*p=std::malloc(n);if(n&&!*p)std::abort();return cudaSuccess;}
inline cudaError_t cudaHostAlloc(void** p,size_t n,int){return cudaMalloc(p,n);}
inline cudaError_t cudaFree(void* p){std::free(p);return cudaSuccess;}
inline cudaError_t cudaFreeHost(void* p){return cudaFree(p);}
inline cudaError_t cudaMemcpy(void* d,const void* s,size_t n,int){std::memcpy(d,s,n);return cudaSuccess;}
inline cudaError_t cudaMemcpyAsync(void* d,const void* s,size_t n,int k,cudaStream_t){return cudaMemcpy(d,s,n,k);}
inline cudaError_t cudaMemsetAsync(void* d,int v,size_t n,cudaStream_t){std::memset(d,v,n);return cudaSuccess;}
inline cudaError_t cudaStreamCreate(cudaStream_t* s){*s=0;return cudaSuccess;}
inline cudaError_t cudaEventCreate(cudaEvent_t* s){*s=0;return cudaSuccess;}
inline cudaError_t cudaStreamDestroy(cudaStream_t){return cudaSuccess;}
inline cudaError_t cudaEventDestroy(cudaEvent_t){return cudaSuccess;}
inline cudaError_t cudaEventRecord(cudaEvent_t,cudaStream_t){return cudaSuccess;}
inline cudaError_t cudaStreamWaitEvent(cudaStream_t,cudaEvent_t,int){return cudaSuccess;}
inline cudaError_t cudaStreamSynchronize(cudaStream_t){return cudaSuccess;}
inline cudaError_t cudaGetLastError(){return cudaSuccess;}
