#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cublas_v2.h>

inline const char* gpu_status_name(cudaError_t status) {
    return cudaGetErrorString(status);
}

inline const char* gpu_status_name(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
#ifdef CUBLAS_STATUS_NOT_SUPPORTED
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#ifdef CUBLAS_STATUS_LICENSE_ERROR
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
        default: return "unknown cuBLAS status";
    }
}

inline bool gpu_status_ok(cudaError_t status) { return status == cudaSuccess; }
inline bool gpu_status_ok(cublasStatus_t status) { return status == CUBLAS_STATUS_SUCCESS; }

template <typename Status>
inline void check_gpu_status(Status status, const char* expression,
                             const char* file, int line) {
    if (gpu_status_ok(status)) return;
    std::fprintf(stderr, "%s:%d: %s failed: %s (%d)\n",
                 file, line, expression, gpu_status_name(status),
                 static_cast<int>(status));
    std::exit(EXIT_FAILURE);
}

#define checkCudaErrors(expression) \
    check_gpu_status((expression), #expression, __FILE__, __LINE__)
