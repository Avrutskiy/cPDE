#pragma once
#include "backend_launch.hpp"

#include <climits>
#include <algorithm>
// Callers may define E globally. CUB uses E as a template parameter, so shield
// its headers without changing the caller's macro state.
#pragma push_macro("E")
#undef E
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#pragma pop_macro("E")
#include "cuda_check_v1.cuh"

namespace cpde_reduce {
constexpr int threads = 256;
constexpr int max_blocks = 1024;

inline int blocks_for(int count) {
#if defined(CPDE_HOST_PROBE) || defined(CPDE_CPU_BACKEND)
    // CPU execution and the test probe use one serial lane for grid-stride loops.
    return 1;
#else
    return std::min(max_blocks, 1 + (count - 1) / threads);
#endif
}

struct CostStats {
    double value;
    int bad_sample;
    int bad_component; // -1: objective; otherwise derivative * outputs + output.
};

__host__ __device__ inline CostStats empty_cost() { return {0.0, INT_MAX, INT_MAX}; }

struct CostCombine {
    __host__ __device__ CostStats operator()(CostStats a, CostStats b) const {
        a.value += b.value; // Signed sum, never an absolute-value sum.
        if (b.bad_sample < a.bad_sample ||
            (b.bad_sample == a.bad_sample && b.bad_component < a.bad_component)) {
            a.bad_sample = b.bad_sample;
            a.bad_component = b.bad_component;
        }
        return a;
    }
};

__device__ inline void mark_bad(CostStats& stats, int sample, int component) {
    CostStats error = {0.0, sample, component};
    stats = CostCombine{}(stats, error);
}

struct GradientStats {
    double squared_norm;
    double max_abs;
    unsigned long long nonfinite_count;
    int first_nonfinite;
};

__host__ __device__ inline GradientStats empty_gradient() { return {0.0, 0.0, 0, INT_MAX}; }

struct GradientCombine {
    __host__ __device__ GradientStats operator()(GradientStats a, GradientStats b) const {
        a.squared_norm += b.squared_norm;
        if (b.max_abs > a.max_abs) a.max_abs = b.max_abs;
        a.nonfinite_count += b.nonfinite_count;
        if (b.first_nonfinite < a.first_nonfinite) a.first_nonfinite = b.first_nonfinite;
        return a;
    }
};

__global__ void clear_costs(CostStats* values, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
        values[i] = empty_cost();
}

__global__ void accumulate_cost(const CostStats* chunk, CostStats* total) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *total = CostCombine{}(*total, *chunk);
}

__global__ void gradient_blocks(const float* gradient, int count, GradientStats* partials) {
    GradientStats local = empty_gradient();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        const double v = gradient[i];
        if (!isfinite(v)) {
            ++local.nonfinite_count;
            if (i < local.first_nonfinite) local.first_nonfinite = i;
        } else {
            local.squared_norm += v * v; // Promote BEFORE squaring float gradients.
            const double magnitude = fabs(v);
            if (magnitude > local.max_abs) local.max_abs = magnitude;
        }
    }
    using Reduce = cub::BlockReduce<GradientStats, threads>;
    __shared__ typename Reduce::TempStorage storage;
    GradientStats result = Reduce(storage).Reduce(local, GradientCombine{});
    if (threadIdx.x == 0) partials[blockIdx.x] = result;
}

// Reused per network. No allocations or scalar host transfers inside a chunk loop.
struct Workspace {
    CostStats *cost_blocks = nullptr, *cost_chunk = nullptr, *cost_totals = nullptr;
    GradientStats *gradient_partials = nullptr, *gradient_totals = nullptr;
    CostStats* host_costs = nullptr;
    GradientStats* host_gradients = nullptr;
    void* temporary = nullptr;
    size_t temporary_bytes = 0;
    int cost_capacity = 0, layer_capacity = 0;

    Workspace() = default;
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;

    ~Workspace() {
        cudaFree(cost_blocks); cudaFree(cost_chunk); cudaFree(cost_totals);
        cudaFree(gradient_partials); cudaFree(gradient_totals); cudaFree(temporary);
        cudaFreeHost(host_costs); cudaFreeHost(host_gradients);
    }

    void reserve(int costs, int layers, cudaStream_t stream) {
        if (!cost_blocks) {
            checkCudaErrors(cudaMalloc((void**)&cost_blocks, max_blocks * sizeof(CostStats)));
            checkCudaErrors(cudaMalloc((void**)&cost_chunk, sizeof(CostStats)));
            checkCudaErrors(cudaMalloc((void**)&gradient_partials, max_blocks * sizeof(GradientStats)));
        }
        if (costs > cost_capacity) {
            checkCudaErrors(cudaFree(cost_totals)); checkCudaErrors(cudaFreeHost(host_costs));
            checkCudaErrors(cudaMalloc((void**)&cost_totals, costs * sizeof(CostStats)));
            checkCudaErrors(cudaHostAlloc((void**)&host_costs, costs * sizeof(CostStats), cudaHostAllocDefault));
            cost_capacity = costs;
        }
        if (layers > layer_capacity) {
            checkCudaErrors(cudaFree(gradient_totals)); checkCudaErrors(cudaFreeHost(host_gradients));
            checkCudaErrors(cudaMalloc((void**)&gradient_totals, layers * sizeof(GradientStats)));
            checkCudaErrors(cudaHostAlloc((void**)&host_gradients, layers * sizeof(GradientStats), cudaHostAllocDefault));
            layer_capacity = layers;
        }
        if (!temporary) {
            size_t cost_bytes = 0, gradient_bytes = 0;
            checkCudaErrors(cub::DeviceReduce::Reduce(nullptr, cost_bytes, cost_blocks, cost_chunk,
                max_blocks, CostCombine{}, empty_cost(), stream));
            checkCudaErrors(cub::DeviceReduce::Reduce(nullptr, gradient_bytes, gradient_partials,
                gradient_totals, max_blocks, GradientCombine{}, empty_gradient(), stream));
            temporary_bytes = std::max(size_t(1), std::max(cost_bytes, gradient_bytes));
            checkCudaErrors(cudaMalloc(&temporary, temporary_bytes));
        }
    }

    void begin(int costs, cudaStream_t stream) {
        CPDE_LAUNCH(clear_costs, blocks_for(costs), threads, 0, stream, cost_totals, costs);
        checkCudaErrors(cudaGetLastError());
    }

    void finish_chunk(int blocks, int cost_index, cudaStream_t stream) {
        size_t bytes = temporary_bytes;
        checkCudaErrors(cub::DeviceReduce::Reduce(temporary, bytes, cost_blocks,
            cost_chunk, blocks, CostCombine{}, empty_cost(), stream));
        CPDE_LAUNCH(accumulate_cost, 1, 1, 0, stream, cost_chunk, cost_totals + cost_index);
        checkCudaErrors(cudaGetLastError());
    }

    void inspect_gradient(const float* gradient, int count, int layer_index, cudaStream_t stream) {
        int blocks = blocks_for(count);
        CPDE_LAUNCH(gradient_blocks, blocks, threads, 0, stream, gradient, count, gradient_partials);
        checkCudaErrors(cudaGetLastError());
        size_t bytes = temporary_bytes;
        checkCudaErrors(cub::DeviceReduce::Reduce(temporary, bytes, gradient_partials,
            gradient_totals + layer_index, blocks, GradientCombine{}, empty_gradient(), stream));
    }

    void download(int costs, int layers, cudaStream_t stream) {
        checkCudaErrors(cudaMemcpyAsync(host_costs, cost_totals, costs * sizeof(CostStats),
            cudaMemcpyDeviceToHost, stream));
        if (layers) checkCudaErrors(cudaMemcpyAsync(host_gradients, gradient_totals,
            layers * sizeof(GradientStats), cudaMemcpyDeviceToHost, stream));
        checkCudaErrors(cudaStreamSynchronize(stream));
    }
};
} // namespace cpde_reduce
