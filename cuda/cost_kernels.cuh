#pragma once
#include "reduction_kernels.cuh"

// One thread owns a sample, so equations sharing output components add dE/dY
// without atomics. Padded lanes are cleared before reading any external data.
template<bool CollectLoss>
__global__ void matching_cost_kernel(float** y, float** g, float** target,
    const int* target_rows, const float* weights, cpde_reduce::CostStats* partials,
    int outputs, int derivatives, int width, int valid, int offset,
    int weight_samples, bool gradient=true) {
    cpde_reduce::CostStats local = cpde_reduce::empty_cost();
    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < width;
         p += blockDim.x * gridDim.x) {
        float value = 0.0f;
        for (int d = 0; d < derivatives; ++d) for (int o = 0; o < outputs; ++o) {
            int i = p * outputs + o;
            if(gradient)g[d][i] = 0.0f;
            if (p >= valid) continue;
            int wp = weight_samples == 1 ? 0 : offset + p;
            float w = weights[(d * weight_samples + wp) * outputs + o];
            if (w == 0.0f) continue;
            float r = y[d][i] - target[target_rows[d]][(offset+p)*outputs+o];
            value += w*r*r;
            if(gradient)g[d][i] = 2.0f*w*r;
        }
        if (!isfinite(value)) cpde_reduce::mark_bad(local, offset + p, -1);
        if (CollectLoss) local.value += double(value);
        if (gradient && p < valid) for (int d = 0; d < derivatives; ++d) for (int o = 0; o < outputs; ++o)
            if (!isfinite(g[d][p * outputs + o])) cpde_reduce::mark_bad(local, offset + p, d * outputs + o);
    }
    using Reduce = cub::BlockReduce<cpde_reduce::CostStats, cpde_reduce::threads>;
    __shared__ typename Reduce::TempStorage storage;
    auto result = Reduce(storage).Reduce(local, cpde_reduce::CostCombine{});
    if (threadIdx.x == 0) partials[blockIdx.x] = result;
}

struct PolynomialTerm { int equation, a, b; }; // b=-1: linear; otherwise quadratic
__device__ float component(float** y, int index, int outputs, int p) {
    return y[index/outputs][p*outputs+index%outputs];
}
__device__ void add_component(float** g, int index, int outputs, int p, float v) {
    g[index/outputs][p*outputs+index%outputs] += v;
}
template<bool CollectLoss>
__global__ void polynomial_cost_kernel(float** y, float** g,
    const PolynomialTerm* terms, const float* coefficients, int term_count,
    const float* rhs, const float* weights, cpde_reduce::CostStats* partials,
    int outputs, int derivatives, int equations, int samples,
    int width, int valid, int offset, bool gradient=true) {
    cpde_reduce::CostStats local = cpde_reduce::empty_cost();
    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < width;
         p += blockDim.x * gridDim.x) {
        if(gradient)for (int d=0; d<derivatives; ++d) for(int o=0;o<outputs;++o)
            g[d][p*outputs+o]=0.0f;
        float total=0.0f;
        if (p < valid) for (int e=0; e<equations; ++e) {
            int sample=offset+p;
            float w=weights[e*samples+sample];
            if(w==0.0f) continue;
            float r=-rhs[e*samples+sample];
            for(int t=0;t<term_count;++t) if(terms[t].equation==e) {
                float c=coefficients[t*samples+sample];
                if(c==0.0f) continue;
                float a=component(y,terms[t].a,outputs,p);
                r+=c*a*(terms[t].b<0 ? 1.0f : component(y,terms[t].b,outputs,p));
            }
            total+=w*r*r;
            float scale=2.0f*w*r;
            if(gradient)for(int t=0;t<term_count;++t) if(terms[t].equation==e) {
                float c=coefficients[t*samples+sample];
                if(c==0.0f) continue;
                if(terms[t].b<0) add_component(g,terms[t].a,outputs,p,scale*c);
                else {
                    add_component(g,terms[t].a,outputs,p,
                        scale*c*component(y,terms[t].b,outputs,p));
                    add_component(g,terms[t].b,outputs,p,
                        scale*c*component(y,terms[t].a,outputs,p));
                }
            }
        }
        if (!isfinite(total)) cpde_reduce::mark_bad(local, offset + p, -1);
        if (CollectLoss) local.value += double(total);
        if (gradient && p < valid) for (int d = 0; d < derivatives; ++d) for (int o = 0; o < outputs; ++o)
            if (!isfinite(g[d][p * outputs + o])) cpde_reduce::mark_bad(local, offset + p, d * outputs + o);
    }
    using Reduce = cub::BlockReduce<cpde_reduce::CostStats, cpde_reduce::threads>;
    __shared__ typename Reduce::TempStorage storage;
    auto result = Reduce(storage).Reduce(local, cpde_reduce::CostCombine{});
    if (threadIdx.x == 0) partials[blockIdx.x] = result;
}
