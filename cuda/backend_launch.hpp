#pragma once

// Grid-stride kernels share their arithmetic between CUDA and the serial CPU
// backend. CUDA retains the original launch dimensions, shared memory and stream.
#if defined(CPDE_CPU_BACKEND) || defined(CPDE_HOST_PROBE)
#define CPDE_LAUNCH(kernel, grid, block, shared, stream, ...) kernel(__VA_ARGS__)
#else
#define CPDE_LAUNCH(kernel, grid, block, shared, stream, ...) \
    kernel<<<grid, block, shared, stream>>>(__VA_ARGS__)
#endif
