#pragma once
// TEST ONLY: the host probe has one simulated thread, which handles every sample.
// This is not a simulation or verification of CUB's parallel implementation.
namespace cub {
template<class T, int Threads> struct BlockReduce {
    struct TempStorage {};
    explicit BlockReduce(TempStorage&) {}
    template<class Op> T Reduce(T value, Op) { return value; }
};
}
