#pragma once
// CPU kernels execute a grid-stride loop on one lane; the local value already
// contains every sample, so the block reduction is the identity operation.
namespace cub {
template<class T, int Threads> struct BlockReduce {
    struct TempStorage {};
    explicit BlockReduce(TempStorage&) {}
    template<class Op> T Reduce(T value, Op) { return value; }
};
}
