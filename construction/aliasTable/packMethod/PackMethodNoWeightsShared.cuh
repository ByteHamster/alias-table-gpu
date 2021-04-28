#ifndef ALIAS_GPU_PACKMETHODNOWEIGHTSSHARED_CUH
#define ALIAS_GPU_PACKMETHODNOWEIGHTSSHARED_CUH

#include "PackMethodNoWeights.cuh"
#include "PackMethodBasicShared.cuh"

/**
 * Pack method that assumes that the alias table is pre-filled with weights.
 * Additionally, it allocates shared memory for l and h arrays instead of taking them from global memory.
 */
namespace PackMethodNoWeightsShared {
    template <typename TableStorage>
    __global__
    void packKernel(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
            SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, TableStorage aliasTable, int p);
}

#endif //ALIAS_GPU_PACKMETHODNOWEIGHTSSHARED_CUH
