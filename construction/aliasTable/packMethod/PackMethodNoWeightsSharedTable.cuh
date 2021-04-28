#ifndef ALIAS_GPU_PACKMETHODNOWEIGHTSSHAREDTABLE_CUH
#define ALIAS_GPU_PACKMETHODNOWEIGHTSSHAREDTABLE_CUH

#include "PackMethodNoWeights.cuh"
#include "PackMethodBasicShared.cuh"

/**
 * Pack method that assumes that the alias table is pre-filled with weights.
 * Additionally, it allocates shared memory for the interesting table entries instead of taking them from global memory.
 *
 * This method has the problem that it does not work for some weight distributions because it needs to
 * copy all relevant table entries. If light and heavy items are separated, it needs to copy too much.
 */
namespace PackMethodNoWeightsSharedMod {
    __global__
    void packKernel(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
                    SafeArray<int> h, SafeArray<int> l, ArrayOfStructs aliasTable);


    template<typename TableStorage>
    void execute(TableStorage aliasTableGpu, double W, int N, int p,
                    SafeArray<SplitConfig> splits, SafeArray<int> l, SafeArray<int> h);
}

#endif //ALIAS_GPU_PACKMETHODNOWEIGHTSSHAREDTABLE_CUH
