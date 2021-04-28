#ifndef ALIAS_GPU_PACKMETHODNOWEIGHTS_CUH
#define ALIAS_GPU_PACKMETHODNOWEIGHTS_CUH

#include "construction/aliasTable/buildMethod/AliasTableSplit.cuh"

/**
 * Pack method that assumes that the alias table is pre-filled with weights.
 */
namespace PackMethodNoWeights {
    template <typename TableStorage>
    __host__ __device__
    void pack(int k, SafeArray<SplitConfig> splits, double W_N,
              SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, TableStorage aliasTable);

    template <typename TableStorage>
    __host__ __device__
    void packOptimized(int k, SafeArray<SplitConfig> splits, double W_N,
                       LH_TYPE const *h, LH_TYPE const *l, TableStorage aliasTable);

    template <typename TableStorage>
    __global__
    void packKernel(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
                SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, TableStorage aliasTable, int p);
}

#endif //ALIAS_GPU_PACKMETHODNOWEIGHTS_CUH
