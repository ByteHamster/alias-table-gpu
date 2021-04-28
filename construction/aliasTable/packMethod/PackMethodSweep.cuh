#ifndef ALIAS_GPU_PACKMETHODSWEEP_CUH
#define ALIAS_GPU_PACKMETHODSWEEP_CUH

#include "construction/aliasTable/buildMethod/AliasTableSplit.cuh"
#include "construction/aliasTable/buildMethod/SplitConfig.cuh"
#include "construction/aliasTable/buildMethod/LhType.cuh"

/**
 * Instead of looking at the light and heavy arrays, the method scans the weight array
 * until it finds the next light/heavy item.
 */
namespace PackMethodSweep {
    template <typename TableStorage>
    __host__ __device__
    void pack(int k, SafeArray<SplitConfig> splits, double W_N,
              SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, TableStorage aliasTable, SafeArray<double> weights);


    template <typename TableStorage>
    __host__ __device__
    void packOptimized(int k, SafeArray<SplitConfig> splits, double W_N,
               LH_TYPE const *h, LH_TYPE const *l, TableStorage aliasTable, double const *weights);

    template <typename TableStorage>
    __global__
    void packKernel(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
                SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, TableStorage aliasTable, int p, SafeArray<double> weights);
}


#endif //ALIAS_GPU_PACKMETHODSWEEP_CUH
