#ifndef ALIAS_GPU_PACKMETHODPRECOMPUTEDWEIGHT_CUH
#define ALIAS_GPU_PACKMETHODPRECOMPUTEDWEIGHT_CUH

#include "construction/aliasTable/buildMethod/AliasTableSplit.cuh"
#include "construction/aliasTable/buildMethod/SplitConfig.cuh"
#include "construction/aliasTable/buildMethod/LhType.cuh"

#define FILL_WEIGHTS_THREADS_PER_BLOCK 512
#define FILL_WEIGHTS_TO_TABLE_ITEMS_PER_THREAD 100

/**
 * Pack method that uses precomputed weight[l[i]]
 */
namespace PackMethodPrecomputedWeight {

    template <typename TableStorage>
    __global__
    void packKernel(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
                SafeArray<LhTypeNoWeight> h, SafeArray<LhTypeNoWeight> l, TableStorage aliasTable, int p,
                double *weights, SafeArray<double> weights_l, SafeArray<double> weights_h);

    __global__
    void fillWeightKernel(SafeArray<double> destination, SafeArray<LhTypeNoWeight> location, SafeArray<double> weights);
}


#endif //ALIAS_GPU_PACKMETHODPRECOMPUTEDWEIGHT_CUH
