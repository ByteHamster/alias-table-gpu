#ifndef ALIAS_GPU_SPLITMETHODPARY_CUH
#define ALIAS_GPU_SPLITMETHODPARY_CUH

#include "construction/aliasTable/buildMethod/AliasTableSplit.cuh"
#include "construction/aliasTable/buildMethod/SplitConfig.cuh"
#include "construction/aliasTable/splitMethod/SplitMethodBasic.cuh"

#ifndef PARY_SEARCH_GROUP_SIZE
#define PARY_SEARCH_GROUP_SIZE 512
#endif

/**
 * Partial p-ary search
 */
namespace SplitMethodParySearch {

    __global__
    void splitKernel(int splitOffset, SafeArray<SplitConfig> splits, int N, double W, int p,
                     SafeArray<double> weights, SafeArray<LH_TYPE> h, SafeArray<double> prefixWeightH,
                     int h_size, SafeArray<double> prefixWeightL, int l_size);
}


#endif //ALIAS_GPU_SPLITMETHODPARY_CUH
