#ifndef ALIAS_GPU_SPLITMETHODINVERSEPARALLEL_CUH
#define ALIAS_GPU_SPLITMETHODINVERSEPARALLEL_CUH

#include "construction/aliasTable/buildMethod/AliasTableSplit.cuh"

namespace SplitMethodInverseParallel {
    __global__
    void splitKernel(SafeArray<SplitConfig> splits, int N, double W, int p,
                     SafeArray<double> weights, SafeArray<LH_TYPE> h, SafeArray<double> prefixWeightH,
                     int h_size, SafeArray<double> prefixWeightL, int l_size);
}


#endif //ALIAS_GPU_SPLITMETHODINVERSEPARALLEL_CUH
