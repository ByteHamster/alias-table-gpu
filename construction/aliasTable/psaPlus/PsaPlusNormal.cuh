#ifndef ALIAS_GPU_PSAPLUS_NORMAL_CUH
#define ALIAS_GPU_PSAPLUS_NORMAL_CUH

#include "construction/aliasTable/buildMethod/AliasTableSplit.cuh"

#ifndef PSA_PLUS_THREADS_PER_BLOCK
#define PSA_PLUS_THREADS_PER_BLOCK 512
#endif

namespace PsaPlusNormal {
    template <typename TableStorage>
    __global__
    void kernel(int N, double W_N, SafeArray<double> weights, LH_TYPE *preAllocLh, int *numH, int *numL, TableStorage aliasTable);
}


#endif //ALIAS_GPU_PSAPLUS_NORMAL_CUH
