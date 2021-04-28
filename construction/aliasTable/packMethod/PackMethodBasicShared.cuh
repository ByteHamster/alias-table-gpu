#ifndef ALIAS_GPU_PACKMETHODBASICSHARED_CUH
#define ALIAS_GPU_PACKMETHODBASICSHARED_CUH

#include "PackMethodBasic.cuh"

#ifndef SHARED_MEMORY_WORKER_THREADS
#define SHARED_MEMORY_WORKER_THREADS 32
#endif

/**
 * Pack method that allocates shared memory for l and h arrays instead of taking them from global memory.
 */
namespace PackMethodBasicShared {
    __device__
    void copyToSharedMemory(int k, int numSections, SafeArray<SplitConfig> splits, LH_TYPE *shared_lh,
            LH_TYPE **shared_l_out, LH_TYPE **shared_h_out, SafeArray<LH_TYPE> l, SafeArray<LH_TYPE> h);

    template <typename TableStorage>
    __global__
    void packKernel(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
            SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, TableStorage aliasTable, double *weights, int p);
}


#endif //ALIAS_GPU_PACKMETHODBASICSHARED_CUH
