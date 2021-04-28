#ifndef ALIAS_GPU_PACKMETHODCHUNKEDSHARED_CUH
#define ALIAS_GPU_PACKMETHODCHUNKEDSHARED_CUH

#include "PackMethodBasic.cuh"

#ifndef CHUNKED_GROUP_SIZE
#define CHUNKED_GROUP_SIZE 32
#endif

#ifndef CHUNKED_WORKER_THREADS
#define CHUNKED_WORKER_THREADS 8
#endif

#ifndef CHUNK_SIZE
#define CHUNK_SIZE 32
#endif

#ifndef CHUNK_THRESHOLD_NEXT_PAGE
#define CHUNK_THRESHOLD_NEXT_PAGE 11
#endif

//#define REUSE_SHARED_ITEMS

/**
 * Pack method that allocates shared memory for l and h arrays instead of taking them from global memory.
 * Can load data in chunks, so no memory restrictions apply because of the shared memory.
 */
namespace PackMethodChunkedShared {
    struct ChunkedLoadingPosition {
        int l;
        int h;
    };

    __device__
    void copyToSharedMemory(SplitConfig *threadStates,
                            ChunkedLoadingPosition *threadLoadingPosition, LH_TYPE *shared_l,
                            LH_TYPE *shared_h, SafeArray<LH_TYPE> l, SafeArray<LH_TYPE> h, int p);

    template <typename TableStorage>
    __global__
    void packKernel(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
            SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, TableStorage aliasTable, SafeArray<double> weights, int p);
}


#endif //ALIAS_GPU_PACKMETHODCHUNKEDSHARED_CUH
