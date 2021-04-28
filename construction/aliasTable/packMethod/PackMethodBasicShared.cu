#include "PackMethodBasicShared.cuh"

__device__
void PackMethodBasicShared::copyToSharedMemory(int k, int numSections, SafeArray<SplitConfig> splits, LH_TYPE *shared_lh,
                                               LH_TYPE **shared_l_out, LH_TYPE **shared_h_out, SafeArray<LH_TYPE> l, SafeArray<LH_TYPE> h) {
    assert(numSections > 0);
    SplitConfig splitCurrent = splits[k - 1 + numSections];
    SplitConfig splitPrevious = splits[k - 1];
    int iLower = splitPrevious.i;
    int iUpper = splitCurrent.i;
    int jLower = splitPrevious.j;
    int jUpper = splitCurrent.j;

    assert(iLower >= 0);
    assert(iUpper >= 0);
    assert(jLower >= 0);
    assert(jUpper >= 0);
    assert(iLower <= iUpper);
    assert(jLower <= jUpper);

    int numJ = jUpper - jLower;
    LH_TYPE *shared_h = &shared_lh[0] - jLower;
    LH_TYPE *shared_l = &shared_lh[0] + numJ + 5 - iLower;
    *shared_l_out = shared_l;
    *shared_h_out = shared_h;

    // Copy L to shared memory (interleaved)
    for (unsigned int i = iLower + threadIdx.x; i <= iUpper; i += blockDim.x) {
        ASSIGN_LH(shared_l[i], l[i])
    }

    // Copy H to shared memory (interleaved)
    for (unsigned int j = jLower + threadIdx.x; j <= jUpper; j += blockDim.x) {
        ASSIGN_LH(shared_h[j], h[j])
    }
    __syncthreads();
}

template <typename TableStorage>
__global__
void PackMethodBasicShared::packKernel(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
                                       SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, TableStorage aliasTable, double *weights, int p) {
    extern __shared__ LH_TYPE shared_lh[];
    unsigned int k = (blockIdx.x * SHARED_MEMORY_WORKER_THREADS) + 1 + splitOffset;

    LH_TYPE *shared_h;
    LH_TYPE *shared_l;

    int copyStart = k;
    int copyEnd = min(p + 1, k + SHARED_MEMORY_WORKER_THREADS);
    int numSections = copyEnd - copyStart;
    if (numSections <= 0) {
        return;
    }
    PackMethodBasicShared::copyToSharedMemory(k, numSections, splits, shared_lh, &shared_l, &shared_h, l, h);

    k += threadIdx.x;

    if (threadIdx.x >= SHARED_MEMORY_WORKER_THREADS || k > p) {
        return;
    }

    SplitConfig splitCurrent = splits[k];
    SplitConfig splitPrevious = splits[k - 1];
    PackMethodBasic::PackAll packMethodLimit;
    PackMethodBasic::packOptimized<TableStorage, PackMethodBasic::PackAll>(
            k, splitCurrent, splitPrevious, W_N, shared_h, shared_l, aliasTable, weights, packMethodLimit);
}

template
__global__
void PackMethodBasicShared::packKernel<ArrayOfStructs>(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
                                                       SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, ArrayOfStructs aliasTable, double *weights, int p);

template
__global__
void PackMethodBasicShared::packKernel<StructOfArrays>(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
                                                       SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, StructOfArrays aliasTable, double *weights, int p);
