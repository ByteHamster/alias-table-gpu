#include "PackMethodNoWeightsShared.cuh"

template <typename TableStorage>
__global__
void PackMethodNoWeightsShared::packKernel(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
        SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, TableStorage aliasTable, int p) {
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

    PackMethodNoWeights::packOptimized<TableStorage>(k, splits, W_N, shared_h, shared_l, aliasTable);
}

template
__global__
void PackMethodNoWeightsShared::packKernel<ArrayOfStructs>(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
                SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, ArrayOfStructs aliasTable, int p);

template
__global__
void PackMethodNoWeightsShared::packKernel<StructOfArrays>(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
               SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, StructOfArrays aliasTable, int p);
