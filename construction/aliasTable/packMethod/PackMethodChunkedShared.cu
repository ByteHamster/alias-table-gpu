#include "PackMethodChunkedShared.cuh"

__device__
void PackMethodChunkedShared::copyToSharedMemory(SplitConfig *threadStates,
                     ChunkedLoadingPosition *threadLoadingPosition, LH_TYPE *shared_l,
                     LH_TYPE *shared_h, SafeArray<LH_TYPE> l, SafeArray<LH_TYPE> h, int p) {

    for (int copyThread = 0; copyThread <= p && copyThread < CHUNKED_WORKER_THREADS; copyThread++) {

        int lightItemsConsumed = threadStates[copyThread].i - threadLoadingPosition[copyThread].l;
        if (lightItemsConsumed > CHUNK_THRESHOLD_NEXT_PAGE) {
            int copyStartPosition = threadStates[copyThread].i;

            int existingItemsToCopy = 0;
            #ifdef REUSE_SHARED_ITEMS
            if (threadLoadingPosition[copyThread].l >= 0) {
                // Something was already loaded. Reuse those which are still interesting (interleaved)
                existingItemsToCopy = min(CHUNKED_WORKER_THREADS, min(CHUNK_SIZE / 2, CHUNK_SIZE - lightItemsConsumed));
                if (threadIdx.x < existingItemsToCopy) {
                    shared_l[copyThread * CHUNK_SIZE + threadIdx.x] =
                            shared_l[copyThread * CHUNK_SIZE + lightItemsConsumed + threadIdx.x];
                }
            }
            #endif

            // Need to copy new light items (interleaved)
            for (unsigned int i = existingItemsToCopy + threadIdx.x; copyStartPosition + i < l.size
                    && i < CHUNK_SIZE; i += blockDim.x) {
                ASSIGN_LH(shared_l[copyThread * CHUNK_SIZE + i], l[copyStartPosition + i])
            }
        }

        int heavyItemsConsumed = threadStates[copyThread].j - threadLoadingPosition[copyThread].h;
        if (heavyItemsConsumed > CHUNK_THRESHOLD_NEXT_PAGE) {
            int copyStartPosition = threadStates[copyThread].j;

            int existingItemsToCopy = 0;
            #ifdef REUSE_SHARED_ITEMS
            if (threadLoadingPosition[copyThread].h >= 0) {
                // Something was already loaded. Reuse those which are still interesting (interleaved)
                existingItemsToCopy = min(CHUNKED_WORKER_THREADS, min(CHUNK_SIZE / 2, CHUNK_SIZE - heavyItemsConsumed));
                if (threadIdx.x < existingItemsToCopy) {
                    shared_h[copyThread * CHUNK_SIZE + threadIdx.x] =
                            shared_h[copyThread * CHUNK_SIZE + heavyItemsConsumed + threadIdx.x];
                }
            }
            #endif

            // Need to copy new heavy items (interleaved)
            for (unsigned int i = threadIdx.x + existingItemsToCopy; copyStartPosition + i < h.size
                    && i < CHUNK_SIZE; i += blockDim.x) {
                ASSIGN_LH(shared_h[copyThread * CHUNK_SIZE + i], h[copyStartPosition + i])
            }
        }
    }

    __syncthreads();

    // Write new positions
    int copyThread = threadIdx.x;
    if (copyThread <= p && copyThread < CHUNKED_WORKER_THREADS) {
        int lightItemsConsumed = threadStates[copyThread].i - threadLoadingPosition[copyThread].l;
        if (lightItemsConsumed > CHUNK_THRESHOLD_NEXT_PAGE) {
            int copyStartPosition = threadStates[copyThread].i;
            threadLoadingPosition[copyThread].l = copyStartPosition;
        }
        int heavyItemsConsumed = threadStates[copyThread].j - threadLoadingPosition[copyThread].h;
        if (heavyItemsConsumed > CHUNK_THRESHOLD_NEXT_PAGE) {
            int copyStartPosition = threadStates[copyThread].j;
            threadLoadingPosition[copyThread].h = copyStartPosition;
        }
    }
}

template <typename TableStorage>
__global__
void PackMethodChunkedShared::packKernel(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
               SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, TableStorage aliasTable, SafeArray<double> weights, int p) {
    assert(blockDim.x >= CHUNKED_WORKER_THREADS);
    __shared__ LH_TYPE shared_l[CHUNKED_WORKER_THREADS * CHUNK_SIZE];
    __shared__ LH_TYPE shared_h[CHUNKED_WORKER_THREADS * CHUNK_SIZE];
    __shared__ SplitConfig threadStates[CHUNKED_WORKER_THREADS];
    __shared__ ChunkedLoadingPosition threadLoadingPosition[CHUNKED_WORKER_THREADS];
    __shared__ int threadsFinished;
    unsigned int k = (blockIdx.x * CHUNKED_WORKER_THREADS) + 1 + splitOffset + threadIdx.x;

    if (threadIdx.x == 0) {
        threadsFinished = 0;
    }

    __syncthreads();

    if (threadIdx.x < CHUNKED_WORKER_THREADS) {
        if (k <= p) {
            threadStates[threadIdx.x] = splits[k - 1];
            threadLoadingPosition[threadIdx.x] = {-CHUNK_SIZE, -CHUNK_SIZE}; // -CHUNK_SIZE to force copy in first iteration
        } else {
            threadStates[threadIdx.x] = {0, 0, 0};
            threadLoadingPosition[threadIdx.x] = {-CHUNK_SIZE, -CHUNK_SIZE}; // -CHUNK_SIZE to force copy in first iteration
            atomicAdd(&threadsFinished, 1);
        }
    }

    bool finishedPacking = false;

    while (threadsFinished < CHUNKED_WORKER_THREADS) {
        __syncthreads();
        PackMethodChunkedShared::copyToSharedMemory(threadStates, threadLoadingPosition, shared_l, shared_h, l, h, p);
        __syncthreads();

        if (threadIdx.x < CHUNKED_WORKER_THREADS && k <= p && !finishedPacking) {
            SplitConfig splitCurrent = splits[k];
            SplitConfig splitPrevious = threadStates[threadIdx.x];
            PackMethodBasic::PackStopAtAndStoreState packStop = {
                    threadStates,
                    threadLoadingPosition[threadIdx.x].l + CHUNK_SIZE - 2,
                    threadLoadingPosition[threadIdx.x].h + CHUNK_SIZE - 2
            };

            LH_TYPE *l_offset = &shared_l[threadIdx.x * CHUNK_SIZE] - threadLoadingPosition[threadIdx.x].l;
            LH_TYPE *h_offset = &shared_h[threadIdx.x * CHUNK_SIZE] - threadLoadingPosition[threadIdx.x].h;

            bool finishedThisRound = PackMethodBasic::packOptimized<TableStorage, PackMethodBasic::PackStopAtAndStoreState>
                    (k, splitCurrent, splitPrevious, W_N, h_offset, l_offset, aliasTable, weights.data, packStop);
            if (finishedThisRound) {
                finishedPacking = true;
                atomicAdd(&threadsFinished, 1);
            }
        }
    }
}

template
__global__
void PackMethodChunkedShared::packKernel<ArrayOfStructs>(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
       SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, ArrayOfStructs aliasTable, SafeArray<double> weights, int p);

template
__global__
void PackMethodChunkedShared::packKernel<StructOfArrays>(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
        SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, StructOfArrays aliasTable, SafeArray<double> weights, int p);
