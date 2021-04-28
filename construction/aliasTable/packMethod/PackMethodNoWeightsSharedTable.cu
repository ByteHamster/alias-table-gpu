#include "PackMethodNoWeightsSharedTable.cuh"

__host__ __device__
void packOptimizedSharedTable(int k, SafeArray<SplitConfig> splits, double W_N, int const *h, int const *l,
                              ArrayOfStructs::TableRow *tablePointer) {
    SplitConfig splitCurrent = splits[k];
    SplitConfig splitPrevious = splits[k - 1];
    int iLower = splitPrevious.i;
    int iUpper = splitCurrent.i;
    int jLower = splitPrevious.j;
    int jUpper = splitCurrent.j;
    double spill = splitPrevious.spill;

    int minIdx = min(h[jLower], l[iLower]);
    int maxIdx = max(h[jUpper], l[iUpper]);

    int i = iLower;
    int j = jLower;
    double w = spill;

    if (spill == 0) {
        w = tablePointer[h[j]].weight;
    }

    int const *aliasLocationPointer;
    int aliasLocation;
    ArrayOfStructs::TableRow rowTemp = {};
    double nextElementWeight;
    int const *nextElementWeightLocation;
    int const *aliasElementLocation;
    while (true) {
        bool weightExhausted = w <= W_N + EPSILON;
        if (weightExhausted && j >= jUpper) {
            weightExhausted = false;
        } else if (!weightExhausted && i >= iUpper) {
            weightExhausted = true;
        }

        if ((weightExhausted && j >= jUpper) || (!weightExhausted && i >= iUpper)) {
            return;
        }

        if (weightExhausted) {
            aliasLocationPointer = h + j;
        } else {
            aliasLocationPointer = l + i;
        }

        aliasLocation = *aliasLocationPointer;

        if (weightExhausted) {
            rowTemp.weight = w;
            aliasElementLocation = h + j + 1;
            nextElementWeightLocation = h + j + 1;
            j++;
        } else {
            rowTemp.weight = tablePointer[aliasLocation].weight;
            aliasElementLocation = h + j;
            nextElementWeightLocation = l + i;
            i++;
        }

        rowTemp.alias = *aliasElementLocation;
        nextElementWeight = tablePointer[*nextElementWeightLocation].weight;

        if (j >= jUpper && weightExhausted) {
            // This element is written to by another thread. Reading it can lead to inconsistencies.
            // Because this is the last heavy item, its weight actually does not matter.
            // It must only be enough to fill the remaining light items.
            nextElementWeight = __builtin_huge_val();
        }

        w += nextElementWeight - W_N;
        tablePointer[aliasLocation] = rowTemp;
        assert(aliasLocation >= minIdx);
        assert(aliasLocation <= maxIdx);
    }
}

__host__ __device__
void packSweep(int k, SafeArray<SplitConfig> splits, double W_N,
                                    int const *h, int const *l, ArrayOfStructs::TableRow *tablePointer) {
    SplitConfig splitCurrent = splits[k];
    SplitConfig splitPrevious = splits[k - 1];
    int iLower = splitPrevious.i;
    int iUpper = splitCurrent.i;
    int jLower = splitPrevious.j;
    int jUpper = splitCurrent.j;
    double spill = splitPrevious.spill;

    int i = l[iLower];
    int j = h[jLower];
    int iMax = l[iUpper]; // Actual max item index, no l/h indirection
    int jMax = h[jUpper];

    double w = spill;
    if (spill == 0) {
        w = tablePointer[h[jLower]].weight;
    }

    //std::cout<<"START: "<<"i="<<i<<" j="<<j<<" w="<<w<<std::endl;
    int aliasLocation;
    ArrayOfStructs::TableRow rowTemp = {};
    while (true) {
        bool weightExhausted = w <= W_N + EPSILON;
        if (weightExhausted && j >= jMax) {
            weightExhausted = false;
        } else if (!weightExhausted && i >= iMax) {
            weightExhausted = true;
        }

        int indexUpdate;
        int indexUpdateMax;

        if (weightExhausted) {
            if (j >= jMax) {
                //std::cout<<"END: "<<"i="<<i<<" j="<<j<<" w="<<w<<std::endl;
                assert(i == iMax);
                return;
            }
            indexUpdate = j;
            indexUpdateMax = jMax;
            aliasLocation = j;
        } else {
            if (i >= iMax) {
                //std::cout<<"END: "<<"i="<<i<<" j="<<j<<" w="<<w<<std::endl;
                assert(j == jMax);
                return;
            }
            indexUpdate = i;
            indexUpdateMax = iMax;
            aliasLocation = i;
        }

        do {
            indexUpdate++;

            double weight = tablePointer[indexUpdate].weight;
            if (weightExhausted && weight > W_N) {
                break;
            } else if (!weightExhausted && weight <= W_N) {
                break;
            }
        } while (indexUpdate < indexUpdateMax);

        int nextElementWeightLocation;
        if (weightExhausted) {
            rowTemp.weight = w;
            rowTemp.alias = indexUpdate;
            nextElementWeightLocation = indexUpdate;
            j = indexUpdate;
        } else {
            rowTemp.weight = tablePointer[aliasLocation].weight;
            rowTemp.alias = j;
            nextElementWeightLocation = i;
            i = indexUpdate;
        }

        w += tablePointer[nextElementWeightLocation].weight - W_N;
        tablePointer[aliasLocation] = rowTemp;
    }
}

__global__
void PackMethodNoWeightsSharedMod::packKernel(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
                                           SafeArray<int> h, SafeArray<int> l, ArrayOfStructs aliasTable) {
    extern __shared__ ArrayOfStructs::TableRow sharedTable[];
    unsigned int k = (blockIdx.x * SHARED_MEMORY_WORKER_THREADS) + 1 + splitOffset;
    SplitConfig splitCurrent = splits[k - 1 + SHARED_MEMORY_WORKER_THREADS];
    SplitConfig splitPrevious = splits[k - 1];
    int iLower = splitPrevious.i;
    int iUpper = splitCurrent.i;
    int jLower = splitPrevious.j;
    int jUpper = splitCurrent.j;

    int minIdx = min(h[jLower], l[iLower]);
    int maxIdx = max(h[jUpper], l[iUpper]);

    k += threadIdx.x;

    ArrayOfStructs::TableRow *tablePointer = &sharedTable[0] - minIdx;
    for (unsigned int i = minIdx + threadIdx.x; i <= maxIdx; i += blockDim.x) {
        tablePointer[i] = aliasTable.rows[i];
    }
    __syncthreads();

    // Run kernel (only one thread)
    if (threadIdx.x < SHARED_MEMORY_WORKER_THREADS) {
        // Method 1:
        //packSweep(k, splits, W_N, h.data, l.data, tablePointer);

        // Method 2:
        packOptimizedSharedTable(k, splits, W_N, h.data, l.data, tablePointer);
    }

    __syncthreads();

    // Only copy back if something changed
    for (unsigned int i = iLower + threadIdx.x; i <= iUpper; i += blockDim.x) {
        aliasTable.rows[l[i]] = tablePointer[l[i]];
    }
    for (unsigned int j = jLower + threadIdx.x; j <= jUpper; j += blockDim.x) {
        aliasTable.rows[h[j]] = tablePointer[h[j]];
    }
}

namespace PackMethodNoWeightsSharedMod {
    template <typename TableStorage>
    void execute(TableStorage aliasTableGpu, double W, int N, int p,
                       SafeArray<SplitConfig> splits, SafeArray<int> l, SafeArray<int> h) {
        assert(false && "Not implemented");
    }

    template<>
    void execute<ArrayOfStructs>(ArrayOfStructs aliasTableGpu, double W,
                   int N, int p, SafeArray<SplitConfig> splits, SafeArray<int> l, SafeArray<int> h) {
        cudaDeviceProp deviceProperties = {};
        cudaGetDeviceProperties(&deviceProperties, 0);
        int sharedMemorySize = deviceProperties.sharedMemPerBlock / 2;
        int rows = sharedMemorySize / sizeof(ArrayOfStructs::TableRow);
        assert((N / p) * SHARED_MEMORY_WORKER_THREADS < rows);
        // Wraps are 32 threads. Make only one wrap wait.
        PackMethodNoWeightsSharedMod::packKernel<<<dim3(p / SHARED_MEMORY_WORKER_THREADS), dim3(64), sharedMemorySize>>>
            (0, splits, W / N, h, l, aliasTableGpu);
    }
}
