#include "PsaPlusNormal.cuh"

#define NUM_LOCAL_SPLITS 16

struct GreedySplitConfig {
    short iMin;
    short iMax;
    short jMin;
    short jMax;
};

template <typename TableStorage>
__global__
void PsaPlusNormal::kernel(int N, double W_N, SafeArray<double> weights, LH_TYPE *preAllocLh, int *numH, int *numL, TableStorage aliasTable) {
    #ifdef LH_TYPE_USE_WEIGHT
    extern __shared__ LH_TYPE shared_lh[];
    __shared__ int sharedNumL;
    __shared__ int sharedNumH;

    if (threadIdx.x == 0) {
        sharedNumL = 0;
        sharedNumH = 0;
    }

    __syncthreads();

    int sectionSize = N / gridDim.x + 1;
    int start = min(N, blockIdx.x * sectionSize);
    int end = min(N, (blockIdx.x + 1) * sectionSize);

    // Copy weight [start..end] to shared_lh (sorted light/heavy), atomically counting number of light and heavy items
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        double weight = weights[i];
        LH_TYPE item = {};
        item.item = i;
        item.setWeight(weight);
        if (weight <= W_N) {
            int oldValue = atomicAdd(&sharedNumL, 1);
            shared_lh[oldValue] = item;
        } else {
            int oldValue = atomicAdd(&sharedNumH, 1);
            shared_lh[sectionSize - oldValue - 1] = item;
        }
    }
    __syncthreads();

    // Pack greedy
    LH_TYPE *l = shared_lh;
    LH_TYPE *h = shared_lh + sectionSize - sharedNumH;

    __shared__ GreedySplitConfig splits[NUM_LOCAL_SPLITS];
    if (threadIdx.x < NUM_LOCAL_SPLITS) {
        int stepsL = sharedNumL / NUM_LOCAL_SPLITS;
        int stepsH = sharedNumH / NUM_LOCAL_SPLITS;
        splits[threadIdx.x].iMin = threadIdx.x * stepsL;
        splits[threadIdx.x].iMax = (threadIdx.x + 1) * stepsL;
        splits[threadIdx.x].jMin = threadIdx.x * stepsH;
        splits[threadIdx.x].jMax = (threadIdx.x + 1) * stepsH;
    }
    if (threadIdx.x == NUM_LOCAL_SPLITS - 1) {
        splits[NUM_LOCAL_SPLITS - 1].iMax = static_cast<short>(sharedNumL);
        splits[NUM_LOCAL_SPLITS - 1].jMax = static_cast<short>(sharedNumH);
    }

    if (threadIdx.x < NUM_LOCAL_SPLITS) {
        int i = splits[threadIdx.x].iMin;
        int j = splits[threadIdx.x].jMin;
        double w = h[j].getWeight(weights.data);
        int iMax = splits[threadIdx.x].iMax;
        int jMax = splits[threadIdx.x].jMax;

        if (iMax - i > 0 && jMax - j > 0) {

            LH_TYPE const *aliasLocationPointer;
            LH_TYPE aliasLocation;
            ArrayOfStructs::TableRow rowTemp = {};
            double nextElementWeight;
            LH_TYPE const *nextElementWeightLocation;
            LH_TYPE const *aliasElementLocation;
            while (true) {
                bool weightExhausted = w <= W_N + EPSILON;
                if ((weightExhausted && j >= jMax - 1) || (!weightExhausted && i >= iMax - 1)) { // -1 because no sentinel
                    break;
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
                    rowTemp.weight = aliasLocation.getWeight(weights.data);
                    aliasElementLocation = h + j;
                    nextElementWeightLocation = l + i;
                    i++;
                }

                rowTemp.alias = aliasElementLocation->item;
                nextElementWeight = nextElementWeightLocation->getWeight(weights.data);

                w += nextElementWeight - W_N;
                aliasTable.setBoth(aliasLocation.item, rowTemp);
            }

            splits[threadIdx.x].jMin = j + 1;
            splits[threadIdx.x].iMin = i;

            LH_TYPE remainderItem = {h[splits[threadIdx.x].jMin - 1].item, w};
            if (remainderItem.weight <= W_N + EPSILON) {
                splits[threadIdx.x].iMin--;
                l[splits[threadIdx.x].iMin] = remainderItem;
            } else {
                splits[threadIdx.x].jMin--;
                h[splits[threadIdx.x].jMin] = remainderItem;
            }
        }
    }

    __syncthreads();

    // Use global atomic variable of L and H to determine location in preAllocLh
    __shared__ int globalLocationL;
    __shared__ int globalLocationH;
    for (int splitToCopy = 0; splitToCopy < NUM_LOCAL_SPLITS; splitToCopy++) {
        int numToCopyL = splits[splitToCopy].iMax - splits[splitToCopy].iMin;
        int numToCopyH = splits[splitToCopy].jMax - splits[splitToCopy].jMin;
        if (threadIdx.x == 0) {
            // atomicAdd returns old value
            globalLocationL = atomicAdd_system(numL, numToCopyL);
            globalLocationH = (N + 2) - atomicAdd_system(numH, numToCopyH) - numToCopyH - 1; // Leave space for sentinel
        }
        __syncthreads();

        // Write to preAllocLh
        for (int i = threadIdx.x; i < numToCopyL; i += blockDim.x) {
            preAllocLh[globalLocationL + i] = l[splits[splitToCopy].iMin + i];
        }
        for (int i = threadIdx.x; i < numToCopyH; i += blockDim.x) {
            preAllocLh[globalLocationH + i] = h[splits[splitToCopy].jMin + i];
        }
        __syncthreads();
    }
    #endif
}

template
__global__
void PsaPlusNormal::kernel<ArrayOfStructs>(int N, double W_N, SafeArray<double> weights, LH_TYPE *preAllocLh, int *numH, int *numL, ArrayOfStructs aliasTable);

template
__global__
void PsaPlusNormal::kernel<StructOfArrays>(int N, double W_N, SafeArray<double> weights, LH_TYPE *preAllocLh, int *numH, int *numL, StructOfArrays aliasTable);
