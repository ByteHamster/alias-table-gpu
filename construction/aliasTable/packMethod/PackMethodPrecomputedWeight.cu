#include "PackMethodPrecomputedWeight.cuh"

template <typename TableStorage>
__device__ __forceinline__
bool packOptimized(int k, SplitConfig splitCurrent, SplitConfig splitPrevious, double W_N,
                   LhTypeNoWeight const *h, LhTypeNoWeight const *l, TableStorage aliasTable,
                   double const *weights, SafeArray<double> weights_l, SafeArray<double> weights_h) {
    int iLower = splitPrevious.i;
    int iUpper = splitCurrent.i;
    int jLower = splitPrevious.j;
    int jUpper = splitCurrent.j;
    double spill = splitPrevious.spill;

    int i = iLower;
    int j = jLower;
    double w = spill;

    if (spill == 0) {
        w = h[j].getWeight(weights);
    }

    LhTypeNoWeight const *aliasLocationPointer;
    LhTypeNoWeight aliasLocation;
    ArrayOfStructs::TableRow rowTemp = {};
    double *nextElementWeight;
    LhTypeNoWeight const *aliasElementLocation;
    while (true) {
        bool weightExhausted = w <= W_N + EPSILON;
        if (weightExhausted && j >= jUpper) {
            weightExhausted = false;
        } else if (!weightExhausted && i >= iUpper) {
            weightExhausted = true;
        }

        if ((weightExhausted && j >= jUpper) || (!weightExhausted && i >= iUpper)) {
            return true;
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
            nextElementWeight = weights_h.data + j + 1;
            j++;
        } else {
            rowTemp.weight = aliasLocation.getWeight(weights);
            aliasElementLocation = h + j;
            nextElementWeight = weights_l.data + i;
            i++;
        }

        rowTemp.alias = aliasElementLocation->item;

        w += *nextElementWeight - W_N;
        aliasTable.setBoth(aliasLocation.item, rowTemp);
    }
}

template <typename TableStorage>
__global__
void PackMethodPrecomputedWeight::packKernel(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
                                 SafeArray<LhTypeNoWeight> h, SafeArray<LhTypeNoWeight> l, TableStorage aliasTable, int p,
                                 double *weights, SafeArray<double> weights_l, SafeArray<double> weights_h) {
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x + 1 + splitOffset;
    if (k > p) {
        return;
    }
    SplitConfig splitCurrent = splits[k];
    SplitConfig splitPrevious = splits[k - 1];

    packOptimized(k, splitCurrent, splitPrevious, W_N, h.data, l.data, aliasTable, weights, weights_l, weights_h);
}

__global__
void PackMethodPrecomputedWeight::fillWeightKernel(SafeArray<double> destination,
                               SafeArray<LhTypeNoWeight> location, SafeArray<double> weights) {
    for (unsigned int i = blockIdx.x * FILL_WEIGHTS_THREADS_PER_BLOCK * FILL_WEIGHTS_TO_TABLE_ITEMS_PER_THREAD + threadIdx.x;
               i < (blockIdx.x+1) * FILL_WEIGHTS_THREADS_PER_BLOCK * FILL_WEIGHTS_TO_TABLE_ITEMS_PER_THREAD + threadIdx.x
               && i < destination.size; i += FILL_WEIGHTS_THREADS_PER_BLOCK) {
        destination[i] = weights[location[i].item];
    }
}

template
__global__
void PackMethodPrecomputedWeight::packKernel<ArrayOfStructs>(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
                 SafeArray<LhTypeNoWeight> h, SafeArray<LhTypeNoWeight> l, ArrayOfStructs aliasTable, int p,
                 double *weights, SafeArray<double> weights_l, SafeArray<double> weights_h);

template
__global__
void PackMethodPrecomputedWeight::packKernel<StructOfArrays>(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
                 SafeArray<LhTypeNoWeight> h, SafeArray<LhTypeNoWeight> l, StructOfArrays aliasTable, int p,
                 double *weights, SafeArray<double> weights_l, SafeArray<double> weights_h);
