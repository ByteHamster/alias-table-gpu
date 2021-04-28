#include "PackMethodNoWeights.cuh"

template <typename TableStorage>
__host__ __device__
void PackMethodNoWeights::pack(int k, SafeArray<SplitConfig> splits, double W_N,
           SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, TableStorage aliasTable) {
    SplitConfig splitCurrent = splits[k];
    SplitConfig splitPrevious = splits[k - 1];
    int iLower = splitPrevious.i;
    int iUpper = splitCurrent.i;
    int jLower = splitPrevious.j;
    int jUpper = splitCurrent.j;
    double spill = splitPrevious.spill;

    int i = iLower;
    int j = jLower;
    double w = spill;

    if (spill == 0) {
        w = aliasTable.weight(h[j]);
    }

    //std::cout<<"START: "<<"i="<<i<<" j="<<j<<" w="<<w<<std::endl;
    while (true) {
        bool weightExhausted = w <= W_N + EPSILON;
        if (weightExhausted && j >= jUpper) {
            weightExhausted = false;
        } else if (!weightExhausted && i >= iUpper) {
            weightExhausted = true;
        }

        if (weightExhausted) {
            if (j >= jUpper) {
                //std::cout<<"END: "<<"i="<<i<<" j="<<j<<" w="<<w<<std::endl;
                return;
            }
            aliasTable.weight(h[j]) = w;
            aliasTable.alias(h[j]) = h[j + 1];

            double nextElementWeight = aliasTable.weight(h[j + 1]);
            if (j + 1 >= jUpper) {
                // This element is written to by another thread. Reading it can lead to inconsistencies.
                // Because this is the last heavy item, its weight actually does not matter.
                // It must only be enough to fill the remaining light items.
                nextElementWeight = __builtin_huge_val();
            }
            w += nextElementWeight - W_N;
            j++;
        } else {
            if (i >= iUpper) {
                //std::cout<<"END: "<<"i="<<i<<" j="<<j<<" w="<<w<<std::endl;
                return;
            }
            aliasTable.weight(l[i]) = aliasTable.weight(l[i]);
            aliasTable.alias(l[i]) = h[j];
            w += aliasTable.weight(l[i]) - W_N;
            i++;
        }
    }
}

template <typename TableStorage>
__host__ __device__
void PackMethodNoWeights::packOptimized(int k, SafeArray<SplitConfig> splits, double W_N,
                                        LH_TYPE const *h, LH_TYPE const *l, TableStorage aliasTable) {
    SplitConfig splitCurrent = splits[k];
    SplitConfig splitPrevious = splits[k - 1];
    int iLower = splitPrevious.i;
    int iUpper = splitCurrent.i;
    int jLower = splitPrevious.j;
    int jUpper = splitCurrent.j;
    double spill = splitPrevious.spill;

    int i = iLower;
    int j = jLower;
    double w = spill;

    if (spill == 0) {
        w = h[j].getWeightFromTable(aliasTable);
    }

    LH_TYPE const *aliasLocationPointer;
    LH_TYPE aliasLocation;
    ArrayOfStructs::TableRow rowTemp = {};
    double nextElementWeight;
    LH_TYPE const *nextElementWeightLocation;
    LH_TYPE const *aliasElementLocation;
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
        //rowTemp = aliasTable.getBoth(aliasLocation);

        if (weightExhausted) {
            rowTemp.weight = w;
            aliasElementLocation = h + j + 1;
            nextElementWeightLocation = h + j + 1;
            j++;
        } else {
            rowTemp.weight = aliasLocation.getWeightFromTable(aliasTable);
            aliasElementLocation = h + j;
            nextElementWeightLocation = l + i;
            i++;
        }

        rowTemp.alias = aliasElementLocation->item;
        nextElementWeight = nextElementWeightLocation->getWeightFromTable(aliasTable);

        if (j >= jUpper && weightExhausted) {
            // This element is written to by another thread. Reading it can lead to inconsistencies.
            // Because this is the last heavy item, its weight actually does not matter.
            // It must only be enough to fill the remaining light items.
            nextElementWeight = __builtin_huge_val();
        }

        w += nextElementWeight - W_N;
        aliasTable.setBoth(aliasLocation.item, rowTemp);
    }
}

template <typename TableStorage>
__global__
void PackMethodNoWeights::packKernel(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
         SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, TableStorage aliasTable, int p) {
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x + 1 + splitOffset;
    if (k > p) {
        return;
    }
    packOptimized<TableStorage>(k, splits, W_N, h.data, l.data, aliasTable);
}

template
__host__ __device__
void PackMethodNoWeights::packOptimized<ArrayOfStructs>(int k, SafeArray<SplitConfig> splits, double W_N,
            LH_TYPE const *h, LH_TYPE const *l, ArrayOfStructs aliasTable);

template
__host__ __device__
void PackMethodNoWeights::packOptimized<StructOfArrays>(int k, SafeArray<SplitConfig> splits, double W_N,
            LH_TYPE const *h, LH_TYPE const *l, StructOfArrays aliasTable);

template
__global__
void PackMethodNoWeights::packKernel<ArrayOfStructs>(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
             SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, ArrayOfStructs aliasTable, int p);

template
__global__
void PackMethodNoWeights::packKernel<StructOfArrays>(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
             SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, StructOfArrays aliasTable, int p);
