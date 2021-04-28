#include "PackMethodBasic.cuh"

template <typename TableStorage, typename LoopCondition>
__host__ __device__
bool PackMethodBasic::pack(int k, SplitConfig splitCurrent, SplitConfig splitPrevious, double W_N,
           SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, TableStorage aliasTable, SafeArray<double> weights, LoopCondition &loopCondition) {
    int iLower = splitPrevious.i;
    int iUpper = splitCurrent.i;
    int jLower = splitPrevious.j;
    int jUpper = splitCurrent.j;
    double spill = splitPrevious.spill;

    int i = iLower;
    int j = jLower;
    double w = spill;

    if (spill == 0) {
        w = h[j].getWeight(weights.data);
    }

    //std::cout<<"START: "<<"i="<<i<<" j="<<j<<" w="<<w<<std::endl;
    while (true) {
        if (loopCondition.shouldStopEarly(i, j, w)) {
            return false;
        }

        bool weightExhausted = w <= W_N + EPSILON;
        if (weightExhausted && j >= jUpper) {
            weightExhausted = false;
        } else if (!weightExhausted && i >= iUpper) {
            weightExhausted = true;
        }
        if (weightExhausted) {
            if (j >= jUpper) {
                //std::cout<<"END: "<<"i="<<i<<" j="<<j<<" w="<<w<<std::endl;
                assert(i == iUpper);
                return true;
            }
            aliasTable.weight(h[j].item) = w;
            LH_TYPE item = h[j + 1];
            aliasTable.alias(h[j].item) = item.item;

            w += item.getWeight(weights.data) - W_N;
            j++;
        } else {
            if (i >= iUpper) {
                //std::cout<<"END: "<<"i="<<i<<" j="<<j<<" w="<<w<<std::endl;
                assert(j == jUpper);
                return true;
            }
            LH_TYPE item = l[i];
            aliasTable.weight(item.item) = item.getWeight(weights.data);
            aliasTable.alias(item.item) = h[j].item;
            w += item.getWeight(weights.data) - W_N;
            i++;
        }
    }
}

template <typename TableStorage>
__global__
void PackMethodBasic::packKernel(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
                                 SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, TableStorage aliasTable, int p, double *weights) {
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x + 1 + splitOffset;
    if (k > p) {
        return;
    }
    SplitConfig splitCurrent = splits[k];
    SplitConfig splitPrevious = splits[k - 1];

    PackMethodBasic::PackAll packMethodLimit;
    packOptimized(k, splitCurrent, splitPrevious, W_N, h.data, l.data, aliasTable, weights, packMethodLimit);
}

template
__host__ __device__
bool PackMethodBasic::pack<ArrayOfStructs, PackMethodBasic::PackAll>(int k, SplitConfig splitCurrent, SplitConfig splitPrevious, double W_N,
               SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, ArrayOfStructs aliasTable, SafeArray<double> weights, PackMethodBasic::PackAll &loopCondition);

template
__global__
void PackMethodBasic::packKernel<ArrayOfStructs>(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
                 SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, ArrayOfStructs aliasTable, int p, double *weights);

template
__host__ __device__
bool PackMethodBasic::pack<StructOfArrays, PackMethodBasic::PackAll>(int k, SplitConfig splitCurrent, SplitConfig splitPrevious, double W_N,
                SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, StructOfArrays aliasTable, SafeArray<double> weights, PackMethodBasic::PackAll &loopCondition);

template
__global__
void PackMethodBasic::packKernel<StructOfArrays>(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
                 SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, StructOfArrays aliasTable, int p, double *weights);
