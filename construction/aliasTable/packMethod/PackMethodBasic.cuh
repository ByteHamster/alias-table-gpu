#ifndef ALIAS_GPU_PACKMETHODBASIC_CUH
#define ALIAS_GPU_PACKMETHODBASIC_CUH

#include "construction/aliasTable/buildMethod/AliasTableSplit.cuh"
#include "construction/aliasTable/buildMethod/SplitConfig.cuh"
#include "construction/aliasTable/buildMethod/LhType.cuh"

/**
 * Pack method as it was described in the paper. Uses direct access to the weights array.
 */
namespace PackMethodBasic {

    struct PackAll {
        __host__ __device__ __forceinline__
        bool shouldStopEarly(int i, int j, double w) {
            return false;
        }
    };

    struct PackStopAtAndStoreState {
        SplitConfig *states;
        int iMax;
        int jMax;

        __host__ __device__ __forceinline__
        bool shouldStopEarly(int i, int j, double w) {
            #ifdef __CUDA_ARCH__
                if (j >= jMax || i >= iMax) {
                    // Not enough items left. Wait for loading another chunk.
                    states[threadIdx.x] = {i, j, w};
                    return true;
                }
            #else
                assert(false && "Not implemented");
            #endif
            return false;
        }
    };

    template <typename TableStorage>
    __global__
    void packKernel(int splitOffset, SafeArray<SplitConfig> splits, double W_N,
                SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, TableStorage aliasTable, int p, double *weights);

    /**
     * Returns true if packing is finished, false if it stopped because it needs to wait for more data to be loaded.
     */
    template <typename TableStorage, typename LoopCondition>
    __host__ __device__
    bool pack(int k, SplitConfig splitCurrent, SplitConfig splitPrevious, double W_N,
              SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, TableStorage aliasTable, SafeArray<double> weights,
              LoopCondition &loopCondition);


    /**
     * Returns true if packing is finished, false if it stopped because it needs to wait for more data to be loaded.
     * Defined in the header, so it can be inlined.
     */
    template <typename TableStorage, typename LoopCondition>
    __device__ __forceinline__
    bool packOptimized(int k, SplitConfig splitCurrent, SplitConfig splitPrevious, double W_N,
                       LH_TYPE const *h, LH_TYPE const *l, TableStorage aliasTable,
                       double const *weights, LoopCondition &loopCondition) {
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

        LH_TYPE const *aliasLocationPointer;
        LH_TYPE aliasLocation;
        ArrayOfStructs::TableRow rowTemp = {};
        double nextElementWeight;
        LH_TYPE const *nextElementWeightLocation;
        LH_TYPE const *aliasElementLocation;
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
                nextElementWeightLocation = h + j + 1;
                j++;
            } else {
                rowTemp.weight = aliasLocation.getWeight(weights);
                aliasElementLocation = h + j;
                nextElementWeightLocation = l + i;
                i++;
            }

            rowTemp.alias = aliasElementLocation->item;
            nextElementWeight = nextElementWeightLocation->getWeight(weights);

            w += nextElementWeight - W_N;
            aliasTable.setBoth(aliasLocation.item, rowTemp);
        }
    }
}


#endif //ALIAS_GPU_PACKMETHODBASIC_CUH
