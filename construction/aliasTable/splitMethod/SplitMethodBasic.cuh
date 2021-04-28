#ifndef ALIAS_GPU_SPLITMETHODBASIC_CUH
#define ALIAS_GPU_SPLITMETHODBASIC_CUH

#include "construction/aliasTable/buildMethod/AliasTableSplit.cuh"
#include "construction/aliasTable/buildMethod/SplitConfig.cuh"

namespace SplitMethodBasic {

    struct BinarySearchChecker {
        char status = 0;
        double sigma;
        double weightPerBucket;
        SafeArray<double> prefixWeightH;
        SafeArray<double> prefixWeightL;

        __host__ __device__ __forceinline__
        BinarySearchChecker(double weightPerBucket, SafeArray<double> prefixWeightH, SafeArray<double> prefixWeightL)
            : weightPerBucket(weightPerBucket), prefixWeightH(prefixWeightH), prefixWeightL(prefixWeightL) {

        }

        __host__ __device__ __forceinline__
        void check(int i, int j) {
            if (j + 1 >= prefixWeightH.size || i - 1 < 0) {
                status = RESULT_CONTINUE_LEFT;
                return;
            } else if (i >= prefixWeightL.size || j < 0) {
                status = RESULT_CONTINUE_RIGHT;
                return;
            }
            sigma = prefixWeightL[i] + prefixWeightH[j];
            double sigmaNext = prefixWeightL[i - 1] + prefixWeightH[j + 1];
            if (sigma <= weightPerBucket && sigmaNext > weightPerBucket) {
                status = RESULT_FOUND;
            } else if (sigma <= weightPerBucket) {
                status = RESULT_CONTINUE_RIGHT;
            } else {
                status = RESULT_CONTINUE_LEFT;
            }
        }
    };

    __host__ __device__
    void split(int k, SafeArray<SplitConfig> splits, int N, double W, int p,
               SafeArray<double> weights, SafeArray<LH_TYPE> h, SafeArray<double> prefixWeightH, int h_size,
               SafeArray<double> prefixWeightL, int l_size);

    __global__
    void splitKernel(int splitOffset,SafeArray<SplitConfig> splits, int N, double W, int p,
                     SafeArray<double> weights, SafeArray<LH_TYPE> h, SafeArray<double> prefixWeightH,
                     int h_size, SafeArray<double> prefixWeightL, int l_size);
}


#endif //ALIAS_GPU_SPLITMETHODBASIC_CUH
