#include "SplitMethodInverse.cuh"

__global__
void SplitMethodInverse::splitKernel(SafeArray<SplitConfig> splits, int N, double W, int p,
                                     SafeArray<double> weights, SafeArray<LH_TYPE> h, SafeArray<double> prefixWeightH,
                                     int h_size, SafeArray<double> prefixWeightL, int l_size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= h_size - 1) {
        return;
    }

    if (j == 0) {
        splits[0] = {0, 0, 0};
        splits[p] = {(int) l_size - 1, (int) h_size - 1, 0};
    }

    for (int k = 1; k < p; k++) {
        int n_ = ceil(((double) N * k) / p);
        if (j >= n_) {
            continue;
        }
        int i = min(n_ - j, (int) l_size - 1);
        double weightPerBucket = (W * n_)/N;
        double sigma = prefixWeightL[i] + prefixWeightH[j];
        double sigmaNext = prefixWeightL[i - 1] + prefixWeightH[j + 1];

        if (sigma <= weightPerBucket && sigmaNext > weightPerBucket) {
            splits[k] = {i, j, sigma + h[j].getWeight(weights.data) - weightPerBucket};
        }
    }
}
