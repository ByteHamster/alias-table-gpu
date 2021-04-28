#include "SplitMethodBasic.cuh"

__global__
void SplitMethodBasic::splitKernel(int splitOffset, SafeArray<SplitConfig> splits, int N, double W, int p,
                                   SafeArray<double> weights, SafeArray<LH_TYPE> h, SafeArray<double> prefixWeightH,
                                   int h_size, SafeArray<double> prefixWeightL, int l_size) {
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x + 1 + splitOffset;
    if (k >= p) {
        return;
    }
    split(k, splits, N, W, p, weights, h, prefixWeightH, h_size, prefixWeightL, l_size);
}

__host__ __device__
void SplitMethodBasic::split(int k, SafeArray<SplitConfig> splits, int N, double W, int p,
                             SafeArray<double> weights, SafeArray<LH_TYPE> h, SafeArray<double> prefixWeightH, int h_size,
                             SafeArray<double> prefixWeightL, int l_size) {
    if (k == 1) {
        splits[0] = {0, 0, 0};
        splits[p] = {(int) l_size - 1, (int) h_size - 1, 0};
    }
    int n_ = ceil(((double) N * k) / p);

    int a = 0;
    int b = min(n_, (int) h_size) - 1;
    double weightPerBucket = (W * n_)/N;
    BinarySearchChecker checker(weightPerBucket, prefixWeightH, prefixWeightL);

    checker.check(min(n_ - a, (int) l_size - 1), a);
    assert(checker.status == RESULT_CONTINUE_RIGHT || checker.status == RESULT_FOUND);
    checker.check(min(n_ - b, (int) l_size - 1), b);
    assert(checker.status == RESULT_CONTINUE_LEFT || checker.status == RESULT_FOUND);

    while (true) {
        int j = (a + b) / 2;
        int i = min(n_ - j, (int) l_size - 1);

        checker.check(i, j);
        if (checker.status == RESULT_FOUND) {
            splits[k] = {i, j, checker.sigma + h[j].getWeight(weights.data) - weightPerBucket};
            return;
        } else if (checker.status == RESULT_CONTINUE_RIGHT) {
            assert(j + 1 > a);
            a = j + 1;
        } else {
            assert(j < b);
            b = j;
        }
    }
}