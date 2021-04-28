#include "SplitMethodParySearch.cuh"

__device__
int getSplitterOfThread(int a, int b, int thread) {
    // Must be reproducible, so using long instead of double
    return a + ((long) (b - a) * thread) / PARY_SEARCH_GROUP_SIZE;
}

__global__
void SplitMethodParySearch::splitKernel(int splitOffset, SafeArray<SplitConfig> splits, int N, double W, int p,
                                    SafeArray<double> weights, SafeArray<LH_TYPE> h, SafeArray<double> prefixWeightH,
                                    int h_size, SafeArray<double> prefixWeightL, int l_size) {
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x + 1 + splitOffset;

    if (k == 1) {
        splits[0] = {0, 0, 0};
        splits[p] = {(int) l_size - 1, (int) h_size - 1, 0};
    }

    unsigned int groupMin_k = blockIdx.x * blockDim.x + 0 + 1 + splitOffset;
    if (groupMin_k >= p) {
        return; // Too many groups were started
    }
    unsigned int groupMax_k = blockIdx.x * blockDim.x + (blockDim.x - 1) + 1 + splitOffset;
    groupMax_k = min(groupMax_k, p - 1);
    assert(groupMin_k <= groupMax_k);

    int groupMin_n_ = ceil(((double) N * groupMin_k) / p);
    int groupMax_n_ = ceil(((double) N * groupMax_k) / p);
    assert(groupMin_n_ >= 0);
    assert(groupMax_n_ <= N);
    assert(groupMin_n_ <= groupMax_n_);
    double groupMin_weightPerBucket = (W * groupMin_n_)/N;
    double groupMax_weightPerBucket = (W * groupMax_n_)/N;

    //////////// P-ary search for items of whole group to reduce binary search range ////////////
    assert(blockDim.x == PARY_SEARCH_GROUP_SIZE);
    __shared__ char checkResults[PARY_SEARCH_GROUP_SIZE];

    int a = 0;
    int b = min(groupMax_n_, (int) h_size) - 1;
    assert(a < b);

    SplitMethodBasic::BinarySearchChecker groupMin_checker(groupMin_weightPerBucket, prefixWeightH, prefixWeightL);
    SplitMethodBasic::BinarySearchChecker groupMax_checker(groupMax_weightPerBucket, prefixWeightH, prefixWeightL);

    while (true) {
        int j = getSplitterOfThread(a, b, threadIdx.x);
        int groupMin_i = min(groupMin_n_ - j, (int) l_size - 1);
        int groupMax_i = min(groupMax_n_ - j, (int) l_size - 1);

        groupMin_checker.check(groupMin_i, j);
        groupMax_checker.check(groupMax_i, j);

        if (groupMax_checker.status == RESULT_CONTINUE_LEFT) {
            // Everything right from this group does not need to be looked at
            checkResults[threadIdx.x] = RESULT_CONTINUE_LEFT;
        } else if (groupMin_checker.status == RESULT_CONTINUE_RIGHT) {
            // Everything left from this group does not need to be looked at
            checkResults[threadIdx.x] = RESULT_CONTINUE_RIGHT;
        } else {
            checkResults[threadIdx.x] = RESULT_FOUND;
        }

        __shared__ int maxTooBigSectionShared;
        __shared__ int minTooSmallSectionShared;

        #ifndef NDEBUG
            __syncthreads();
            if (threadIdx.x == 0) {
                // Validity check: RIGHT...FOUND...LEFT, not mixed.
                int i = 0;
                while (i < blockDim.x && checkResults[i] == RESULT_CONTINUE_RIGHT) { i++; }
                while (i < blockDim.x && checkResults[i] == RESULT_FOUND) { i++; }
                while (i < blockDim.x && checkResults[i] == RESULT_CONTINUE_LEFT) { i++; }
                assert(i == blockDim.x);
                // Validity check end
            }
        #endif

        if (threadIdx.x == 0) {
            // If all sections are RESULT_CONTINUE_RIGHT, minTooSmallSectionShared would not be filled otherwise.
            maxTooBigSectionShared = 0;
            minTooSmallSectionShared = blockDim.x;
        }
        __syncthreads();

        if (checkResults[threadIdx.x] == RESULT_CONTINUE_RIGHT
            && (threadIdx.x == PARY_SEARCH_GROUP_SIZE - 1
                || (threadIdx.x < PARY_SEARCH_GROUP_SIZE - 1 && checkResults[threadIdx.x + 1] != RESULT_CONTINUE_RIGHT))) {
            // It checks the first item of a section => -1
            maxTooBigSectionShared = max(0, threadIdx.x - 1);
        }
        if (checkResults[threadIdx.x] == RESULT_CONTINUE_LEFT
            && (threadIdx.x == 0
                || (threadIdx.x > 0 && checkResults[threadIdx.x - 1] != RESULT_CONTINUE_LEFT))) {
            minTooSmallSectionShared = threadIdx.x;
        }
        __syncthreads();

        int aNew = getSplitterOfThread(a, b, maxTooBigSectionShared);
        int bNew = getSplitterOfThread(a, b, minTooSmallSectionShared);

        a = max(a, aNew - 1);
        b = min(b, bNew + 1);
        assert(a < b);

        if (minTooSmallSectionShared - maxTooBigSectionShared > PARY_SEARCH_GROUP_SIZE / 32) {
            // Stop searching: Most of the threads returned RESULT_FOUND
            break;
        }
    }

    if (k >= p) {
        return;
    }

    //////////// Normal binary search ////////////

    int n_ = ceil(((double) N * k) / p);
    assert(n_ <= groupMax_n_);
    b = min(b, min(n_, (int) h_size) - 1);

    assert(a >= 0);
    assert(a < b);

    double weightPerBucket = (W * n_)/N;
    SplitMethodBasic::BinarySearchChecker checker(weightPerBucket, prefixWeightH, prefixWeightL);
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