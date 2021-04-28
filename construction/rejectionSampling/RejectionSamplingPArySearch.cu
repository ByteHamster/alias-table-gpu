#include "RejectionSamplingPArySearch.cuh"

std::string RejectionSamplingPArySearch::name() {
    return "RejectionSamplingPArySearch";
}

RejectionSamplingPArySearch::RejectionSamplingPArySearch(int size, WeightDistribution weightDistribution)
        : RejectionSampling(size, weightDistribution) {

}

__device__
int getRejectionSplitterOfThread(int a, int b, int thread) {
    // Must be reproducible, so using long instead of double
    return a + ((long) (b - a) * thread) / REJECTION_THREADS_PER_BLOCK;
}

__global__
void fillAPArySearch(SafeArray<int> A, SafeArray<int> elementsPerItem) {
    unsigned int numBlockMin = blockIdx.x * REJECTION_THREADS_PER_BLOCK;
    unsigned int numBlockMax = blockIdx.x * REJECTION_THREADS_PER_BLOCK + REJECTION_THREADS_PER_BLOCK - 1;
    numBlockMin = min(numBlockMin, (unsigned int) A.size - 1);
    numBlockMax = min(numBlockMax, (unsigned int) A.size - 1);

    assert(blockDim.x == REJECTION_THREADS_PER_BLOCK);
    __shared__ char checkResults[REJECTION_THREADS_PER_BLOCK];

    int a = 0;
    int b = elementsPerItem.size;
    assert(a < b);

    //////////// P-ary search for items of whole group to reduce binary search range ////////////
    while (true) {
        int j = getRejectionSplitterOfThread(a, b, threadIdx.x);
        int checkItem = elementsPerItem[j];

        if (checkItem >= numBlockMax) {
            // Everything right from this group does not need to be looked at
            checkResults[threadIdx.x] = RESULT_CONTINUE_LEFT;
        } else if (checkItem <= numBlockMin) {
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
            && (threadIdx.x == REJECTION_THREADS_PER_BLOCK - 1
                || (threadIdx.x < REJECTION_THREADS_PER_BLOCK - 1 && checkResults[threadIdx.x + 1] != RESULT_CONTINUE_RIGHT))) {
            // It checks the first item of a section => -1
            maxTooBigSectionShared = max(0, threadIdx.x - 1);
        }
        if (checkResults[threadIdx.x] == RESULT_CONTINUE_LEFT
            && (threadIdx.x == 0
                || (threadIdx.x > 0 && checkResults[threadIdx.x - 1] != RESULT_CONTINUE_LEFT))) {
            minTooSmallSectionShared = threadIdx.x;
        }
        __syncthreads();

        int aNew = getRejectionSplitterOfThread(a, b, maxTooBigSectionShared);
        int bNew = getRejectionSplitterOfThread(a, b, minTooSmallSectionShared);

        a = max(a, aNew - 1);
        b = min(b, bNew + 1);
        assert(a < b);

        if (minTooSmallSectionShared - maxTooBigSectionShared > REJECTION_THREADS_PER_BLOCK / 64) {
            // Stop searching: Most of the threads returned RESULT_FOUND
            break;
        }
    }

    unsigned int num = blockIdx.x * REJECTION_THREADS_PER_BLOCK + threadIdx.x;
    if (num >= A.size) {
        return;
    }

    //////////// Normal binary search ////////////

    int left = a;
    int right = b;
    while (right - left > 1) {
        int centerIdx = (right + left) / 2;
        if (elementsPerItem[centerIdx] > num) {
            right = centerIdx;
        } else {
            left = centerIdx;
        }
    }
    if (elementsPerItem[left] <= num) {
        A[num] = right;
    } else {
        A[num] = left;
    }
}

void RejectionSamplingPArySearch::fillA() {
    fillAPArySearch<<<dim3(A.size / REJECTION_THREADS_PER_BLOCK + 1), dim3(REJECTION_THREADS_PER_BLOCK)>>>(
            A, elementsPerItem);
}
