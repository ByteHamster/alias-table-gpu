#include "RejectionSamplingBinarySearch.cuh"

std::string RejectionSamplingBinarySearch::name() {
    return "RejectionSamplingBinarySearch";
}

RejectionSamplingBinarySearch::RejectionSamplingBinarySearch(int size, WeightDistribution weightDistribution)
        : RejectionSampling(size, weightDistribution) {

}

__global__
void fillABinarySearch(SafeArray<int> A, SafeArray<int> elementsPerItem) {
    unsigned int num = blockIdx.x * REJECTION_THREADS_PER_BLOCK + threadIdx.x;
    if (num >= A.size) {
        return;
    }
    int left = 0;
    int right = elementsPerItem.size;
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

void RejectionSamplingBinarySearch::fillA() {
    fillABinarySearch<<<dim3(A.size / REJECTION_THREADS_PER_BLOCK + 1), dim3(REJECTION_THREADS_PER_BLOCK)>>>(
            A, elementsPerItem);
}
