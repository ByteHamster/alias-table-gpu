#include "RejectionSamplingDirect.cuh"

std::string RejectionSamplingDirect::name() {
    return "RejectionSamplingDirect";
}

RejectionSamplingDirect::RejectionSamplingDirect(int size, WeightDistribution weightDistribution)
        : RejectionSampling(size, weightDistribution) {

}

__global__
void fillADirect(SafeArray<int> A, SafeArray<int> elementsPerItem) {
    unsigned int num = blockIdx.x * REJECTION_THREADS_PER_BLOCK + threadIdx.x;
    if (num >= elementsPerItem.size) {
        return;
    }
    int numbersToWrite;
    if (num == 0) {
        numbersToWrite = elementsPerItem[0];
    } else {
        numbersToWrite = elementsPerItem[num] - elementsPerItem[num - 1];
    }
    int lastIdx = elementsPerItem[num];
    for (;numbersToWrite > 0; numbersToWrite--) {
        A[lastIdx - numbersToWrite] = num;
    }
}

void RejectionSamplingDirect::fillA() {
    fillADirect<<<dim3(A.size / REJECTION_THREADS_PER_BLOCK + 1), dim3(REJECTION_THREADS_PER_BLOCK)>>>(
            A, elementsPerItem);
}
