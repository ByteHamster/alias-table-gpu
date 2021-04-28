#include "RejectionSampling.cuh"

RejectionSampling::RejectionSampling(int size, WeightDistribution weightDistribution)
        : SamplingAlgorithm(size, weightDistribution) {
    weightsGpu.malloc(weights.size);
}

RejectionSampling::~RejectionSampling() {
    weightsGpu.free();
    A.free();
}

void RejectionSampling::preBuild() {
    weightsGpu.copyFrom(weights);
    elementsPerItem.malloc(weights.size);
    cudaDeviceSynchronize();
}

bool RejectionSampling::postBuild() {
    elementsPerItem.free();
    ACpu.malloc(A.size);
    ACpu.copyFrom(A);
    return true;
}

struct CalcNumOccurrencesOperator {
    int N;
    double W;
    __host__ __device__ __forceinline__
    int operator()(const double &xi) const {
        return floor((double) (N * xi) / (double) W) + 1;
    }
};

void RejectionSampling::build() {
    sumWeightsCpu();

    void *tempStorage = nullptr;
    size_t tempStorageSize = 0;
    CalcNumOccurrencesOperator calcNumOccurrencesOperator = {};
    calcNumOccurrencesOperator.N = N;
    calcNumOccurrencesOperator.W = W;
    cub::TransformInputIterator<int, CalcNumOccurrencesOperator, double*> calcNumOccurrencesIterator(weightsGpu.data, calcNumOccurrencesOperator);
    ERRCHECK(cub::DeviceScan::InclusiveSum(tempStorage, tempStorageSize, calcNumOccurrencesIterator, elementsPerItem.data, elementsPerItem.size))
    cudaMalloc(&tempStorage, tempStorageSize);
    ERRCHECK(cub::DeviceScan::InclusiveSum(tempStorage, tempStorageSize, calcNumOccurrencesIterator, elementsPerItem.data, elementsPerItem.size))
    cudaFree(tempStorage);

    int totalElements;
    ERRCHECK(cudaMemcpy(&totalElements, &elementsPerItem[elementsPerItem.size - 1], sizeof(int), cudaMemcpyDeviceToHost))
    if (A.size == 0) {
        // Allow benchmarks
        A.malloc(totalElements);
    }
    fillA();
}

float RejectionSampling::benchmarkBuild(double weightOfOutlier) {
    weights[0] = weightOfOutlier;
    preBuild();
    Timer timer;
    timer.start();
    build();
    for (int i = 0; i < BENCHMARK_BUILD_SPEED_ITERATIONS; i++) {
        fillA();
    }
    timer.stop();
    postBuild();
    return timer.elapsedMillis() / BENCHMARK_BUILD_SPEED_ITERATIONS;
}
