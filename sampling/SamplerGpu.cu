#include "SamplerGpu.cuh"

template <typename SamplingAlgorithm>
SamplerGpu<SamplingAlgorithm>::SamplerGpu(SamplingAlgorithm &algorithm)
    : samplingAlgorithm(algorithm) {

}

template <typename SamplingAlgorithm>
std::vector<int> SamplerGpu<SamplingAlgorithm>::getSamplingDistribution(int numSamples) {
    #ifndef DEBUG_SUPPORT_SAMPLING_DISTRIBUTION
        assert(false && "Can only test sampling distribution if DEBUG_SUPPORT_SAMPLING_DISTRIBUTION is defined");
        return std::vector<int>(0);
    #else
        int *distributionOutput;
        ERRCHECK(cudaMalloc(&distributionOutput, samplingAlgorithm.N * sizeof(int)))
        ERRCHECK(cudaMemset(distributionOutput, 0, samplingAlgorithm.N * sizeof(int)))
        executeBenchmarkSampling(numSamples, distributionOutput);
        int distributionOutputHost[samplingAlgorithm.N];
        ERRCHECK(cudaMemcpy(distributionOutputHost, distributionOutput, samplingAlgorithm.N * sizeof(int), cudaMemcpyDeviceToHost));
        ERRCHECK(cudaFree(distributionOutput));
        return std::vector<int>(distributionOutputHost, distributionOutputHost + samplingAlgorithm.N);
    #endif
}