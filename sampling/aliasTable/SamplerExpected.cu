#include "SamplerExpected.cuh"

SamplerExpected::SamplerExpected(SamplingAlgorithm &samplingAlgorithm)
        : samplingAlgorithm(samplingAlgorithm) {

}

std::string SamplerExpected::name() {
    return "SamplerExpected";
}

double SamplerExpected::executeBenchmarkSampling(int numSamples, int *distributionOutput) {
    assert(false && "Benchmarking expected distribution does not make sense");
    return 0;
}

std::vector<int> SamplerExpected::getSamplingDistribution(int numSamples) {
    assert(samplingAlgorithm.W > 0);
    std::vector<int> output(samplingAlgorithm.N);
    for (int i = 0; i < samplingAlgorithm.N; i++) {
        output.at(i) = (int)((samplingAlgorithm.weights[i]/samplingAlgorithm.W) * numSamples);
    }
    return output;
}
