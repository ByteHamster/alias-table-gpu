#ifndef ALIAS_GPU_SAMPLER_EXPECTED_CUH
#define ALIAS_GPU_SAMPLER_EXPECTED_CUH

#include "sampling/Sampler.cuh"

/**
 * Returns the expected distribution without actually drawing random numbers
 */
class SamplerExpected : public Sampler {
    public:
        explicit SamplerExpected(SamplingAlgorithm &samplingAlgorithm);
        std::string name() override;
        double executeBenchmarkSampling(int numSamples, int *distributionOutput) override;
        std::vector<int> getSamplingDistribution(int numSamples) override;
    private:
        SamplingAlgorithm &samplingAlgorithm;
};


#endif //ALIAS_GPU_SAMPLER_EXPECTED_CUH
