#ifndef ALIAS_GPU_SAMPLER_CPU_CUH
#define ALIAS_GPU_SAMPLER_CPU_CUH

#include "sampling/Sampler.cuh"
#include "construction/aliasTable/AliasTable.cuh"

/**
 * Samples from Alias Tables on the CPU without performance optimization
 */
class SamplerCpu : public Sampler {
    public:
        explicit SamplerCpu(AliasTable<ArrayOfStructs> &samplingAlgorithm);
        std::string name() override;
        double executeBenchmarkSampling(int numSamples, int *distributionOutput) override;
        std::vector<int> getSamplingDistribution(int numSamples) override;

        /**
         * Samples one single number.
         */
        virtual int sample();
    private:
        AliasTable<ArrayOfStructs> &samplingAlgorithm;
        CpuPrng random = CpuPrng();
};


#endif //ALIAS_GPU_SAMPLER_CPU_CUH
