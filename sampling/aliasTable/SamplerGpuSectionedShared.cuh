#ifndef ALIAS_GPU_SAMPLER_GPU_SECTIONED_SHARED_CUH
#define ALIAS_GPU_SAMPLER_GPU_SECTIONED_SHARED_CUH

#include "sampling/Sampler.cuh"
#include "sampling/aliasTable/SamplerGpuBasic.cuh"
#include "sampling/aliasTable/SamplerGpuSectioned.cuh"
#include "construction/aliasTable/AliasTable.cuh"

/**
 * Samples on the GPU from Alias Tables. Splits the samples into multiple sections.
 * Utilizes shared memory
 */
class SamplerGpuSectionedShared : public SamplerGpu<AliasTable<ArrayOfStructs>> {
    public:
        explicit SamplerGpuSectionedShared(AliasTable<ArrayOfStructs> &samplingAlgorithm,
                                           int itemsPerSection = SECTION_SIZE_AUTO, bool limitGroups = false);
        std::string name() override;
        double executeBenchmarkSampling(int numSamples, int *distributionOutput) override;

        __device__ __forceinline__
        static int sample(ArrayOfStructs::TableRow *shared_section, double W_N, int length, XorWowPrng &state, int rowOffset);
    private:
        int itemsPerSection;
        bool limitGroups;
};

#endif //ALIAS_GPU_SAMPLER_GPU_SECTIONED_SHARED_CUH
