#ifndef ALIAS_GPU_SAMPLER_GPU_SECTIONED_CUH
#define ALIAS_GPU_SAMPLER_GPU_SECTIONED_CUH

#include <random>
#include "sampling/Sampler.cuh"
#include "sampling/aliasTable/SamplerGpuBasic.cuh"
#include "construction/aliasTable/AliasTable.cuh"

struct SamplerSection {
    int start;
    int end;
    int numSamples;
};

#define SECTION_SIZE_AUTO -1
#define SET_CACHE_CONFIG

/**
 * Samples on the GPU from Alias Tables. Splits the samples into multiple sections.
 */
template <typename TableStorage>
class SamplerGpuSectioned : public SamplerGpu<AliasTable<TableStorage>> {
    public:
        explicit SamplerGpuSectioned(AliasTable<TableStorage> &samplingAlgorithm,
                                     int itemsPerSection = SECTION_SIZE_AUTO, bool limitGroups = false);
        std::string name() override;
        double executeBenchmarkSampling(int numSamples, int *distributionOutput) override;

        __device__
        static SamplerSection calculateSection(int N, int numSamples);

        __device__ __forceinline__
        static int sample(TableStorage table, double W_N, int offset, int length, XorWowPrng &random);
    private:
        int itemsPerSection;
        bool limitGroups;
};

template class SamplerGpuSectioned<ArrayOfStructs>;
template class SamplerGpuSectioned<StructOfArrays>;

#endif //ALIAS_GPU_SAMPLER_GPU_SECTIONED_CUH
