#ifndef ALIAS_GPU_SAMPLERGPU_CUH
#define ALIAS_GPU_SAMPLERGPU_CUH

#include <construction/aliasTable/AliasTable.cuh>
#include <construction/rejectionSampling/RejectionSampling.cuh>
#include "sampling/Sampler.cuh"

#ifndef NDEBUG
#define DEBUG_SUPPORT_SAMPLING_DISTRIBUTION
#endif

template <typename SamplingAlgorithm>
class SamplerGpu : public Sampler {
    public:
        explicit SamplerGpu(SamplingAlgorithm &algorithm);
        std::vector<int> getSamplingDistribution(int numSamples) override;
    protected:
        SamplingAlgorithm &samplingAlgorithm;
};

template class SamplerGpu<AliasTable<ArrayOfStructs>>;
template class SamplerGpu<AliasTable<StructOfArrays>>;
template class SamplerGpu<RejectionSampling>;

#endif //ALIAS_GPU_SAMPLERGPU_CUH
