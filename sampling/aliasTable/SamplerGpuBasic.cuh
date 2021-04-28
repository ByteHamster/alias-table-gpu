#ifndef ALIAS_GPU_SAMPLER_GPU_BASIC_CUH
#define ALIAS_GPU_SAMPLER_GPU_BASIC_CUH

#include <type_traits>
#include "sampling/SamplerGpu.cuh"
#include "construction/aliasTable/AliasTable.cuh"

#define DIMENS_DEFAULT -1

/**
 * Samples on the GPU from Alias Tables.
 */
template <typename TableStorage, typename Prng = XorWowPrng>
class SamplerGpuBasic : public SamplerGpu<AliasTable<TableStorage>> {
    public:
        explicit SamplerGpuBasic(AliasTable<TableStorage> &samplingAlgorithm,
                                 int numBlocks = DIMENS_DEFAULT, int numThreads = DIMENS_DEFAULT);
        std::string name() override;
        double executeBenchmarkSampling(int numSamples, int *distributionOutput) override;
        __device__ __forceinline__
        static int sample(TableStorage table, double W_N, int N, Prng &random);
    private:
        int numBlocks;
        int numThreads;
};

template class SamplerGpuBasic<ArrayOfStructs, XorWowPrng>;
template class SamplerGpuBasic<ArrayOfStructs, XorWowPkPtPrng<HashNone>>;
template class SamplerGpuBasic<ArrayOfStructs, XorWowPkPtPrng<HashSha1>>;
template class SamplerGpuBasic<ArrayOfStructs, XorWowPkPtPrng<HashMd5>>;
template class SamplerGpuBasic<ArrayOfStructs, MtPrng>;
template class SamplerGpuBasic<StructOfArrays, XorWowPrng>;
template class SamplerGpuBasic<StructOfArrays, XorWowPkPtPrng<HashNone>>;
template class SamplerGpuBasic<StructOfArrays, XorWowPkPtPrng<HashSha1>>;
template class SamplerGpuBasic<StructOfArrays, XorWowPkPtPrng<HashMd5>>;
template class SamplerGpuBasic<StructOfArrays, MtPrng>;

#endif //ALIAS_GPU_SAMPLER_GPU_BASIC_CUH
