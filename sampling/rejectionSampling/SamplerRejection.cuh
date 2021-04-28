#ifndef ALIAS_GPU_SAMPLER_REJECTION_CUH
#define ALIAS_GPU_SAMPLER_REJECTION_CUH

#include "sampling/SamplerGpu.cuh"
#include "construction/rejectionSampling/RejectionSampling.cuh"

/**
 * Samples on the GPU from the rejection sampling structure.
 */
class SamplerRejection : public SamplerGpu<RejectionSampling> {
    public:
        explicit SamplerRejection(RejectionSampling &samplingAlgorithm);
        std::string name() override;
        double executeBenchmarkSampling(int numSamples, int *distributionOutput) override;

        __device__ __forceinline__
        static int sample(SafeArray<int> A, SafeArray<double> weights, int N, double W, XorWowPrng &random);
};


#endif //ALIAS_GPU_SAMPLER_REJECTION_CUH
