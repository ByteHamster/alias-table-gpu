#ifndef ALIAS_GPU_SAMPLER_REJECTION_GIVE_UP_CUH
#define ALIAS_GPU_SAMPLER_REJECTION_GIVE_UP_CUH

#include "sampling/SamplerGpu.cuh"
#include "construction/rejectionSampling/RejectionSampling.cuh"

#define MAX_TRIES 8
#define SAMPLE_SKIPPED -1

/**
 * Samples on the GPU from the rejection sampling structure.
 * If a sample can't be drawn in MAX_TRIES iterations, just give up and try again next time.
 */
class SamplerRejectionGiveUp : public SamplerGpu<RejectionSampling> {
    public:
        explicit SamplerRejectionGiveUp(RejectionSampling &samplingAlgorithm);
        std::string name() override;
        double executeBenchmarkSampling(int numSamples, int *distributionOutput) override;

        __device__ __forceinline__
        static int sample(SafeArray<int> A, SafeArray<double> weights, int N, double W, XorWowPrng &random);
};


#endif //ALIAS_GPU_SAMPLER_REJECTION_GIVE_UP_CUH
