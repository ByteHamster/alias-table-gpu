#ifndef ALIAS_GPU_SAMPLER_CUH
#define ALIAS_GPU_SAMPLER_CUH

#include "construction/SamplingAlgorithm.cuh"
#include "utils/Prng.cuh"

/**
 * An algorithm that can sample from a sampling data structure.
 */
class Sampler {
    public:
        /**
         * Name of this algorithm for CSV output.
         */
        virtual std::string name() = 0;

        /**
         * Returns the number of Giga samples per second.
         */
        double benchmarkSampling(int numSamples) {
            return executeBenchmarkSampling(numSamples, nullptr);
        }

        /**
         * Samples multiple times and returns how many times each item was sampled.
         */
        virtual std::vector<int> getSamplingDistribution(int numSamples) = 0;

    protected:
        virtual double executeBenchmarkSampling(int numSamples, int *distributionOutput) = 0;
};


#endif //ALIAS_GPU_SAMPLER_CUH
