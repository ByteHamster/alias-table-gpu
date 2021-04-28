#ifndef ALIAS_GPU_REJECTIONSAMPLING_BINARY_CUH
#define ALIAS_GPU_REJECTIONSAMPLING_BINARY_CUH

#include <curand_kernel.h>

#include "utils/Utils.cuh"
#include "utils/Timer.cuh"
#include "RejectionSampling.cuh"

class RejectionSamplingBinarySearch : public RejectionSampling {
    public:
        explicit RejectionSamplingBinarySearch(int size, WeightDistribution weightDistribution = weightDistributionSine);
        void fillA() override;
        std::string name() override;
};

#endif //ALIAS_GPU_REJECTIONSAMPLING_BINARY_CUH
