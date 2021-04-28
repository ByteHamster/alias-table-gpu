#ifndef ALIAS_GPU_REJECTIONSAMPLING_DIRECT_CUH
#define ALIAS_GPU_REJECTIONSAMPLING_DIRECT_CUH

#include <curand_kernel.h>

#include "utils/Utils.cuh"
#include "utils/Timer.cuh"
#include "RejectionSampling.cuh"

class RejectionSamplingDirect : public RejectionSampling {
    public:
        explicit RejectionSamplingDirect(int size, WeightDistribution weightDistribution = weightDistributionSine);
        void fillA() override;
        std::string name() override;
};

#endif //ALIAS_GPU_REJECTIONSAMPLING_DIRECT_CUH
