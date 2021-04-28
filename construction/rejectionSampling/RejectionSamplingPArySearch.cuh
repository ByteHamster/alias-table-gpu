#ifndef ALIAS_GPU_REJECTIONSAMPLING_PARY_CUH
#define ALIAS_GPU_REJECTIONSAMPLING_PARY_CUH

#include <curand_kernel.h>

#include "utils/Utils.cuh"
#include "utils/Timer.cuh"
#include "RejectionSampling.cuh"

class RejectionSamplingPArySearch : public RejectionSampling {
    public:
        explicit RejectionSamplingPArySearch(int size, WeightDistribution weightDistribution = weightDistributionSine);
        void fillA() override;
        std::string name() override;
};

#endif //ALIAS_GPU_REJECTIONSAMPLING_PARY_CUH
