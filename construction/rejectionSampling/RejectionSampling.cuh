#ifndef ALIAS_GPU_REJECTIONSAMPLING_CUH
#define ALIAS_GPU_REJECTIONSAMPLING_CUH

#include <curand_kernel.h>
#include <cub/iterator/transform_input_iterator.cuh>

#include "utils/Utils.cuh"
#include "utils/Timer.cuh"
#include "construction/SamplingAlgorithm.cuh"

#define REJECTION_THREADS_PER_BLOCK 512

class RejectionSampling : public SamplingAlgorithm {
    public:
        explicit RejectionSampling(int size, WeightDistribution weightDistribution = weightDistributionSine);
        ~RejectionSampling();
        void preBuild() override;
        void build() override;
        bool postBuild() override;
        std::string name() override = 0;

        float benchmarkBuild(double weightOfOutlier);
        virtual void fillA() = 0;

        SafeArray<int> ACpu = SafeArray<int>(HOST);
        SafeArray<int> A = SafeArray<int>(DEVICE);
        SafeArray<int> elementsPerItem = SafeArray<int>(DEVICE);
        SafeArray<double> weightsGpu = SafeArray<double>(DEVICE);
};

#endif //ALIAS_GPU_REJECTIONSAMPLING_CUH
