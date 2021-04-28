#ifndef ALIAS_GPU_SAMPLING_ALGORITHM_CUH
#define ALIAS_GPU_SAMPLING_ALGORITHM_CUH

#include <curand_kernel.h>
#include <algorithm>
#include <sstream>
#include <random>
#include "utils/Utils.cuh"
#include "utils/Timer.cuh"
#include "utils/SafeArray.cuh"
#include "utils/Adder.cuh"

#define BENCHMARK_BUILD_SPEED_ITERATIONS 500
#define EPSILON 1e-10

enum WeightDistribution {
    weightDistributionSine = 1,
    weightDistributionBestWorstCase = 2,
    weightDistributionUniform = 3,
    weightDistributionRamp = 4,
    weightDistributionPowerLaw1 = 5,
    weightDistributionPowerLaw2 = 6,
    weightDistributionOneHeavy = 7,
    weightDistributionOneLight = 8,
    weightDistributionPowerLaw1Shuffled = 9,
    weightDistributionPowerLaw05Shuffled = 10
};
#define MAX_WEIGHT_DISTRIBUTION 10

class SamplingAlgorithm {
    public:
        explicit SamplingAlgorithm(int size, WeightDistribution weightDistribution);
        ~SamplingAlgorithm();
        virtual void preBuild() = 0;
        virtual void build() = 0;
        virtual bool postBuild() = 0;
        virtual std::string name() = 0;
        bool fullBuild();

        SafeArray<double> weights = SafeArray<double>(HOST);
        double W = 0;
        int N = 1000;
    protected:
        void sumWeightsCpu();
    private:
        void generateWeights();
        WeightDistribution weightDistribution;
};

#endif //ALIAS_GPU_SAMPLING_ALGORITHM_CUH
