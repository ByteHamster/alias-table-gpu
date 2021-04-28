#include "SamplingAlgorithm.cuh"

SamplingAlgorithm::SamplingAlgorithm(int size, WeightDistribution weightDistribution)
        : weightDistribution(weightDistribution) {
    N = size;
    weights.malloc(N);
    generateWeights();
}

SamplingAlgorithm::~SamplingAlgorithm() {
    weights.free();
}

void SamplingAlgorithm::sumWeightsCpu() {
    W = 0;
    for (int i = 0; i < N; i++) {
        W += weights[i];
    }
}

void SamplingAlgorithm::generateWeights() {
    for (int i = 0; i < N; i++) {
        double x = i + 1;
        switch (weightDistribution) {
            case weightDistributionSine:
                weights[i] = 20.f * (1.1f + cos(0.008 * i));
                break;
            case weightDistributionBestWorstCase:
                if (i == 0) {
                    weights[i] = 0.5;
                } else if (i == N - 1) {
                    weights[i] = 1.5;
                } else {
                    weights[i] = 1.0;
                }
                break;
            case weightDistributionOneHeavy:
                weights[i] = (i == 0) ? 1.5 : 1.0;
                break;
            case weightDistributionOneLight:
                weights[i] = (i == 0) ? 0.5 : 1.0;
                break;
            case weightDistributionRamp:
                weights[i] = i;
                break;
            case weightDistributionUniform:
                weights[i] = 10.0 * ((double) (int) rand() / RAND_MAX);
                break;
            case weightDistributionPowerLaw1:
                weights[i] = 1.0 / x;
                break;
            case weightDistributionPowerLaw1Shuffled:
                weights[i] = 1.0 / x;
                break;
            case weightDistributionPowerLaw05Shuffled:
                weights[i] = 1.0 / sqrt(x);
                break;
            case weightDistributionPowerLaw2:
                weights[i] = 1.0 / (x * x);
                break;
            default:
                assert(false && "Unknown distribution");
        }
        assert(!isinf(weights[i]));
    }
    if (weightDistribution == weightDistributionPowerLaw1Shuffled
            || weightDistribution == weightDistributionPowerLaw05Shuffled) {
        std::shuffle(weights.data, weights.data + N,
                     std::default_random_engine(time(nullptr)));
    }
}

bool SamplingAlgorithm::fullBuild() {
    preBuild();
    build();
    return postBuild();
}
