#include "AliasTable.cuh"

template <typename TableStorage>
AliasTable<TableStorage>::AliasTable(int size, WeightDistribution weightDistribution)
        : SamplingAlgorithm(size, weightDistribution) {
    aliasTable.malloc(N);
}

template <typename TableStorage>
AliasTable<TableStorage>::~AliasTable() {
    aliasTable.free();
}

template <typename TableStorage>
void AliasTable<TableStorage>::preBuild() {

}

template <typename TableStorage>
bool AliasTable<TableStorage>::postBuild() {
    return verifyTableCorrectness();
}

template <typename TableStorage>
bool AliasTable<TableStorage>::verifyTableCorrectness() {
    #ifdef NDEBUG
        return true;
    #endif

    bool okay = true;
    std::vector<double> listedWeight(N, 0.0);
    double maxError = std::max(N * 1e-8, 1e-3);
    double W_N = W / N;
    for (int i = 0; i < N; i++) {
        if (aliasTable.weight(i) < 0) {
            if (okay) {
                std::cerr<<std::endl;
            }
            std::cerr<<name()<<": Listed weight "<<i<<" too small: "<<(aliasTable.weight(i)/W_N)<<std::endl;
            okay = false;
        } else if (aliasTable.weight(i) > W_N * (1.0 + maxError)) {
            if (okay) {
                std::cerr<<std::endl;
            }
            std::cerr<<name()<<": Listed weight "<<i<<" too big: "<<(aliasTable.weight(i)/W_N)<<std::endl;
            okay = false;
        }
        listedWeight[i] += aliasTable.weight(i);
        if (0 <= aliasTable.alias(i) && aliasTable.alias(i) < N) { // Uninitialized when weight = 1
            listedWeight[aliasTable.alias(i)] += W_N - aliasTable.weight(i);
        }
    }

    for (int i = 0; i < N; i++) {
        double delta = listedWeight[i] - weights[i];
        if (std::abs(delta) > maxError) {
            if (okay) {
                std::cerr<<std::endl;
            }
            std::cerr<<name()<<": Listed weight "<<i<<" off by "<<delta<<std::endl;
            okay = false;
        }
    }
    return okay;
}
