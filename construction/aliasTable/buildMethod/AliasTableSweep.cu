#include "AliasTableSweep.cuh"

std::string AliasTableSweep::name() {
    return "AliasTableSweep";
}

AliasTableSweep::AliasTableSweep(int size, WeightDistribution weightDistribution)
        : AliasTable(size + 2, weightDistribution) {
    N = size; // Initialized weights with 2 extra elements
}

void AliasTableSweep::build() {
    sumWeightsCpu();
    weights[N] = std::numeric_limits<double>::infinity();
    weights[N + 1] = 0;

    int i = 0;
    while (weights[i] > W/N) {
        i++;
    }
    int j = 0;
    while (weights[j] <= W/N) {
        j++;
    }

    double w = weights[j];
    while (true) {
        bool weightExhausted = w <= W/N + EPSILON;
        if (weightExhausted && j >= N) {
            weightExhausted = false;
        } else if (!weightExhausted && i >= N) {
            weightExhausted = true;
        }
        if (weightExhausted) {
            if (j >= N) {
                return;
            }

            aliasTable.weight(j) = w;
            int j_ = j;
            do {
                j_++;
            } while (j < N + 2 && weights[j_] <= W / N);
            aliasTable.alias(j) = j_;
            w += weights[j_] - W / N;
            j = j_;
        } else {
            if (i >= N) {
                return;
            }

            aliasTable.weight(i) = weights[i];
            aliasTable.alias(i) = j;
            w += weights[i] - W / N;
            do {
                i++;
            } while (i < N + 2 && weights[i] > W / N);
        }
    }
}
