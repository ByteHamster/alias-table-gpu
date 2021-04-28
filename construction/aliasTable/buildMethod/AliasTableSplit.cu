#include "AliasTableSplit.cuh"

template <typename TableStorage>
std::string AliasTableSplit<TableStorage>::name() {
    return "AliasTableSplit" + TableStorage::name();
}

template <typename TableStorage>
AliasTableSplit<TableStorage>::AliasTableSplit(int size, int _numThreads, WeightDistribution weightDistribution)
        : AliasTable<TableStorage>(size + 2, weightDistribution) {
    this->N = size; // Initialized weights with 2 extra elements
    numThreads = _numThreads;

    if (numThreads > this->N / 2) {
        std::cerr << "Table size " << this->N << " can not be used with " << numThreads << " threads!" << std::endl;
        numThreads = this->N / 10;
    }
}

template <typename TableStorage>
AliasTableSplit<TableStorage>::~AliasTableSplit() = default;

template <typename TableStorage>
void AliasTableSplit<TableStorage>::build() {
    this->sumWeightsCpu();
    this->weights[this->N] = std::numeric_limits<double>::infinity();
    this->weights[this->N + 1] = 0;

    for (int i = 0; i < this->N; i++) {
        this->aliasTable.weight(i) = this->weights[i];
    }

    int p = numThreads;

    SafeArray<int> prefixNumberOfHeavyItems(this->N + 2, HOST);

    for (int i = 0; i < this->N + 2; i++) {
        prefixNumberOfHeavyItems[i] = (this->weights[i] > this->W / this->N) ? 1 : 0;
    }
    PrefixSum::exclusivePrefixSum<int>(prefixNumberOfHeavyItems, prefixNumberOfHeavyItems);
    int numH = prefixNumberOfHeavyItems[this->N + 1];
    int numL = (this->N + 2) - numH;
    assert(numH > 1);
    assert(numL > 1);

    l.malloc(numL);
    prefixWeightL.malloc(numL);
    h.malloc(numH);
    prefixWeightH.malloc(numH);
    SafeArray<SplitConfig> splits(p + 1, HOST);

    for (int i = 0; i < this->N + 2; i++) {
        if (this->weights[i] > this->W/this->N) {
            h[prefixNumberOfHeavyItems[i]].item = i;
            h[prefixNumberOfHeavyItems[i]].setWeight(this->weights[i]);
            prefixWeightH[prefixNumberOfHeavyItems[i]] = this->weights[i];
        } else {
            l[i - prefixNumberOfHeavyItems[i]].item = i;
            l[i - prefixNumberOfHeavyItems[i]].setWeight(this->weights[i]);
            prefixWeightL[i - prefixNumberOfHeavyItems[i]] = this->weights[i];
        }
    }

    PrefixSum::exclusivePrefixSum<double>(prefixWeightL, prefixWeightL);
    PrefixSum::exclusivePrefixSum<double>(prefixWeightH, prefixWeightH);

    for (int k = 1; k <= p - 1; k++) {
        SplitMethodBasic::split(k, splits, this->N, this->W, p, this->weights, h, prefixWeightH, numH, prefixWeightL, numL);
    }
    for (int k = 1; k <= p; k++) {
        PackMethodSweep::pack(k, splits, this->W/this->N, h, l, this->aliasTable, this->weights);
    }

    splits.free();
    l.free();
    prefixWeightL.free();
    h.free();
    prefixWeightH.free();
    prefixNumberOfHeavyItems.free();
}

std::ostream& operator<<(std::ostream &os, SplitConfig &split) {
    os << "(i=" << split.i << "\tj=" << split.j << "\tspill=" << split.spill << ")";
    return os;
}
