#include "AliasTableNp.cuh"

template <typename TableStorage>
std::string AliasTableNp<TableStorage>::name() {
    return std::string("AliasTableNp") + TableStorage::name() + (useBestCase ? "BestCase" : "WorstCase");
}

template <typename TableStorage>
AliasTableNp<TableStorage>::AliasTableNp(int size, bool bestCase)
        : AliasTable<TableStorage>(size, weightDistributionBestWorstCase) {
    useBestCase = bestCase;
}

template <typename TableStorage>
void AliasTableNp<TableStorage>::build() {
    this->sumWeightsCpu();
    if (useBestCase) {
        buildPerfect();
    } else {
        buildWorstCase();
    }
}

template<typename TableStorage>
void AliasTableNp<TableStorage>::buildPerfect() {
    this->aliasTable.alias(0) = this->N - 1;
    this->aliasTable.weight(0) = 0.5 * this->W/this->N;

    for (int i = 1; i < this->N - 1; i++) {
        this->aliasTable.alias(i) = 42;
        this->aliasTable.weight(i) = this->W/this->N;
    }

    this->aliasTable.alias(this->N - 1) = 42;
    this->aliasTable.weight(this->N - 1) = this->W/this->N;
}

template<typename TableStorage>
void AliasTableNp<TableStorage>::buildWorstCase() {
    for (int i = 0; i < this->N - 1; i++) {
        this->aliasTable.alias(i) = i + 1;
        this->aliasTable.weight(i) = 0.5 * this->W/this->N;
    }

    this->aliasTable.alias(this->N - 1) = 42;
    this->aliasTable.weight(this->N - 1) = this->W/this->N;
}