#ifndef ALIAS_GPU_ALIASTABLE_CUH
#define ALIAS_GPU_ALIASTABLE_CUH

#include <curand_kernel.h>
#include <algorithm>
#include <sstream>

#include "utils/Utils.cuh"
#include "utils/Timer.cuh"
#include "construction/SamplingAlgorithm.cuh"
#include "TableStorage.cuh"

template <typename TableStorage>
class AliasTable : public SamplingAlgorithm {
    public:
        explicit AliasTable(int size, WeightDistribution weightDistribution = weightDistributionSine);
        ~AliasTable();
        void preBuild() override;
        void build() override = 0;
        bool postBuild() override;
        std::string name() override = 0;
        bool verifyTableCorrectness();
        TableStorage aliasTable = TableStorage(HOST);
};

template class AliasTable<ArrayOfStructs>;
template class AliasTable<StructOfArrays>;

#endif //ALIAS_GPU_ALIASTABLE_CUH
