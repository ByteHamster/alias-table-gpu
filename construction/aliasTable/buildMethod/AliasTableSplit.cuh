#ifndef ALIAS_GPU_ALIASTABLESPLIT_CUH
#define ALIAS_GPU_ALIASTABLESPLIT_CUH

#include <curand_kernel.h>
#include <algorithm>

#include "utils/Utils.cuh"
#include "utils/Timer.cuh"
#include "utils/PrefixSum.cuh"
#include "construction/aliasTable/AliasTable.cuh"
#include "construction/aliasTable/packMethod/PackMethodBasic.cuh"
#include "construction/aliasTable/packMethod/PackMethodSweep.cuh"
#include "construction/aliasTable/splitMethod/SplitMethodBasic.cuh"
#include "construction/aliasTable/buildMethod/SplitConfig.cuh"
#include "construction/aliasTable/buildMethod/LhType.cuh"

template <typename TableStorage>
class AliasTableSplit : public AliasTable<TableStorage> {
    public:
        explicit AliasTableSplit(int size, int numThreads,
                                 WeightDistribution weightDistribution = weightDistributionSine);
        ~AliasTableSplit();
        void build() override;
        std::string name() override;
    protected:
        int numThreads;
    private:
        SafeArray<double> prefixWeightL = SafeArray<double>(HOST);
        SafeArray<double> prefixWeightH = SafeArray<double>(HOST);
        SafeArray<LH_TYPE> l = SafeArray<LH_TYPE>(HOST);
        SafeArray<LH_TYPE> h = SafeArray<LH_TYPE>(HOST);
};

template class AliasTableSplit<ArrayOfStructs>;
template class AliasTableSplit<StructOfArrays>;

std::ostream& operator<<(std::ostream &os, SplitConfig &split);

#endif //ALIAS_GPU_ALIASTABLESPLIT_CUH
