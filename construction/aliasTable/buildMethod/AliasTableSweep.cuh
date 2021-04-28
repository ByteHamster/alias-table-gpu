#ifndef ALIAS_GPU_ALIASTABLESWEEP_CUH
#define ALIAS_GPU_ALIASTABLESWEEP_CUH

#include <curand_kernel.h>
#include <algorithm>

#include "utils/Utils.cuh"
#include "utils/Timer.cuh"
#include "construction/aliasTable/AliasTable.cuh"

class AliasTableSweep : public AliasTable<ArrayOfStructs> {
    public:
        explicit AliasTableSweep(int size, WeightDistribution weightDistribution = weightDistributionSine);
        void build() override;
        std::string name() override;
};

#endif //ALIAS_GPU_ALIASTABLESWEEP_CUH
