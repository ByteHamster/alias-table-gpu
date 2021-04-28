#ifndef ALIAS_GPU_ALIASTABLESTACK_CUH
#define ALIAS_GPU_ALIASTABLESTACK_CUH

#include <curand_kernel.h>
#include <algorithm>

#include "utils/Utils.cuh"
#include "utils/Timer.cuh"
#include "construction/aliasTable/AliasTable.cuh"

class AliasTableStack : public AliasTable<ArrayOfStructs> {
    public:
        explicit AliasTableStack(int size, WeightDistribution weightDistribution = weightDistributionSine);
        void build() override;
        std::string name() override;

        std::vector<int> h;
        std::vector<int> l;
};

#endif //ALIAS_GPU_ALIASTABLESTACK_CUH
