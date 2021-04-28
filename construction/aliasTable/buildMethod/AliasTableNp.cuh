#ifndef ALIAS_GPU_ALIASTABLE_NP_CUH
#define ALIAS_GPU_ALIASTABLE_NP_CUH

#include <curand_kernel.h>
#include <algorithm>
#include <sstream>
#include "utils/Utils.cuh"
#include "utils/Timer.cuh"
#include "construction/SamplingAlgorithm.cuh"
#include "AliasTableSplit.cuh"

template <typename TableStorage>
class AliasTableNp : public AliasTable<TableStorage> {
    public:
        AliasTableNp(int size, bool bestCase);
        void build() override;
        std::string name() override;
    private:
        void buildPerfect();
        void buildWorstCase();
        bool useBestCase = false;
};

template class AliasTableNp<ArrayOfStructs>;
template class AliasTableNp<StructOfArrays>;

#endif //ALIAS_GPU_ALIASTABLE_NP_CUH
