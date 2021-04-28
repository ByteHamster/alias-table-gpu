#ifndef ALIAS_GPU_UNITTESTS_CUH
#define ALIAS_GPU_UNITTESTS_CUH

#include "utils/PrefixSum.cuh"
#include "utils/Adder.cuh"
#include "construction/aliasTable/buildMethod/AliasTableStack.cuh"
#include "construction/aliasTable/buildMethod/AliasTableSweep.cuh"
#include "construction/aliasTable/buildMethod/AliasTableSplit.cuh"
#include "construction/aliasTable/buildMethod/AliasTableSplitGpu.cuh"
#include "construction/rejectionSampling/RejectionSamplingBinarySearch.cuh"
#include "construction/rejectionSampling/RejectionSamplingPArySearch.cuh"
#include "sampling/aliasTable/SamplerCpu.cuh"
#include "sampling/aliasTable/SamplerExpected.cuh"
#include "sampling/aliasTable/SamplerGpuBasic.cuh"
#include "sampling/aliasTable/SamplerGpuSectioned.cuh"
#include "sampling/aliasTable/SamplerGpuSectionedShared.cuh"
#include "sampling/rejectionSampling/SamplerRejection.cuh"
#include "sampling/rejectionSampling/SamplerRejectionGiveUp.cuh"

class UnitTests {
    public:
        static void execute();
    private:
        static void testTableCorrectness();
        static void testSamplingDistribution();
};

#endif //ALIAS_GPU_UNITTESTS_CUH
