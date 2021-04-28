#ifndef ALIAS_GPU_ALIASTABLESPLITGPU_CUH
#define ALIAS_GPU_ALIASTABLESPLITGPU_CUH

#include <curand_kernel.h>
#include <algorithm>
#include <cuda_profiler_api.h>
#include <random>
#include <cub/iterator/transform_input_iterator.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/device/device_partition.cuh>

#include "utils/Utils.cuh"
#include "utils/Timer.cuh"
#include "utils/PrefixSum.cuh"
#include "construction/aliasTable/AliasTable.cuh"
#include "construction/aliasTable/buildMethod/AliasTableSplit.cuh"
#include "construction/aliasTable/buildMethod/SplitConfig.cuh"
#include "construction/aliasTable/packMethod/PackMethodBasic.cuh"
#include "construction/aliasTable/packMethod/PackMethodBasicShared.cuh"
#include "construction/aliasTable/packMethod/PackMethodNoWeights.cuh"
#include "construction/aliasTable/packMethod/PackMethodNoWeightsShared.cuh"
#include "construction/aliasTable/packMethod/PackMethodNoWeightsSharedTable.cuh"
#include "construction/aliasTable/packMethod/PackMethodPrecomputedWeight.cuh"
#include "construction/aliasTable/packMethod/PackMethodSweep.cuh"
#include "construction/aliasTable/packMethod/PackMethodChunkedShared.cuh"
#include "construction/aliasTable/splitMethod/SplitMethodBasic.cuh"
#include "construction/aliasTable/splitMethod/SplitMethodInverse.cuh"
#include "construction/aliasTable/splitMethod/SplitMethodInverseParallel.cuh"
#include "construction/aliasTable/splitMethod/SplitMethodParySearch.cuh"
#include "construction/aliasTable/psaPlus/PsaPlusNormal.cuh"

enum SplitVariant {
    splitVariantBasic,
    splitVariantInverse,
    splitVariantInverseParallel,
    splitVariantParySearch
};

enum PackVariant {
    packVariantBasic,
    packVariantWithoutWeights,
    packVariantSweep,
    packVariantPrecomputedWeight,
    packVariantChunkedShared
};

#define NUMBER_THREADS_AUTO -4
#define MINIMUM_ITEMS_PER_SPLIT 10

#ifndef SPLIT_THREADS_PER_BLOCK
#define SPLIT_THREADS_PER_BLOCK 512
#endif

#ifndef PACK_THREADS_PER_BLOCK
#define PACK_THREADS_PER_BLOCK 512
#endif

#define THREADS_PER_BLOCK 512

struct Variant {
    Variant(const SplitVariant split, const PackVariant pack, const bool sharedMemory)
            : pack(pack), sharedMemory(sharedMemory), split(split) {
        if (sharedMemory) {
            assert(pack == packVariantBasic || pack == packVariantWithoutWeights);
        }
        if (pack == packVariantSweep) {
            devicePartition = false;
        }
        setPsaPlusIfSupported(psaPlus);
        if (split == splitVariantInverse || split == splitVariantInverseParallel) {
            interleavedSplitPack = false;
        }
        if (pack == packVariantChunkedShared) {
            this->sharedMemory = false; // Method behaves as if it did not need shared memory (split size arbitrary)
        }
    }
    const SplitVariant split;
    const PackVariant pack;
    bool sharedMemory;
    bool shuffle = false;
    bool devicePartition = true;
    bool interleavedSplitPack = false;
    bool psaPlus = false;

    void setPsaPlusIfSupported(bool psaPlusNew) {
        #ifndef LH_TYPE_USE_WEIGHT
            psaPlus = false;
            return;
        #endif
        psaPlus = psaPlusNew
                  && pack != packVariantSweep // L, H must stay sorted
                  && pack != packVariantWithoutWeights; // copyWeightsToTable overwrites partially packed areas
    }
};

template <typename TableStorage>
class AliasTableSplitGpu : public AliasTableSplit<TableStorage> {
    public:
        explicit AliasTableSplitGpu(int size, Variant variant, int _numThreads = NUMBER_THREADS_AUTO,
                                    WeightDistribution weightDistribution = weightDistributionSine);
        ~AliasTableSplitGpu();
        void preBuild() override;
        void build() override;
        bool postBuild() override;
        std::string name() override;

        /**
         * Offset and limit allow to calculate the splits only partially.
         */
        void performSplit(int splitOffset = 0, int splitLimit = INT_MAX, cudaStream_t stream = 0);

        /**
         * Offset and limit allow to work off the splits only partially.
         */
        void performPack(int splitOffset = 0, int splitLimit = INT_MAX, cudaStream_t stream = 0);

        void performSplitAndPackInterleaved();
        void partitionLhCub();
        void partitionLhPrefixSum();
        void psaPlus();

        void copyWeightsToTable();
        void shuffleArraysLH();
        void freeMemory();

        int numH;
        int numL;
        TableSplitTimer timer;
    private:
        const Variant variant;
        SafeArray<double> weightsGpu = SafeArray<double>(DEVICE);
        TableStorage aliasTableGpu = TableStorage(DEVICE);
        SafeArray<double> prefixWeightL = SafeArray<double>(DEVICE);
        SafeArray<double> prefixWeightH = SafeArray<double>(DEVICE);
        SafeArray<LH_TYPE> l = SafeArray<LH_TYPE>(DEVICE);
        SafeArray<LH_TYPE> h = SafeArray<LH_TYPE>(DEVICE);
        SafeArray<SplitConfig> splits = SafeArray<SplitConfig>(DEVICE);
        SafeArray<int> prefixNumberOfHeavyItems = SafeArray<int>(DEVICE);
        LH_TYPE *preAllocLH;
        double *preAllocPrefixLH;
        SafeArray<double> precomputedWeightsL = SafeArray<double>(DEVICE); // only for packVariantPrecomputedWeight
        SafeArray<double> precomputedWeightsH = SafeArray<double>(DEVICE);

};

template class AliasTableSplitGpu<ArrayOfStructs>;
template class AliasTableSplitGpu<StructOfArrays>;

#endif //ALIAS_GPU_ALIASTABLESPLITGPU_CUH
