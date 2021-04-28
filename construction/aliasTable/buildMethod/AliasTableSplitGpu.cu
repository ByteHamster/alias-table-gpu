#include "AliasTableSplitGpu.cuh"

template <typename TableStorage>
std::string AliasTableSplitGpu<TableStorage>::name() {
    std::string variantString;
    switch (variant.split) {
        case splitVariantInverse:
            variantString = "InverseSplit";
            break;
        case splitVariantInverseParallel:
            variantString = "InverseParallelSplit";
            break;
        case splitVariantParySearch:
            variantString = "ParySearchSplit";
            break;
        case splitVariantBasic:
            variantString = "";
            break;
        default:
            assert(false && "Unknown split variant");
    }
    switch (variant.pack) {
        case packVariantBasic:
            variantString += "";
            break;
        case packVariantWithoutWeights:
            variantString += "NoWeightPack";
            break;
        case packVariantSweep:
            variantString += "SweepPack";
            break;
        case packVariantChunkedShared:
            variantString += "ChunkedSharedPack";
            break;
        case packVariantPrecomputedWeight:
            variantString += "PrecomputedWeight";
            break;
        default:
            assert(false && "Unknown pack variant");
    }
    if (variant.sharedMemory) {
        variantString += "SharedMemory";
    }
    if (variant.psaPlus) {
        variantString += "PsaPlus";
    }
    if (!variant.devicePartition) {
        variantString += "NoDp";
    }
    return "AliasTableSplitGpu" + TableStorage::name() + variantString;
}

template <typename TableStorage>
AliasTableSplitGpu<TableStorage>::AliasTableSplitGpu(int size, Variant variant, int _numThreads,
                                                     WeightDistribution weightDistribution)
        : AliasTableSplit<TableStorage>(size, _numThreads, weightDistribution), variant(variant) {

    if (_numThreads == NUMBER_THREADS_AUTO) {
        cudaDeviceProp deviceProperties = {};
        cudaGetDeviceProperties(&deviceProperties, 0);

        if (variant.pack == packVariantChunkedShared) {
            this->numThreads = SHARED_MEMORY_WORKER_THREADS * deviceProperties.multiProcessorCount * 10;
        } else if (variant.sharedMemory) {
            int maxItemsPerGroup = deviceProperties.sharedMemPerBlock / (sizeof(LH_TYPE) * SHARED_MEMORY_WORKER_THREADS) - 10;
            this->numThreads = size / maxItemsPerGroup + 1;
        } else {
            this->numThreads = 1400;
        }
    }
    this->numThreads = std::min(this->numThreads, size / MINIMUM_ITEMS_PER_SPLIT);
    if (this->numThreads < 2) {
        this->numThreads = 2;
    }
}

template <typename TableStorage>
AliasTableSplitGpu<TableStorage>::~AliasTableSplitGpu() = default;

__global__
void fillLightHeavyArrays(SafeArray<LH_TYPE> h, SafeArray<LH_TYPE> l, SafeArray<int> prefixNumberOfHeavyItems, int N, SafeArray<double> weights) {
    int num = blockIdx.x * blockDim.x + threadIdx.x;
    if (num >= prefixNumberOfHeavyItems.size) {
        return;
    }
    bool isHeavy;
    int prefixPosition = prefixNumberOfHeavyItems[num];
    int prefixPositionNext = prefixNumberOfHeavyItems[min((int) num + 1, (int) prefixNumberOfHeavyItems.size - 1)];
    if (num == N) {
        isHeavy = true; // Always infinity
    } else if (num == N + 1) {
        isHeavy = false; // Always 0
    } else {
        isHeavy = prefixPositionNext - prefixPosition == 1;
    }
    int index = isHeavy ? prefixPosition : num - prefixPosition;
    LH_TYPE *array = isHeavy ? h.data : l.data;
    array[index].item = num;
    array[index].setWeight(weights[num]);
}

#define COPY_WEIGHTS_TO_TABLE_ITEMS_PER_THREAD 31

template <typename TableStorage>
__global__
static void copyWeightsToTable(SafeArray<double> weights, TableStorage table) {
    for (unsigned int i = blockIdx.x * THREADS_PER_BLOCK * COPY_WEIGHTS_TO_TABLE_ITEMS_PER_THREAD + threadIdx.x;
         i < (blockIdx.x+1) * THREADS_PER_BLOCK * COPY_WEIGHTS_TO_TABLE_ITEMS_PER_THREAD + threadIdx.x && i < weights.size; i += THREADS_PER_BLOCK) {
        table.weight(i) = weights[i];
    }
}

struct WeightsLoaderOperator {
    double *weightsGpu;
    __host__ __device__ __forceinline__
    double operator()(const LH_TYPE &a) const {
        #ifdef LH_TYPE_USE_WEIGHT
            return a.weight;
        #else
            return weightsGpu[a.item];
        #endif
    }
};

struct LightVsHeavyClassifyOperator {
    double W_N;
    __host__ __device__ __forceinline__
    int operator()(const double &weight) const {
        return (weight > W_N) ? 1 : 0;
    }
};

struct CreateItemClassifierOperator {
    double *weights;
    __host__ __device__ __forceinline__
    LH_TYPE operator()(const int &index) const {
        #ifdef LH_TYPE_USE_WEIGHT
            return {index, weights[index]};
        #else
            return {index};
        #endif
    }
};

struct LightVsHeavyFilter {
    double W_N;
    double *weights;
    __host__ __device__ __forceinline__
    bool operator()(const LH_TYPE &index) const {
        return index.getWeight(weights) <= W_N;
    }
};

template <typename TableStorage>
void AliasTableSplitGpu<TableStorage>::copyWeightsToTable() {
    ::copyWeightsToTable<<<dim3((this->N + 2) / (COPY_WEIGHTS_TO_TABLE_ITEMS_PER_THREAD * THREADS_PER_BLOCK) + 1),
    dim3(THREADS_PER_BLOCK)>>>
            (weightsGpu, aliasTableGpu);
    //ERRCHECK(cudaMemcpy2D(aliasTableGpu.data, sizeof(AliasTable::TableRow), weightsGpu.data, sizeof(double), sizeof(double), N + 2, cudaMemcpyDeviceToDevice))
}

template<>
void AliasTableSplitGpu<StructOfArrays>::copyWeightsToTable() {
    cudaMemcpy(aliasTableGpu.weights.data, weightsGpu.data, weightsGpu.size * sizeof(double), cudaMemcpyDeviceToDevice);
}

template <typename TableStorage>
void AliasTableSplitGpu<TableStorage>::shuffleArraysLH() {
    int blockSize = 16 * 1024;
    {
        int *shuffleTemp;
        cudaMalloc(&shuffleTemp, l.size * sizeof(int));
        cudaMemcpyAsync(shuffleTemp, l.data, l.size * sizeof(int), cudaMemcpyDeviceToDevice);
        int blockNum = (l.size - 2) / blockSize;
        std::vector<int> blocksToCopy(blockNum);
        for (int i = 0; i < blockNum; i++) {
            blocksToCopy.at(i) = i;
        }
        auto rng = std::default_random_engine{};
        std::shuffle(std::begin(blocksToCopy), std::end(blocksToCopy), rng);
        for (int i = 0; i < blockNum; i++) {
            cudaMemcpyAsync(l.data + i * blockSize, shuffleTemp + blocksToCopy[i] * blockSize,
                            blockSize * sizeof(int), cudaMemcpyDeviceToDevice);
        }
        cudaFree(shuffleTemp);
    }
    {
        int *shuffleTemp;
        cudaMalloc(&shuffleTemp, h.size * sizeof(int));
        cudaMemcpyAsync(shuffleTemp, h.data, h.size * sizeof(int), cudaMemcpyDeviceToDevice);
        int blockNum = (h.size - 2) / blockSize;
        std::vector<int> blocksToCopy(blockNum);
        for (int i = 0; i < blockNum; i++) {
            blocksToCopy.at(i) = i;
        }
        auto rng = std::default_random_engine{};
        std::shuffle(std::begin(blocksToCopy), std::end(blocksToCopy), rng);
        for (int i = 0; i < blockNum; i++) {
            cudaMemcpyAsync(h.data + i * blockSize, shuffleTemp + blocksToCopy[i] * blockSize,
                            blockSize * sizeof(int), cudaMemcpyDeviceToDevice);
        }
        cudaFree(shuffleTemp);
    }
}

template <typename TableStorage>
void AliasTableSplitGpu<TableStorage>::preBuild() {
    this->weights[this->N] = std::numeric_limits<double>::infinity();
    this->weights[this->N + 1] = 0;
    weightsGpu.malloc(this->weights.size);
    aliasTableGpu.malloc(this->weights.size);
    splits.malloc(this->numThreads + 1);
    prefixNumberOfHeavyItems.malloc(this->N + 2);
    ERRCHECK(cudaMalloc(&preAllocLH, (this->N + 2) * sizeof(LH_TYPE)))
    ERRCHECK(cudaMalloc(&preAllocPrefixLH, (this->N + 2) * sizeof(double)))
    weightsGpu.copyFrom(this->weights);
}


template <typename TableStorage>
void AliasTableSplitGpu<TableStorage>::performSplitAndPackInterleaved() {
    int numSections = 10;
    int itemPadding = 2048;
    cudaStream_t streamSplit;
    cudaStream_t streamPack;
    cudaEvent_t events[numSections];

    cudaStreamCreate(&streamSplit);
    cudaStreamCreate(&streamPack);
    for (int i = 0; i < numSections; i++) {
        cudaEventCreate(&events[i]);
    }

    int itemsPerSection = this->numThreads / numSections;
    int done = 0;
    for (int i = 0; i < numSections - 1; i++) {
        performSplit(done, itemsPerSection, streamSplit);
        cudaEventRecord(events[i], streamSplit);
        done += itemsPerSection;
    }
    performSplit(done, INT_MAX, streamSplit);
    cudaEventRecord(events[numSections - 1], streamSplit);

    cudaStreamWaitEvent(streamPack, events[0], 0);
    performPack(0, itemsPerSection - itemPadding, streamPack);
    done = itemsPerSection - itemPadding;
    for (int i = 1; i < numSections - 1; i++) {
        cudaStreamWaitEvent(streamPack, events[i], 0);
        performPack(done, itemsPerSection, streamPack);
        done += itemsPerSection;
    }
    cudaStreamWaitEvent(streamPack, events[numSections - 1], 0);
    performPack(done, INT_MAX, streamPack);

    ERRCHECK(cudaDeviceSynchronize());

    cudaStreamDestroy(streamSplit);
    cudaStreamDestroy(streamPack);
    for (int i = 0; i < numSections; i++) {
        cudaEventDestroy(events[i]);
    }
}

__global__
static void copyWeightsToLHKernel(SafeArray<double> weights, LhTypeWithWeight *items, int N) {
    for (int i = blockIdx.x * SPLIT_THREADS_PER_BLOCK * COPY_WEIGHTS_TO_TABLE_ITEMS_PER_THREAD + threadIdx.x;
         i < (blockIdx.x+1) * SPLIT_THREADS_PER_BLOCK * COPY_WEIGHTS_TO_TABLE_ITEMS_PER_THREAD + threadIdx.x && i < N; i += SPLIT_THREADS_PER_BLOCK) {
        items[i] = {i, weights[i]};
    }
}

template <typename TableStorage>
void AliasTableSplitGpu<TableStorage>::partitionLhCub() {
    int *d_num_selected_out;
    cudaMalloc(&d_num_selected_out, sizeof(int));
    LightVsHeavyFilter select_op = {};
    select_op.W_N = this->W / this->N;
    select_op.weights = this->weightsGpu.data;
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::CountingInputIterator<int> counter(0);
    CreateItemClassifierOperator createItemOperator = {};
    createItemOperator.weights = weightsGpu.data;
    cub::TransformInputIterator<LH_TYPE, CreateItemClassifierOperator, cub::CountingInputIterator<int>>
            createItemClassifier(counter, createItemOperator);

    cub::DevicePartition::If(d_temp_storage, temp_storage_bytes, createItemClassifier, preAllocLH, d_num_selected_out,
                             this->N + 2, select_op);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DevicePartition::If(d_temp_storage, temp_storage_bytes, createItemClassifier, preAllocLH, d_num_selected_out,
                             this->N + 2, select_op);
    cudaMemcpy(&numL, d_num_selected_out, sizeof(int), cudaMemcpyDeviceToHost);
    numH = (this->N + 2) - numL;

    // DevicePartition reverses the non-selected list.
    // This is not a problem per se (only for the sweep method) but
    // the sentinel element of the heavy array (weights[N] = inf) is now the first element.
    // Copy the last element there and fill the last with the sentinel N.
    cudaMemcpy(&preAllocLH[numL], &preAllocLH[this->N + 1], sizeof(LH_TYPE), cudaMemcpyDeviceToDevice);
    LH_TYPE heavySentinel = {};
    heavySentinel.item = this->N;
    heavySentinel.setWeight(std::numeric_limits<double>::infinity());
    cudaMemcpy(&preAllocLH[this->N + 1], &heavySentinel, sizeof(LH_TYPE), cudaMemcpyHostToDevice);

    l.usePreInitialized(preAllocLH, numL);
    h.usePreInitialized(preAllocLH + numL, numH);
    prefixWeightL.usePreInitialized(preAllocPrefixLH, numL);
    prefixWeightH.usePreInitialized(preAllocPrefixLH + numL, numH);
    cudaFree(d_num_selected_out);
    cudaFree(d_temp_storage);
}

template <typename TableStorage>
void AliasTableSplitGpu<TableStorage>::partitionLhPrefixSum() {
    #ifdef LH_TYPE_USE_WEIGHT
    copyWeightsToLHKernel<<<dim3((this->N + 2) / (COPY_WEIGHTS_TO_TABLE_ITEMS_PER_THREAD * THREADS_PER_BLOCK) + 1), dim3(THREADS_PER_BLOCK)>>>
            (weightsGpu, preAllocLH, this->N);
    #endif

    void *tempStorageLightHeavy = nullptr;
    size_t tempStorageSizeLightHeavy = 0;
    LightVsHeavyClassifyOperator lightVsHeavyOperator = {};
    lightVsHeavyOperator.W_N = this->W / this->N;
    cub::TransformInputIterator<int, LightVsHeavyClassifyOperator, double *> lightVsHeavyIterator(weightsGpu.data, lightVsHeavyOperator);
    ERRCHECK(cub::DeviceScan::ExclusiveSum(tempStorageLightHeavy, tempStorageSizeLightHeavy, lightVsHeavyIterator, prefixNumberOfHeavyItems.data, weightsGpu.size))
    cudaMalloc(&tempStorageLightHeavy, tempStorageSizeLightHeavy);
    ERRCHECK(cub::DeviceScan::ExclusiveSum(tempStorageLightHeavy, tempStorageSizeLightHeavy, lightVsHeavyIterator, prefixNumberOfHeavyItems.data, weightsGpu.size))
    cudaFree(tempStorageLightHeavy);
    cudaMemcpy(&numH, &prefixNumberOfHeavyItems[this->N + 1], sizeof(int), cudaMemcpyDeviceToHost);
    numL = (this->N + 2) - numH;

    l.usePreInitialized(preAllocLH, numL);
    h.usePreInitialized(preAllocLH + numL, numH);
    prefixWeightL.usePreInitialized(preAllocPrefixLH, numL);
    prefixWeightH.usePreInitialized(preAllocPrefixLH + numL, numH);

    fillLightHeavyArrays<<<dim3((this->N + 2) / THREADS_PER_BLOCK + 1), dim3(THREADS_PER_BLOCK)>>>
            (h, l, prefixNumberOfHeavyItems, this->N, weightsGpu);
}

template <typename TableStorage>
void AliasTableSplitGpu<TableStorage>::psaPlus() {
    #ifndef LH_TYPE_USE_WEIGHT
        std::cerr<<"PSA+ only supported for LH type with weights"<<std::endl;
        raise(SIGABRT);
    #endif

    int *numLGpu;
    int *numHGpu;
    cudaMalloc(&numLGpu, sizeof(int));
    cudaMalloc(&numHGpu, sizeof(int));
    cudaMemset(numLGpu, 0, sizeof(int));
    cudaMemset(numHGpu, 0, sizeof(int));

    int sectionSize = std::min(1000, this->N / MINIMUM_ITEMS_PER_SPLIT);
    PsaPlusNormal::kernel<TableStorage><<<dim3(this->N / sectionSize + 1),
        dim3(PSA_PLUS_THREADS_PER_BLOCK), sectionSize * sizeof(LH_TYPE)>>>
        (this->N, this->W/this->N, weightsGpu, preAllocLH, numHGpu, numLGpu, aliasTableGpu);

    cudaMemcpy(&numL, numLGpu, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&numH, numHGpu, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(numLGpu);
    cudaFree(numHGpu);

    LH_TYPE heavySentinel = {};
    heavySentinel.item = this->N;
    heavySentinel.setWeight(std::numeric_limits<double>::infinity());
    cudaMemcpy(&preAllocLH[this->N + 1], &heavySentinel, sizeof(LH_TYPE), cudaMemcpyHostToDevice);

    LH_TYPE lightSentinel = {};
    lightSentinel.item = this->N + 1;
    lightSentinel.setWeight(0);
    cudaMemcpy(&preAllocLH[numL], &lightSentinel, sizeof(LH_TYPE), cudaMemcpyHostToDevice);

    numL++; // Sentinels
    numH++;
    l.usePreInitialized(preAllocLH, numL);
    h.usePreInitialized(preAllocLH + (this->N + 2) - numH, numH);
    prefixWeightL.usePreInitialized(preAllocPrefixLH, numL);
    prefixWeightH.usePreInitialized(preAllocPrefixLH + (this->N + 2) - numH, numH);
    this->numThreads = std::min(this->numThreads, (numL + numH) / MINIMUM_ITEMS_PER_SPLIT);
    if (this->numThreads < 2) {
        this->numThreads = 2;
    }
}

template <typename TableStorage>
void AliasTableSplitGpu<TableStorage>::build() {
    timer.notify(EVENT_START);
    this->W = Adder::addGpuCub(this->N, weightsGpu);
    timer.notify(EVENT_SUM_FINISHED);

    if (variant.psaPlus) {
        psaPlus();
    } else if (variant.devicePartition) {
        partitionLhCub();
    } else {
        partitionLhPrefixSum();
    }
    timer.notify(EVENT_PARTITION_FINISHED);

    assert(l.size == numL && numL > 1); // >1 because of the sentinels
    assert(h.size == numH && numH > 1); // >1 because of the sentinels

    void *tempStorageL = nullptr;
    void *tempStorageH = nullptr;
    size_t tempStorageSizeL = 0;
    size_t tempStorageSizeH = 0;
    WeightsLoaderOperator weightsLoaderOperator = {};
    weightsLoaderOperator.weightsGpu = weightsGpu.data;
    cub::TransformInputIterator<double, WeightsLoaderOperator, LH_TYPE*> loadWeightIteratorL(l.data, weightsLoaderOperator);
    cub::TransformInputIterator<double, WeightsLoaderOperator, LH_TYPE*> loadWeightIteratorH(h.data, weightsLoaderOperator);
    // Prepare only
    ERRCHECK(cub::DeviceScan::ExclusiveSum(tempStorageL, tempStorageSizeL, loadWeightIteratorL, prefixWeightL.data, prefixWeightL.size))
    ERRCHECK(cub::DeviceScan::ExclusiveSum(tempStorageH, tempStorageSizeH, loadWeightIteratorH, prefixWeightH.data, prefixWeightH.size))
    cudaMalloc(&tempStorageL, tempStorageSizeL);
    cudaMalloc(&tempStorageH, tempStorageSizeH);

    if (variant.pack == packVariantWithoutWeights) {
        copyWeightsToTable();
    }

    if (variant.shuffle) {
        shuffleArraysLH();
        cudaDeviceSynchronize();
    }

    ERRCHECK(cub::DeviceScan::ExclusiveSum(tempStorageL, tempStorageSizeL, loadWeightIteratorL, prefixWeightL.data, prefixWeightL.size))
    ERRCHECK(cub::DeviceScan::ExclusiveSum(tempStorageH, tempStorageSizeH, loadWeightIteratorH, prefixWeightH.data, prefixWeightH.size))

    timer.notify(EVENT_PREFIXSUM_FINISHED);

    if (variant.pack == packVariantPrecomputedWeight) {
        #ifdef LH_TYPE_USE_WEIGHT
            std::cerr<<"packVariantPrecomputedWeight only supported for LH_TYPE without weight"<<std::endl;
            raise(SIGABRT);
        #else
            precomputedWeightsL.malloc(numL);
            PackMethodPrecomputedWeight::fillWeightKernel
                <<<dim3(numL / THREADS_PER_BLOCK + 1), dim3(THREADS_PER_BLOCK)>>>(precomputedWeightsL, l, weightsGpu);
            precomputedWeightsH.malloc(numH);
            PackMethodPrecomputedWeight::fillWeightKernel
                <<<dim3(numH / THREADS_PER_BLOCK + 1), dim3(THREADS_PER_BLOCK)>>>(precomputedWeightsH, h, weightsGpu);
        #endif
    }

    cudaProfilerStart();

    if (variant.interleavedSplitPack) {
        performSplitAndPackInterleaved();
    } else {
        performSplit();
        timer.notify(EVENT_SPLIT_FINISHED);
        performPack();
        timer.notify(EVENT_PACK_FINISHED);
    }

    cudaFree(tempStorageL);
    cudaFree(tempStorageH);

    if (variant.pack == packVariantPrecomputedWeight) {
        precomputedWeightsH.free();
        precomputedWeightsL.free();
    }

    LASTERR
}

template <typename TableStorage>
void AliasTableSplitGpu<TableStorage>::performSplit(int splitOffset, int splitLimit, cudaStream_t stream) {
    splitLimit = std::min(splitLimit, this->numThreads - splitOffset);

    int nToSplit = numH + numL - 2;

    double totalWeightL;
    double totalWeightH;
    cudaMemcpy(&totalWeightL, &prefixWeightL[numL - 1], sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&totalWeightH, &prefixWeightH[numH - 1], sizeof(double), cudaMemcpyDeviceToHost);
    double wToSplit = totalWeightL + totalWeightH;

    if (variant.split == splitVariantInverse) {
        assert(splitOffset == 0 && splitLimit == this->numThreads);
        SplitMethodInverse::splitKernel<<<dim3(numH / SPLIT_THREADS_PER_BLOCK + 1), dim3(SPLIT_THREADS_PER_BLOCK), 0, stream>>>
            (splits, nToSplit, wToSplit, this->numThreads, weightsGpu, h, prefixWeightH, numH, prefixWeightL, numL);
    } else if (variant.split == splitVariantInverseParallel) {
        assert(splitOffset == 0 && splitLimit == this->numThreads);
        int threadsPerBlock = 16; // SPLIT_THREADS_PER_BLOCK is too big
        SplitMethodInverseParallel::splitKernel<<<dim3(numH / threadsPerBlock + 1, this->numThreads / threadsPerBlock + 1),
                dim3(threadsPerBlock, threadsPerBlock), 0, stream>>>
            (splits, nToSplit, wToSplit, this->numThreads, weightsGpu, h, prefixWeightH, numH, prefixWeightL, numL);
    } else if (variant.split == splitVariantBasic) {
        int numGroups = (splitLimit - 1) / 1024 + 1;
        SplitMethodBasic::splitKernel<<<dim3(numGroups), dim3(1024), 0, stream>>>
            (splitOffset, splits, nToSplit, wToSplit, this->numThreads, weightsGpu, h, prefixWeightH, numH, prefixWeightL, numL);
    } else if (variant.split == splitVariantParySearch) {
        int numGroups = (splitLimit - 1) / PARY_SEARCH_GROUP_SIZE + 1;
        SplitMethodParySearch::splitKernel<<<dim3(numGroups), dim3(PARY_SEARCH_GROUP_SIZE), 0, stream>>>
            (splitOffset, splits, nToSplit, wToSplit, this->numThreads, weightsGpu, h, prefixWeightH, numH, prefixWeightL, numL);
    } else {
        assert(false && "Unknown split variant");
    }
    LASTERR
}

template <typename TableStorage>
void AliasTableSplitGpu<TableStorage>::performPack(int splitOffset, int splitLimit, cudaStream_t stream) {
    splitLimit = std::min(splitLimit, this->numThreads - splitOffset);
    int nToSplit = numH + numL - 2;

    #define DEBUG_PACK_CPU false
    if (DEBUG_PACK_CPU) {
        SafeArray<SplitConfig> splitsCpu(HOST);
        splitsCpu.malloc(splits.size);
        splitsCpu.copyFrom(splits);

        SafeArray<LH_TYPE> hCpu(HOST);
        hCpu.malloc(h.size);
        hCpu.copyFrom(h);

        SafeArray<LH_TYPE> lCpu(HOST);
        lCpu.malloc(l.size);
        lCpu.copyFrom(l);

        for (int k = 1; k <= this->numThreads; k++) {
            SplitConfig splitCurrent = splitsCpu[k];
            SplitConfig splitPrevious = splitsCpu[k - 1];
            PackMethodBasic::PackAll packMethodLimit;
            PackMethodBasic::pack<TableStorage, PackMethodBasic::PackAll>(
                    k, splitCurrent, splitPrevious, this->W / this->N, hCpu, lCpu, this->aliasTable, this->weights, packMethodLimit);
        }
        aliasTableGpu.copyFrom(this->aliasTable);
    } else if (variant.pack == packVariantWithoutWeights) {
        if (variant.sharedMemory) {
            int numGroups = splitLimit / SHARED_MEMORY_WORKER_THREADS + 1;
            int sharedMemorySize = SHARED_MEMORY_WORKER_THREADS * (nToSplit / this->numThreads + 10) * sizeof(LH_TYPE);

            cudaDeviceProp deviceProperties = {};
            cudaGetDeviceProperties(&deviceProperties, 0);
            assert(sharedMemorySize <= deviceProperties.sharedMemPerBlock);

            PackMethodNoWeightsShared::packKernel<<<dim3(numGroups), dim3(512), sharedMemorySize, stream>>>
                (splitOffset, splits, this->W/this->N, h, l, aliasTableGpu, this->numThreads);

            // PackMethodNoWeightsSharedTable::execute(aliasTableGpu, this->W, this->N, p, splits, l, h);
        } else {
            int numGroups = splitLimit / PACK_THREADS_PER_BLOCK + 1;
            PackMethodNoWeights::packKernel<<<dim3(numGroups), dim3(PACK_THREADS_PER_BLOCK), 0, stream>>>
                (splitOffset, splits, this->W/this->N, h, l, aliasTableGpu, this->numThreads);
        }
    } else if (variant.pack == packVariantBasic) {
        if (variant.sharedMemory) {
            int numGroups = splitLimit / SHARED_MEMORY_WORKER_THREADS + 1;
            int sharedMemorySize = SHARED_MEMORY_WORKER_THREADS * (nToSplit / this->numThreads + 10) * sizeof(LH_TYPE);

            cudaDeviceProp deviceProperties = {};
            cudaGetDeviceProperties(&deviceProperties, 0);
            assert(sharedMemorySize <= deviceProperties.sharedMemPerBlock);

            PackMethodBasicShared::packKernel<<<dim3(numGroups), dim3(512), sharedMemorySize, stream>>>
                (splitOffset, splits, this->W/this->N, h, l, aliasTableGpu, weightsGpu.data, this->numThreads);
        } else {
            int numGroups = splitLimit / PACK_THREADS_PER_BLOCK + 1;
            PackMethodBasic::packKernel<<<dim3(numGroups), dim3(PACK_THREADS_PER_BLOCK), 0, stream>>>
                (splitOffset, splits, this->W/this->N, h, l, aliasTableGpu, this->numThreads, weightsGpu.data);
        }
    } else if (variant.pack == packVariantSweep) {
        assert(!variant.sharedMemory);
        int numGroups = splitLimit / PACK_THREADS_PER_BLOCK + 1;
        PackMethodSweep::packKernel<<<dim3(numGroups), dim3(PACK_THREADS_PER_BLOCK), 0, stream>>>
            (splitOffset, splits, this->W/this->N, h, l, aliasTableGpu, this->numThreads, weightsGpu);
    } else if (variant.pack == packVariantChunkedShared) {
        assert(!variant.sharedMemory); // Method behaves as if it did not need shared memory (split size arbitrary)
        int numGroups = splitLimit / CHUNKED_WORKER_THREADS + 1;
        PackMethodChunkedShared::packKernel<<<dim3(numGroups), dim3(CHUNKED_GROUP_SIZE), 0, stream>>>
            (splitOffset, splits, this->W/this->N, h, l, aliasTableGpu, weightsGpu, this->numThreads);
    } else if (variant.pack == packVariantPrecomputedWeight) {
        #ifdef LH_TYPE_USE_WEIGHT
            std::cerr<<"packVariantPrecomputedWeight only supported for LH_TYPE without weight"<<std::endl;
            raise(SIGABRT);
        #else
            int numGroups = splitLimit / PACK_THREADS_PER_BLOCK + 1;
            PackMethodPrecomputedWeight::packKernel<<<dim3(numGroups), dim3(PACK_THREADS_PER_BLOCK), 0, stream>>>
                (splitOffset, splits, this->W/this->N, h, l, aliasTableGpu, this->numThreads, weightsGpu.data,
                    precomputedWeightsL, precomputedWeightsH);
        #endif
    } else {
        assert(false && "Unknown pack variant");
    }
    LASTERR
}

template <typename TableStorage>
bool AliasTableSplitGpu<TableStorage>::postBuild() {
    assert(false && "Not implemented");
    return false;
}

template<>
bool AliasTableSplitGpu<ArrayOfStructs>::postBuild() {
    aliasTable.rows.copyFrom(aliasTableGpu.rows);
    freeMemory();
    return this->verifyTableCorrectness();
}

template<>
bool AliasTableSplitGpu<StructOfArrays>::postBuild() {
    aliasTable.aliases.copyFrom(aliasTableGpu.aliases);
    aliasTable.weights.copyFrom(aliasTableGpu.weights);
    freeMemory();
    return this->verifyTableCorrectness();
}

template <typename TableStorage>
void AliasTableSplitGpu<TableStorage>::freeMemory() {
    prefixNumberOfHeavyItems.free();
    weightsGpu.free();
    aliasTableGpu.free();
    splits.free();
    ERRCHECK(cudaFree(preAllocLH))
    ERRCHECK(cudaFree(preAllocPrefixLH))
}
