#include "SamplerGpuSectioned.cuh"

template <typename TableStorage>
SamplerGpuSectioned<TableStorage>::SamplerGpuSectioned(AliasTable<TableStorage> &samplingAlgorithm, int itemsPerSection,
                                                       bool limitGroups)
        : itemsPerSection(itemsPerSection), limitGroups(limitGroups), SamplerGpu<AliasTable<TableStorage>>(samplingAlgorithm) {
    if (this->itemsPerSection == SECTION_SIZE_AUTO) {
        this->itemsPerSection = (samplingAlgorithm.N < 5e6) ? 4000 : 8000;
    }
}

template <typename TableStorage>
std::string SamplerGpuSectioned<TableStorage>::name() {
    return std::string("SamplerGpuSectioned") + (limitGroups ? "Limited" : "");
}

// Basically a useless copy but if the method is included from the other file,
// it is not inlined, which causes a 50% performance impact.
template <typename TableStorage>
__device__ __forceinline__
int SamplerGpuSectioned<TableStorage>::sample(TableStorage table, double W_N, int offset, int length, XorWowPrng &random) {
    double rand = offset + random.next() * length;
    int tableRow = rand;
    double weight = table.weight(tableRow);
    if (weight < W_N) {
        double percentage = (W_N) * (rand - tableRow);
        if (percentage < weight) {
            return tableRow;
        } else {
            return table.alias(tableRow);
        }
    } else {
        return tableRow;
    }
}

template <>
__device__ __forceinline__
int SamplerGpuSectioned<ArrayOfStructs>::sample(ArrayOfStructs table, double W_N, int offset, int length, XorWowPrng &random) {
    double rand = offset + random.next() * length;
    int tableRow = rand;
    ArrayOfStructs::TableRow row = {};
    ASSIGN_128(row, table.rows.data[tableRow])
    //row = table.rows.data[tableRow];
    double weight = row.weight;
    if (weight < W_N) {
        double percentage = (W_N) * (rand - tableRow);
        if (percentage < weight) {
            return tableRow;
        } else {
            return row.alias;
        }
    } else {
        return tableRow;
    }
}

template <typename TableStorage>
__device__
SamplerSection SamplerGpuSectioned<TableStorage>::calculateSection(int N, int numSamples) {
    XorWowPrng randomMakeSection = XorWowPrng();
    randomMakeSection.initGpu(0, 0); // All threads in all blocks need the same numbers
    int numSplits = 32 - __clz(gridDim.x); // ceil(log2(x))
    int iteration = 0;
    int samplesThisBlock = numSamples;
    int tableSectionStart = 0;
    int tableSectionEnd = N;
    int blockStart = 0;
    int blockEnd = gridDim.x;
    while (iteration < numSplits) {
        int blockSize =  blockEnd - blockStart;
        int halfBlock = blockSize / 2;
        int blockCenter = blockStart + halfBlock;
        float blockPercentage = (float) halfBlock / (float) blockSize;

        int tableSectionSize = tableSectionEnd - tableSectionStart;
        int tableSectionCenter = tableSectionStart + blockPercentage * tableSectionSize;
        float tableSectionPercentage = (blockPercentage * tableSectionSize) / tableSectionSize;

        int x = Binomial::draw(samplesThisBlock, tableSectionPercentage, randomMakeSection.state);
        if (blockIdx.x < blockCenter) {
            // Continue left
            tableSectionEnd = tableSectionCenter;
            blockEnd = blockCenter;
            samplesThisBlock = x;
        } else {
            // Continue right
            tableSectionStart = tableSectionCenter;
            blockStart = blockCenter;
            randomMakeSection.skipAhead(1 << (numSplits - iteration));
            samplesThisBlock = samplesThisBlock - x;
        }
        iteration++;
    }
    return {tableSectionStart, tableSectionEnd, samplesThisBlock};
}

template <typename TableStorage>
__global__
void sampleBenchmarkSectioned(TableStorage table, int N, double W_N, int numSamples, int *distributionOutput) {
    XorWowPrng random = XorWowPrng();
    random.initGpu(blockIdx.x * blockDim.x + threadIdx.x);

    __shared__ SamplerSection section;
    if (threadIdx.x == 0) {
        section = SamplerGpuSectioned<TableStorage>::calculateSection(N, numSamples);
    }
    __syncthreads();

    int size = section.end - section.start;
    if (size <= 0) {
        return;
    }

    int samplesThisThread = section.numSamples / blockDim.x + 1;

    int dummy = 0;
    for (int i = 0; i < samplesThisThread; i++) {
        int sample = SamplerGpuSectioned<TableStorage>::sample(table, W_N, section.start, size, random);
        assert(sample >= 0);
        assert(sample < N);
        dummy += (sample % 2) + 1;

        #ifdef DEBUG_SUPPORT_SAMPLING_DISTRIBUTION
            if (distributionOutput != nullptr) {
                atomicAdd_system(&distributionOutput[sample], 1);
            }
        #endif
    }

    while (dummy == 0) {
        // Trick compiler into not optimizing away the sample() calls
        dummy = table.alias(0) = dummy % 2;
        assert(dummy >= 10 && "Problem with sampling");
    }
}

template <typename TableStorage>
double SamplerGpuSectioned<TableStorage>::executeBenchmarkSampling(int numSamples, int *distributionOutput) {
    TableStorage aliasTableGpu(this->samplingAlgorithm.aliasTable.size, DEVICE);
    aliasTableGpu.copyFrom(this->samplingAlgorithm.aliasTable);
    int numBlocks = aliasTableGpu.size / itemsPerSection + 1;

    int memPerBlock = 0;
    if (limitGroups) {
        #ifdef SET_CACHE_CONFIG
        cudaFuncSetCacheConfig(sampleBenchmarkSectioned<TableStorage>, cudaFuncCachePreferL1);
        #endif
        cudaDeviceProp deviceProperties = {};
        cudaGetDeviceProperties(&deviceProperties, 0);
        memPerBlock = 0.6f * deviceProperties.sharedMemPerBlock; // Ensure that only 1 group can run on each SM
    }

    Timer timer;
    timer.start();
    sampleBenchmarkSectioned<<<dim3(numBlocks), dim3(128), memPerBlock>>>(aliasTableGpu, this->samplingAlgorithm.N,
                     this->samplingAlgorithm.W / this->samplingAlgorithm.N, numSamples, distributionOutput);
    timer.stop();
    cudaDeviceSynchronize();
    #ifdef SET_CACHE_CONFIG
    cudaFuncSetCacheConfig(sampleBenchmarkSectioned<TableStorage>, cudaFuncCachePreferNone);
    #endif
    LASTERR
    aliasTableGpu.free();
    return ((double) numSamples) / (timer.elapsedMillis() * 1000 * 1000);
}
