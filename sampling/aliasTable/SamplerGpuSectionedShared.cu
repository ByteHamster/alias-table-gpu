#include "SamplerGpuSectionedShared.cuh"

SamplerGpuSectionedShared::SamplerGpuSectionedShared(AliasTable<ArrayOfStructs> &samplingAlgorithm, int itemsPerSection,
                                                     bool limitGroups)
    : itemsPerSection(itemsPerSection), limitGroups(limitGroups), SamplerGpu<AliasTable<ArrayOfStructs>>(samplingAlgorithm) {
    cudaDeviceProp deviceProperties = {};
    cudaGetDeviceProperties(&deviceProperties, 0);
    int maxItemsPerSection = deviceProperties.sharedMemPerBlock / sizeof(ArrayOfStructs::TableRow);
    if (this->itemsPerSection == SECTION_SIZE_AUTO) {
        this->itemsPerSection = 3000;
    }
    assert(this->itemsPerSection <= maxItemsPerSection);
}

std::string SamplerGpuSectionedShared::name() {
    return std::string("SamplerGpuSectionedShared") + (limitGroups ? "Limited" : "");
}

__device__ __forceinline__
int SamplerGpuSectionedShared::sample(ArrayOfStructs::TableRow *shared_section, double W_N, int length, XorWowPrng &random, int rowOffset) {
    double rand = random.next() * length;
    int tableRow = rand;
    ArrayOfStructs::TableRow row = {};
    ASSIGN_128(row, shared_section[tableRow])
    //row = shared_section[tableRow];
    double weight = row.weight;
    if (weight < W_N) {
        double percentage = (W_N) * (rand - tableRow);
        if (percentage < weight) {
            return tableRow + rowOffset;
        } else {
            return row.alias;
        }
    } else {
        return tableRow + rowOffset;
    }
}

__global__
void sampleBenchmarkSectionedShared(ArrayOfStructs table, int N, double W_N, int numSamples, int *distributionOutput) {
    extern __shared__ ArrayOfStructs::TableRow shared_section[];
    XorWowPrng random = XorWowPrng();
    random.initGpu(blockIdx.x * blockDim.x + threadIdx.x);

    __shared__ SamplerSection section;
    if (threadIdx.x == 0) {
        section = SamplerGpuSectioned<ArrayOfStructs>::calculateSection(N, numSamples);
    }
    __syncthreads();

    int size = section.end - section.start;
    assert(size < 4000); // Actual value is smaller, this is just a fail-safe

    if (size <= 0) {
        return;
    }

    // Copy table section to shared memory (interleaved)
    for (unsigned int j = threadIdx.x; j < size; j += blockDim.x) {
        ASSIGN_128(shared_section[j], table.rows[section.start + j])
        //shared_section[j] = table.rows[section.start + j];
    }
    __syncthreads();

    int samplesThisThread = section.numSamples / blockDim.x + 1;

    int dummy = 0;
    for (int i = 0; i < samplesThisThread; i++) {
        int sample = SamplerGpuSectionedShared::sample(shared_section, W_N, size, random, section.start);
        assert(sample >= 0);
        assert(sample < N);
        dummy += (sample % 2) + 1;

        #ifdef DEBUG_SUPPORT_SAMPLING_DISTRIBUTION
            if (distributionOutput != nullptr) {
                atomicAdd_system(&distributionOutput[sample], 1);
            }
        #endif
    }
    assert(samplesThisThread > 0);
    assert(dummy != 0);

    while (dummy == 0) {
        // Trick compiler into not optimizing away the sample() calls
        dummy = table.alias(0) = dummy % 2;
        assert(dummy >= 10 && "Problem with sampling");
    }
}

double SamplerGpuSectionedShared::executeBenchmarkSampling(int numSamples, int *distributionOutput) {
    ArrayOfStructs aliasTableGpu(samplingAlgorithm.aliasTable.size, DEVICE);
    aliasTableGpu.copyFrom(samplingAlgorithm.aliasTable);
    int numBlocks = aliasTableGpu.size / itemsPerSection + 2;

    int sharedMem = itemsPerSection * sizeof(ArrayOfStructs::TableRow);
    if (limitGroups) {
        cudaDeviceProp deviceProperties = {};
        cudaGetDeviceProperties(&deviceProperties, 0);
        sharedMem = std::max(sharedMem, (int) (0.8f * deviceProperties.sharedMemPerBlock)); // Ensure that only 1 group can run on each SM
    }

    Timer timer;
    timer.start();
    sampleBenchmarkSectionedShared<<<dim3(numBlocks), dim3(128), sharedMem>>>
        (aliasTableGpu, samplingAlgorithm.N, samplingAlgorithm.W / samplingAlgorithm.N, numSamples, distributionOutput);
    timer.stop();
    cudaDeviceSynchronize();
    LASTERR
    aliasTableGpu.free();
    return ((double) numSamples) / (timer.elapsedMillis() * 1000 * 1000);
}
