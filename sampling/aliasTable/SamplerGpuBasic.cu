#include "SamplerGpuBasic.cuh"

template <typename TableStorage, typename Prng>
SamplerGpuBasic<TableStorage, Prng>::SamplerGpuBasic(AliasTable<TableStorage> &samplingAlgorithm,
                                                     int numBlocks, int numThreads)
        : SamplerGpu<AliasTable<TableStorage>>(samplingAlgorithm), numBlocks(numBlocks), numThreads(numThreads) {
    if (this->numThreads == DIMENS_DEFAULT) {
        if (std::is_same<Prng, MtPrng>::value) {
            this->numThreads = 224;
        } else {
            this->numThreads = 160;
        }
    }
    if (this->numBlocks == DIMENS_DEFAULT) {
        if (std::is_same<Prng, MtPrng>::value) {
            this->numBlocks = 46;
        } else {
            this->numBlocks = 46;
        }
    }
}

template <typename TableStorage, typename Prng>
std::string SamplerGpuBasic<TableStorage, Prng>::name() {
    return "SamplerGpuBasic" + Prng::name();
}

template <typename TableStorage>
__device__ __forceinline__
int doSample(TableStorage table, double W_N, int N, double randomNumber) {
    // Must do the same PRNG calls in all threads to prevent deadlocks
    // because MTGP32 uses syncthreads() internally.
    double rand = randomNumber * N;
    int tableRow = rand;
    double percentage = W_N * (rand - tableRow);
    double weight = table.weight(tableRow);
    if (weight < W_N) {
        if (percentage < weight) {
            return tableRow;
        } else {
            return table.alias(tableRow);
        }
    } else {
        return tableRow;
    }
}

template<>
__device__ __forceinline__
int doSample<ArrayOfStructs>(ArrayOfStructs table, double W_N, int N, double randomNumber) {
    // Must do the same PRNG calls in all threads to prevent deadlocks
    // because MTGP32 uses syncthreads() internally.
    double rand = randomNumber * N;
    int tableRow = rand;
    double percentage = W_N * (rand - tableRow);
    ArrayOfStructs::TableRow row = {};
    ASSIGN_128(row, table.rows.data[tableRow])
    //row = table.rows.data[tableRow];
    double weight = row.weight;
    if (weight < W_N) {
        if (percentage < weight) {
            return tableRow;
        } else {
            return row.alias;
        }
    } else {
        return tableRow;
    }
}

template <typename TableStorage, typename Prng>
__device__ __forceinline__
int SamplerGpuBasic<TableStorage, Prng>::sample(TableStorage table, double W_N, int N, Prng &random) {
    return doSample<TableStorage>(table, W_N, N, random.next());
}

template <typename TableStorage, typename Prng>
__global__
void sampleBenchmarkBasic(TableStorage table, int N, double W_N, int numSamples, Prng random, int *distributionOutput) {
    random.initGpu(blockIdx.x * blockDim.x + threadIdx.x);

    int samplesThisThread = numSamples / (gridDim.x * blockDim.x);
    if (blockIdx.x == 0) {
        int samplesLeftOutBecauseOfRounding = numSamples - samplesThisThread * gridDim.x * blockDim.x;
        samplesThisThread += samplesLeftOutBecauseOfRounding / blockDim.x;
    }

    int dummy = 0;
    for (int i = 0; i < samplesThisThread; i++) {
        int sample = SamplerGpuBasic<TableStorage, Prng>::sample(table, W_N, N, random);
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

template <typename TableStorage, typename Prng>
double SamplerGpuBasic<TableStorage, Prng>::executeBenchmarkSampling(int numSamples, int *distributionOutput) {
    TableStorage aliasTableGpu(this->samplingAlgorithm.aliasTable.size, DEVICE);
    aliasTableGpu.copyFrom(this->samplingAlgorithm.aliasTable);
    Prng random = Prng();

    int numBlocks = this->numBlocks;
    int numThreads = this->numThreads;
    if (numBlocks * numThreads > numSamples) {
        numThreads = 32;
        numBlocks = numSamples / (numThreads * 10) + 1;
    }

    random.initCpu(numBlocks);

    Timer timer;
    timer.start();
    sampleBenchmarkBasic<<<dim3(numBlocks), dim3(numThreads)>>>(aliasTableGpu, this->samplingAlgorithm.N,
                         this->samplingAlgorithm.W / this->samplingAlgorithm.N, numSamples, random, distributionOutput);
    timer.stop();
    LASTERR
    aliasTableGpu.free();
    random.free();
    return ((double) numSamples) / (timer.elapsedMillis() * 1000 * 1000);
}
