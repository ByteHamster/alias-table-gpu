#include "SamplerRejectionGiveUp.cuh"

SamplerRejectionGiveUp::SamplerRejectionGiveUp(RejectionSampling &samplingAlgorithm)
        : SamplerGpu<RejectionSampling>(samplingAlgorithm) {

}

std::string SamplerRejectionGiveUp::name() {
    return "SamplerRejectionGiveUp";
}

__host__ __device__ __forceinline__
float fracf(float x) {
    return x - floorf(x);
}

__device__ __forceinline__
int SamplerRejectionGiveUp::sample(SafeArray<int> A, SafeArray<double> weights, int N, double W, XorWowPrng &random) {
    int item;
    int tries = 0;
    while (tries++ < MAX_TRIES) {
        int index = random.next() * A.size;
        item = A[index];
        assert(item < weights.size);
        if (index == 0 || item != A[index - 1]) {
            // Is first item in block
            float weight = weights[item];
            float fraction = fracf((double) (N * weight) / (double) W);
            if (random.next() > fraction) {
                // Reject
                continue;
            }
        }
        return item; // Accept
    }
    return SAMPLE_SKIPPED;
}

__global__
void sampleBenchmarkRejectionGiveUp(SafeArray<int> A, SafeArray<double> weights, int N, double S, int numSamples, int *distributionOutput) {
    XorWowPrng random = XorWowPrng();
    random.initGpu(blockIdx.x * blockDim.x + threadIdx.x);

    int samplesPerThread = numSamples / (blockDim.x * gridDim.x);
    int dummy = 0;
    for (int i = 0; i < samplesPerThread; i++) {
        int sample = SamplerRejectionGiveUp::sample(A, weights, N, S, random);
        if (sample == SAMPLE_SKIPPED) {
            i--;
            continue; // Allow threads to synchronize earlier, so waiting averages out
        }
        assert(sample >= 0);
        assert(sample < N);
        dummy += (sample % 2) + 1;

        #ifdef DEBUG_SUPPORT_SAMPLING_DISTRIBUTION
        if (distributionOutput != nullptr) {
            distributionOutput[sample]++;
        }
        #endif
    }


    while (dummy <= 0) {
        // This endless loop is never executed. Trick compiler into not optimizing away the sample() calls
        dummy = A[0] = dummy % 2;
    }
}

double SamplerRejectionGiveUp::executeBenchmarkSampling(int numSamples, int *distributionOutput) {
    cudaDeviceSynchronize();
    Timer timer;
    timer.start();
    sampleBenchmarkRejectionGiveUp<<<dim3(128), dim3(128)>>>(
            samplingAlgorithm.A, samplingAlgorithm.weightsGpu, samplingAlgorithm.N,
            samplingAlgorithm.W, numSamples, distributionOutput);
    timer.stop();
    return ((double) numSamples) / (timer.elapsedMillis() * 1000 * 1000);
}
