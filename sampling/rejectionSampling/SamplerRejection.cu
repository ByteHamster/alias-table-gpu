#include "SamplerRejection.cuh"

SamplerRejection::SamplerRejection(RejectionSampling &samplingAlgorithm)
        : SamplerGpu<RejectionSampling>(samplingAlgorithm) {

}

std::string SamplerRejection::name() {
    return "SamplerRejection";
}

__host__ __device__ __forceinline__
float fracf(float x) {
    return x - floorf(x);
}

__device__ __forceinline__
int SamplerRejection::sample(SafeArray<int> A, SafeArray<double> weights, int N, double W, XorWowPrng &random) {
    int item;
    while (true) {
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
}

__global__
void sampleBenchmarkRejection(SafeArray<int> A, SafeArray<double> weights, int N, double S, int numSamples, int *distributionOutput) {
    XorWowPrng random = XorWowPrng();
    random.initGpu(blockIdx.x * blockDim.x + threadIdx.x);

    int samplesPerThread = numSamples / (blockDim.x * gridDim.x);
    int dummy = 0;
    for (int i = 0; i < samplesPerThread; i++) {
        int sample = SamplerRejection::sample(A, weights, N, S, random);
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

double SamplerRejection::executeBenchmarkSampling(int numSamples, int *distributionOutput) {
    cudaDeviceSynchronize();
    Timer timer;
    timer.start();
    sampleBenchmarkRejection<<<dim3(128), dim3(128)>>>(
            samplingAlgorithm.A, samplingAlgorithm.weightsGpu, samplingAlgorithm.N,
            samplingAlgorithm.W, numSamples, distributionOutput);
    timer.stop();
    return ((double) numSamples) / (timer.elapsedMillis() * 1000 * 1000);
}
