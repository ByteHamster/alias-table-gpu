#include <iostream>
#include <string>

#include "utils/Adder.cuh"
#include "construction/aliasTable/buildMethod/AliasTableSplit.cuh"

#ifndef ALIAS_GPU_PLOT_OTHERS
#define ALIAS_GPU_PLOT_OTHERS
namespace PlotOthers {

    void plotPrefixSumSpeed(int n, std::string name, void (*executor)(SafeArray<int>, int *)) {
        int numMeasurements = 20;
        SafeArray<int> dataHost(n, HOST);
        for (int i = 0; i < dataHost.size; i++) {
            dataHost[i] = i % 7;
        }
        SafeArray<int> data(n, DEVICE);
        data.copyFrom(dataHost);
        int *buffer;
        cudaMallocManaged(&buffer, 2 * sizeof(int));

        Timer timer;
        timer.start();
        for (int i = 0; i < numMeasurements; i++) {
            executor(data, buffer);
        }
        timer.stop();
        cudaFree(buffer);
        data.free();
        dataHost.free();
        std::cout << name << ";" << timer.elapsedMillis() / numMeasurements << ";" << n << std::endl;
    }

    void plotPrefixSumSpeed() {
        std::cout << "type;duration;N" << std::endl;
        for (int n = 16 * 16 * 16; n <= 1200000; n += 16 * 16 * 16) {
            plotPrefixSumSpeed(n, "Work efficient (32 threads/block)", [](SafeArray<int> data, int *buffer) {
                PrefixSum::exclusivePrefixSumWorkEfficient(data, buffer, 32);
            });
            plotPrefixSumSpeed(n, "Work efficient (64 threads/block)", [](SafeArray<int> data, int *buffer) {
                PrefixSum::exclusivePrefixSumWorkEfficient(data, buffer, 64);
            });
            plotPrefixSumSpeed(n, "Naive", [](SafeArray<int> data, int *buffer) {
                PrefixSum::exclusivePrefixSumGpu(data, data);
            });
            //plotPrefixSumSpeed(n, "Thrust", [](SafeArray<int> data, int *buffer) {
            //    thrust::exclusive_scan(thrust::device_pointer_cast(data.data),
            //                           thrust::device_pointer_cast(data.data + data.size),
            //                           thrust::device_pointer_cast(data.data));
            //});
            plotPrefixSumSpeed(n, "Cub", [](SafeArray<int> data, int *buffer) {
                PrefixSum::exclusivePrefixSumGpuCub(data);
            });
        }
    }

    template <typename Prng>
    __global__
    void testPrngSpeed(int numPerThread, int *dummyResult, Prng random) {
        random.initGpu(blockIdx.x * blockDim.x + threadIdx.x);

        int x = 0;
        for (int i = 0; i < numPerThread; i++) {
            x += random.next() * 20;
        }
        atomicAdd_system(dummyResult, x);
    }

    /**
     * Returns GSamples/s
     */
    template <typename Prng>
    void performPrngTest(Prng random, int numBlocks) {
        int *dummyResult;
        cudaMalloc(&dummyResult, sizeof(int));
        int numThreads = 128;
        int samplesPerThread = 1e5;
        random.initCpu(numBlocks);
        Timer timer;
        timer.start();
        testPrngSpeed<Prng><<<numBlocks, numThreads>>>(samplesPerThread, dummyResult, random);
        timer.stop();
        std::cout<<numBlocks<<";"<<Prng::name()<<";"<<
            ((double) samplesPerThread * numThreads * numBlocks) / (timer.elapsedMillis() * 1000 * 1000)<<std::endl;
        random.free();
        cudaFree(dummyResult);
    }

    void plotPrngSpeed() {
        std::cout<<"numBlocks;type;GSamples"<<std::endl;
        for (int numBlocks = 5; numBlocks <= 200; numBlocks += 1) {
            performPrngTest(MtPrng(), numBlocks);
            performPrngTest(XorWowPrng(), numBlocks);
        }
    }
}
#endif // ALIAS_GPU_PLOT_OTHERS
