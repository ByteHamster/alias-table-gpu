#include <iostream>
#include <string>

#include "utils/Adder.cuh"

#ifndef ALIAS_GPU_PLOT_ADDER
#define ALIAS_GPU_PLOT_ADDER

#define ADDING_MEASURE_ITERATIONS 1000000
__global__
extern void addNormal(int n, const long *weights, long *output);
__global__
extern void addAtomic(int n, const long *weights, long *output);
__global__
extern void addInterleaved(int n, const long *weights, long *output);
__global__
extern void addAtomicInterleaved(int n, const long *weights, long *output);
__global__
extern void addInterleavedCompat(int n, SafeArray<double> input, SafeArray<double> output);

namespace PlotAdder {
    void benchmarkAddCpuSequential(unsigned int N, long *weights, long *buffer1, long *buffer2) {
        long sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += weights[i];
        }
        std::cout << "CPU;" << sum << ";" << 0 << std::endl;
    }

    void benchmarkAddGpuNormal(unsigned int N, long *weights, long *buffer1, long *buffer2) {
        dim3 threadsPerBlock(ADDER_THREADS_PER_BLOCK);
        dim3 numBlocks(N / (threadsPerBlock.x * ADDER_THREADS_PER_BLOCK) + 1);
        cudaMemcpy(buffer1, weights, N * sizeof(long), cudaMemcpyHostToDevice);

        Timer timer;
        timer.start();
        for (int i = 0; i < ADDING_MEASURE_ITERATIONS; i++) {
            unsigned int n = N;
            while (n > 1) {
                addNormal<<<numBlocks, threadsPerBlock>>>(n, buffer1, buffer2);
                std::swap(buffer1, buffer2);
                n = numBlocks.x;
                numBlocks.x = n / (threadsPerBlock.x * ADDER_THREADS_PER_BLOCK) + 1;
            }
        }
        timer.stop();

        cudaDeviceSynchronize();
        std::cout << "Naive;" << buffer1[0] << ";" << timer.elapsedMillis() << std::endl;
    }

    void benchmarkAddGpuAtomic(unsigned int N, long *weights, long *buffer1, long *buffer2) {
        dim3 threadsPerBlock(ADDER_THREADS_PER_BLOCK);
        dim3 numBlocks(N / (threadsPerBlock.x * ADDER_THREADS_PER_BLOCK) + 1);
        cudaMemcpy(buffer1, weights, N * sizeof(long), cudaMemcpyHostToDevice);

        Timer timer;
        timer.start();
        for (int i = 0; i < ADDING_MEASURE_ITERATIONS; i++) {
            unsigned int n = N;
            while (n > 1) {
                addAtomic<<<numBlocks, threadsPerBlock>>>(n, buffer1, buffer2);
                std::swap(buffer1, buffer2);
                n = numBlocks.x;
                numBlocks.x = n / (threadsPerBlock.x * ADDER_THREADS_PER_BLOCK) + 1;
            }
        }
        timer.stop();

        cudaDeviceSynchronize();
        std::cout << "Atomic;" << buffer1[0] << ";" << timer.elapsedMillis() << std::endl;
    }

    void benchmarkAddGpuInterleaved(unsigned int N, long *weights, long *buffer1, long *buffer2) {
        dim3 threadsPerBlock(ADDER_THREADS_PER_BLOCK);
        dim3 numBlocks(N / (threadsPerBlock.x * ADDER_THREADS_PER_BLOCK) + 1);
        cudaMemcpy(buffer1, weights, N * sizeof(long), cudaMemcpyHostToDevice);

        Timer timer;
        timer.start();
        for (int i = 0; i < ADDING_MEASURE_ITERATIONS; i++) {
            unsigned int n = N;
            while (n > 1) {
                addInterleaved<<<numBlocks, threadsPerBlock>>>(n, buffer1, buffer2);
                std::swap(buffer1, buffer2);
                n = numBlocks.x;
                numBlocks.x = n / (threadsPerBlock.x * ADDER_THREADS_PER_BLOCK) + 1;
            }
        }
        timer.stop();

        cudaDeviceSynchronize();
        std::cout << "Interleaved;" << buffer1[0] << ";" << timer.elapsedMillis() << std::endl;
    }

    void benchmarkAddGpuAtomicInterleaved(unsigned int N, long *weights, long *buffer1, long *buffer2) {
        cudaMemcpy(buffer1, weights, N * sizeof(long), cudaMemcpyHostToDevice);
        dim3 threadsPerBlock(ADDER_THREADS_PER_BLOCK);
        dim3 numBlocks(N / (threadsPerBlock.x * ADDER_THREADS_PER_BLOCK) + 1);

        Timer timer;
        timer.start();
        for (int i = 0; i < ADDING_MEASURE_ITERATIONS; i++) {
            unsigned int n = N;
            while (n > 1) {
                addAtomicInterleaved<<<numBlocks, threadsPerBlock>>>(n, buffer1, buffer2);
                std::swap(buffer1, buffer2);
                n = numBlocks.x;
                numBlocks.x = n / (threadsPerBlock.x * ADDER_THREADS_PER_BLOCK) + 1;
            }
        }
        timer.stop();

        cudaDeviceSynchronize();
        std::cout << "AtomicInterleaved;" << buffer1[0] << ";" << timer.elapsedMillis() << std::endl;
    }

    void plotAddMethods() {
        std::cout << "method;result;time" << std::endl;

        unsigned int N = 10000000;
        long *weights;
        long *buffer1;
        long *buffer2;
        weights = new long[N];

        // Allocate Unified Memory – accessible from CPU or GPU
        cudaMallocManaged(&buffer1, N * sizeof(long));
        cudaMallocManaged(&buffer2, N * sizeof(long));

        for (int i = 0; i < N; i++) {
            weights[i] = static_cast<long>((rand() / static_cast<float>(RAND_MAX)) * 10);
        }

        benchmarkAddCpuSequential(N, weights, buffer1, buffer2);
        benchmarkAddGpuNormal(N, weights, buffer1, buffer2);
        benchmarkAddGpuAtomic(N, weights, buffer1, buffer2);
        benchmarkAddGpuInterleaved(N, weights, buffer1, buffer2);
        benchmarkAddGpuAtomicInterleaved(N, weights, buffer1, buffer2);

        delete[] weights;
        cudaFree(buffer1);
        cudaFree(buffer2);
    }
}
#endif // ALIAS_GPU_PLOT_ADDER
