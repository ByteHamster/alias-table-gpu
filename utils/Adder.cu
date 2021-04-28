#include "Adder.cuh"

__global__
void addNormal(int n, const long *weights, long *output) {
    __shared__ long blockResults[ADDER_THREADS_PER_BLOCK];

    unsigned int threadId = blockIdx.x * ADDER_THREADS_PER_BLOCK + threadIdx.x;

    long threadSum = 0;
    for (unsigned int i = threadId * ADDER_ITEMS_PER_THREAD;
         i < (threadId + 1) * ADDER_ITEMS_PER_THREAD && i < n; i++) {
        threadSum += weights[i];
    }
    blockResults[threadIdx.x] = threadSum;
    __syncthreads();

    if (threadIdx.x == 0) {
        long blockSum = 0;
        for (long blockResult : blockResults) {
            blockSum += blockResult;
        }
        output[blockIdx.x] = blockSum;
    }
}

__global__
void addAtomic(int n, const long *weights, long *output) {
    unsigned int threadId = blockIdx.x * ADDER_THREADS_PER_BLOCK + threadIdx.x;

    if (threadIdx.x == 0) {
        output[blockIdx.x] = 0;
    }
    __syncthreads();

    long threadSum = 0;
    for (unsigned int i = threadId * ADDER_ITEMS_PER_THREAD;
         i < (threadId + 1) * ADDER_ITEMS_PER_THREAD && i < n; i++) {
        threadSum += weights[i];
    }
    atomicAdd(reinterpret_cast<int*>(&output[blockIdx.x]), threadSum);
}

__global__
void addInterleaved(int n, const long *weights, long *output) {
    __shared__ long blockResults[ADDER_THREADS_PER_BLOCK];

    long threadSum = 0;
    for (unsigned int i = blockIdx.x * ADDER_THREADS_PER_BLOCK * ADDER_ITEMS_PER_THREAD + threadIdx.x;
         i < (blockIdx.x+1) * ADDER_THREADS_PER_BLOCK * ADDER_ITEMS_PER_THREAD + threadIdx.x && i < n; i += ADDER_THREADS_PER_BLOCK) {
        threadSum += weights[i];
    }
    blockResults[threadIdx.x] = threadSum;
    __syncthreads();

    if (threadIdx.x == 0) {
        long blockSum = 0;
        for (long blockResult : blockResults) {
            blockSum += blockResult;
        }
        output[blockIdx.x] = blockSum;
    }
}

__global__
void addAtomicInterleaved(int n, const long *weights, long *output) {
    if (threadIdx.x == 0) {
        output[blockIdx.x] = 0;
    }
    __syncthreads();

    long threadSum = 0;
    for (unsigned int i = blockIdx.x * ADDER_THREADS_PER_BLOCK * ADDER_ITEMS_PER_THREAD + threadIdx.x;
         i < (blockIdx.x+1) * ADDER_THREADS_PER_BLOCK * ADDER_ITEMS_PER_THREAD + threadIdx.x && i < n; i += ADDER_THREADS_PER_BLOCK) {
        threadSum += weights[i];
    }
    atomicAdd(reinterpret_cast<int*>(&output[blockIdx.x]), threadSum);
}

__global__
void addInterleavedCompat(int n, SafeArray<double> input, SafeArray<double> output) {
    #if __CUDA_ARCH__ < 600
        __shared__ double blockResults[ADDER_THREADS_PER_BLOCK];
    #endif

    double threadSum = 0;
    for (unsigned int i = blockIdx.x * ADDER_THREADS_PER_BLOCK * ADDER_ITEMS_PER_THREAD + threadIdx.x;
         i < (blockIdx.x+1) * ADDER_THREADS_PER_BLOCK * ADDER_ITEMS_PER_THREAD + threadIdx.x && i < n; i += ADDER_THREADS_PER_BLOCK) {
        threadSum += input[i];
    }

    #if __CUDA_ARCH__ < 600
        blockResults[threadIdx.x] = threadSum;
        __syncthreads();

        if (threadIdx.x == 0) {
            double blockSum = 0;
            for (double blockResult : blockResults) {
                blockSum += blockResult;
            }
            output[blockIdx.x] = blockSum;
        }
    #else
        atomicAdd(&output[blockIdx.x], threadSum);
    #endif
}

double Adder::addGpu(int N, SafeArray<double> numbers) {
    dim3 threadsPerBlock(ADDER_THREADS_PER_BLOCK);
    dim3 numBlocks(N / (threadsPerBlock.x * ADDER_ITEMS_PER_THREAD) + 1);

    int firstOutputSize = N / (threadsPerBlock.x * ADDER_ITEMS_PER_THREAD) + 1;
    int secondOutputSize = firstOutputSize / (threadsPerBlock.x * ADDER_ITEMS_PER_THREAD) + 1;

    SafeArray<double> buffer1(firstOutputSize, DEVICE);
    SafeArray<double> buffer2(secondOutputSize, DEVICE);

    unsigned int n = N;
    int round = 0;
    while (n > 1) {
        if (round == 0) {
            addInterleavedCompat<<<numBlocks, threadsPerBlock>>>(n, numbers, buffer1);
        } else if (round % 2 == 1) {
            addInterleavedCompat<<<numBlocks, threadsPerBlock>>>(n, buffer1, buffer2);
        } else {
            addInterleavedCompat<<<numBlocks, threadsPerBlock>>>(n, buffer2, buffer1);
        }
        n = numBlocks.x;
        numBlocks.x = n / (threadsPerBlock.x * ADDER_ITEMS_PER_THREAD) + 1;
        round++;
    }

    cudaDeviceSynchronize();
    double result;
    if (round % 2 == 1) {
        ERRCHECK(cudaMemcpy(&result, &buffer1[0], sizeof(double), cudaMemcpyDeviceToHost))
    } else {
        ERRCHECK(cudaMemcpy(&result, &buffer2[0], sizeof(double), cudaMemcpyDeviceToHost))
    }
    buffer1.free();
    buffer2.free();
    return result;
}

double Adder::addGpuCub(int N, SafeArray<double> numbers) {
    size_t tempStorageBytes;
    double* tempStorage = nullptr;
    double* result = nullptr;
    ERRCHECK(cub::DeviceReduce::Sum(tempStorage, tempStorageBytes, numbers.data, result, N))
    cudaMalloc(&result, 1 * sizeof(double));
    cudaMalloc(&tempStorage, tempStorageBytes);
    ERRCHECK(cub::DeviceReduce::Sum(tempStorage, tempStorageBytes, numbers.data, result, N))
    double sum;
    cudaMemcpy(&sum, result, 1 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(tempStorage);
    cudaFree(result);
    return sum;
}

void Adder::unitTest() {
    SafeArray<double> numbers(1000, HOST);
    for (int i = 0; i < numbers.size; i++) {
        numbers[i] = i;
    }
    SafeArray<double> numbersGpu(1000, DEVICE);
    numbersGpu.copyFrom(numbers);

    cudaDeviceSynchronize();
    assert((1000 * 999) / 2 == Adder::addGpu(numbersGpu.size, numbersGpu));
    assert((1000 * 999) / 2 == Adder::addGpuCub(numbersGpu.size, numbersGpu));
    numbers.free();
}
