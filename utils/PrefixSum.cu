#include "PrefixSum.cuh"

template<typename T>
void PrefixSum::exclusivePrefixSum(SafeArray<T> input, SafeArray<T> output) {
    T sum = 0;
    for (int i = 0; i < input.size; i++) {
        T item = input[i]; // input can be the same as output
        output[i] = sum;
        sum += item;
    }
}

template void PrefixSum::exclusivePrefixSum<int>(SafeArray<int> input, SafeArray<int> output);
template void PrefixSum::exclusivePrefixSum<double>(SafeArray<double> input, SafeArray<double> output);

template<typename T>
__global__
static void iteratePrefixSum(int offset, SafeArray<T> input, SafeArray<T> output, int N) {
    unsigned int num = blockIdx.x * blockDim.x + threadIdx.x;
    if (num >= N) {
        return;
    }
    if (num < offset) {
        output[num] = input[num];
    } else {
        output[num] = input[num] + input[num - offset];
    }
}

template __global__ void iteratePrefixSum<int>(int offset, SafeArray<int> input, SafeArray<int> output, int N);
template __global__ void iteratePrefixSum<double>(int offset, SafeArray<double> input, SafeArray<double> output, int N);

template<typename T>
__global__
static void makeExclusive(SafeArray<T> input, SafeArray<T> output, int N) {
    unsigned int num = blockIdx.x * blockDim.x + threadIdx.x;
    if (num >= N) {
        return;
    }
    if (num == 0) {
        output[num] = 0;
    } else {
        output[num] = input[num - 1];
    }
}

template __global__ void makeExclusive<int>(SafeArray<int> input, SafeArray<int> output, int N);
template __global__ void makeExclusive<double>(SafeArray<double> input, SafeArray<double> output, int N);

template<typename T>
void PrefixSum::exclusivePrefixSumGpu(SafeArray<T> input, SafeArray<T> output) {
    dim3 threadsPerBlock(PREFIX_SUM_THREADS_PER_BLOCK);
    dim3 numBlocks((input.size / threadsPerBlock.x) + 1);

    SafeArray<T> buffer(input.size, DEVICE);

    unsigned int offset = 1;
    int roundNumber = 0;
    while (offset <= input.size * 2) {
        if (roundNumber == 0) {
            iteratePrefixSum<<<numBlocks, threadsPerBlock>>>(offset, input, buffer, input.size);
        } else if (roundNumber % 2 == 1) {
            iteratePrefixSum<<<numBlocks, threadsPerBlock>>>(offset, buffer, output, input.size);
        } else {
            iteratePrefixSum<<<numBlocks, threadsPerBlock>>>(offset, output, buffer, input.size);
        }
        offset *= 2;
        roundNumber++;
    }

    if (roundNumber % 2 == 0) {
        // Hack: Copy from output to buffer to have the final output in the right place
        iteratePrefixSum<<<numBlocks, threadsPerBlock>>>(input.size + 1, output, buffer, input.size);
    }
    makeExclusive<<<numBlocks, threadsPerBlock>>>(buffer, output, input.size);

    cudaDeviceSynchronize();
    buffer.free();
}

template void PrefixSum::exclusivePrefixSumGpu<int>(SafeArray<int> input, SafeArray<int> output);
template void PrefixSum::exclusivePrefixSumGpu<double>(SafeArray<double> input, SafeArray<double> output);

#define SPREAD(index) spread * (index) + (spread - 1)

template<typename T>
__global__
static void iteratePrefixSumWorkEfficient1(SafeArray<T> input, int offset, int spread) {
    int blockStart = blockDim.x * blockIdx.x;
    unsigned int index = threadIdx.x;
    for (int stride = 1; stride < blockDim.x; stride*=2) {
        if ((index & stride) == 0) {
            return;
            // Half of the threads is instantly terminated.
            // This is not good but changing this could help with a factor of 2
            // and cub is a lot faster than factor 2.
        }
        int sum = input[offset + SPREAD(blockStart + threadIdx.x)] + input[offset + SPREAD(blockStart + threadIdx.x - stride)];
        __syncthreads();
        input[offset + SPREAD(blockStart + threadIdx.x)] = sum;
        __syncthreads();
    }
}

template __global__ void iteratePrefixSumWorkEfficient1<int>(SafeArray<int> input, int offset, int spread);
template __global__ void iteratePrefixSumWorkEfficient1<double>(SafeArray<double> input, int offset, int spread);

template<typename T>
__global__
static void iteratePrefixSumWorkEfficientSetLastItem(SafeArray<T> input, int offset, int index, T *bufferElement) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (offset == 0) {
            input[index] = 0;
        } else {
            input[index] = input[offset - 1] + bufferElement[1];
            // bufferElement[1] is the item that originally was at input[offset - 1]
            // but got replaced by the exclusive sum
        }
    }
}

template __global__ void iteratePrefixSumWorkEfficientSetLastItem<int>(SafeArray<int> input, int offset, int index, int *bufferElement);
template __global__ void iteratePrefixSumWorkEfficientSetLastItem<double>(SafeArray<double> input, int offset, int index, double *bufferElement);

template<typename T>
__global__
static void iteratePrefixSumWorkEfficient2(SafeArray<T> input, int offset, int spread) {
    int blockStart = blockDim.x * blockIdx.x;
    unsigned int index = threadIdx.x;
    bool active = false;
    for (int stride = blockDim.x; stride >= 2; stride/=2) {
        if (((index + 1) % stride) == 0) {
            active = true;
        }
        if (active) {
            int right = input[offset + SPREAD(blockStart + threadIdx.x)];
            input[offset + SPREAD(blockStart + threadIdx.x)] = right + input[offset + SPREAD(blockStart + threadIdx.x - stride/2)];
            input[offset + SPREAD(blockStart + threadIdx.x - stride/2)] = right;
        }
        __syncthreads();
    }
}

template __global__ void iteratePrefixSumWorkEfficient2<int>(SafeArray<int> input, int offset, int spread);
template __global__ void iteratePrefixSumWorkEfficient2<double>(SafeArray<double> input, int offset, int spread);


template<typename T>
__global__
static void iteratePrefixSumWorkEfficientScrap(int start, SafeArray<T> input, int N, T *bufferElement) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        // The number of items left is smaller than PREFIX_SUM_THREADS_PER_BLOCK.
        // This is not worth it synchronizing.
        return;
    }
    T sum = 0;
    if (start > 0) {
        sum = bufferElement[1] + input[start - 1];
    }
    for (int i = 0; i < N; i++) {
        T itemValue = input[start + i];
        input[start + i] = sum;
        sum += itemValue;
    }
}

template __global__ void iteratePrefixSumWorkEfficientScrap<int>(int start, SafeArray<int> input, int N, int *bufferElement);
template __global__ void iteratePrefixSumWorkEfficientScrap<double>(int start, SafeArray<double> input, int N, double *bufferElement);

template<typename T>
void exclusivePrefixSumGpuWorkEfficientRecurse(SafeArray<T> input, int offset, int N, int spread, T *bufferElement, int threadsPerBlock) {
    dim3 blockSize(threadsPerBlock);

    dim3 numBlocks(N / (blockSize.x));
    int handledSize = blockSize.x * numBlocks.x;
    assert(N - handledSize == 0);
    iteratePrefixSumWorkEfficient1<<<numBlocks, blockSize>>>(input, offset, spread);
    //Utils::printArray("Up  ", input.data, input.size);
    if (numBlocks.x > 1) {
        exclusivePrefixSumGpuWorkEfficientRecurse(input, offset, numBlocks.x, spread * blockSize.x, bufferElement,
                                                  threadsPerBlock);
    }
    if (numBlocks.x == 1) {
        iteratePrefixSumWorkEfficientSetLastItem<<<dim3(1), dim3(1)>>>
                (input, offset, offset + spread * handledSize - 1, bufferElement);
        //Utils::printArray("INNER", input.data, input.size);
    }
    iteratePrefixSumWorkEfficient2<<<numBlocks, blockSize>>>(input, offset, spread);
    //Utils::printArray("Down", input.data, input.size);
}

template<typename T>
__global__
static void iteratePrefixWorkEfficientStoreBuffer(T* bufferElement, SafeArray<T> input, int index) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    bufferElement[1] = bufferElement[0];
    bufferElement[0] = input[index];
}

template __global__ void iteratePrefixWorkEfficientStoreBuffer<int>(int* bufferElement, SafeArray<int> input, int index);
template __global__ void iteratePrefixWorkEfficientStoreBuffer<double>(double* bufferElement, SafeArray<double> input, int index);

template<typename T>
void PrefixSum::exclusivePrefixSumWorkEfficient(SafeArray<T> input, T *bufferElement, int threadsPerBlock) {
    //Utils::printArray("Before", input.data, input.size);
    int handled = 0;
    while (handled < input.size) {
        int unhandled = input.size - handled;
        int currentBatch = 1;
        while (currentBatch * threadsPerBlock <= unhandled) {
            currentBatch *= threadsPerBlock;
        }

        // Letztes Element wird überschrieben, weil exclusive. Das wird aber im nächsten Batch gebraucht!
        iteratePrefixWorkEfficientStoreBuffer<<<dim3(1), dim3(1)>>>(bufferElement, input, handled + currentBatch - 1);

        if (currentBatch < threadsPerBlock) { // Last iteration
            //std::cout<<"Starting Scrap iteration with " << unhandled << " items at position " << handled << std::endl;
            iteratePrefixSumWorkEfficientScrap<<<dim3(1), dim3(1)>>>(handled, input, unhandled, bufferElement);
            //Utils::printArray("Scrap", input.data, input.size);
            break;
        } else {
            //std::cout<<"Starting batch with " << currentBatch << " items at position " << handled << std::endl;
            exclusivePrefixSumGpuWorkEfficientRecurse(input, handled, currentBatch, 1, bufferElement, threadsPerBlock);
        }
        handled += currentBatch;
        //Utils::printArray("Batch", input.data, input.size);
    }
}

template void PrefixSum::exclusivePrefixSumWorkEfficient<int>(SafeArray<int> input, int *bufferElement, int threadsPerBlock);
template void PrefixSum::exclusivePrefixSumWorkEfficient<double>(SafeArray<double> input, double *bufferElement, int threadsPerBlock);

template<typename T>
void PrefixSum::exclusivePrefixSumGpuCub(SafeArray<T> input) {
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    ERRCHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input.data, input.data, input.size))
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    ERRCHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input.data, input.data, input.size))
    // d_out s<-- [0, 8, 14, 21, 26, 29, 29]
    cudaFree(d_temp_storage);
}

template void PrefixSum::exclusivePrefixSumGpuCub<int>(SafeArray<int> input);
template void PrefixSum::exclusivePrefixSumGpuCub<double>(SafeArray<double> input);

__global__
void validatePrefixResults(SafeArray<int> testInGpu, SafeArray<int> testOutGpu,
                           SafeArray<int> testInOutGpuWorkEfficient, SafeArray<int> testInOutGpuCub) {
    for (int i = 0; i < testInGpu.size; i++) {
        assert(testInGpu[i] == i % 7);
    }
    for (int i = 0; i < testOutGpu.size; i++) {
        assert(testOutGpu[i] == testInOutGpuWorkEfficient[i]);
        assert(testOutGpu[i] == testInOutGpuCub[i]);
    }
}

void PrefixSum::unitTest() {
    SafeArray<int> testIn(1000, HOST);
    SafeArray<int> testInGpu(1000, DEVICE);
    SafeArray<int> testInOutGpuWorkEfficient(testIn.size, DEVICE);
    SafeArray<int> testInOutGpuCub(testIn.size, DEVICE);
    SafeArray<int> testOutGpu(testIn.size, DEVICE);
    SafeArray<int> testOutCpu(testIn.size, HOST);

    for (int i = 0; i < testIn.size; i++) {
        testIn[i] = i % 7;
    }
    testInGpu.copyFrom(testIn);
    testInOutGpuWorkEfficient.copyFrom(testIn);
    testInOutGpuCub.copyFrom(testIn);

    PrefixSum::exclusivePrefixSumGpu(testInGpu, testOutGpu);
    int *buffer;
    cudaMallocManaged(&buffer, 2 * sizeof(int));
    PrefixSum::exclusivePrefixSumWorkEfficient(testInOutGpuWorkEfficient, buffer, 64);
    cudaFree(buffer);
    PrefixSum::exclusivePrefixSum(testIn, testOutCpu);
    PrefixSum::exclusivePrefixSumGpuCub(testInOutGpuCub);

    validatePrefixResults<<<1, 1>>>(testInGpu, testOutGpu, testInOutGpuWorkEfficient, testInOutGpuCub);

    testIn.free();
    testInGpu.free();
    testOutGpu.free();
    testOutCpu.free();
    testInOutGpuWorkEfficient.free();
    testInOutGpuCub.free();
}
