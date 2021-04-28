#ifndef ALIAS_GPU_PREFIXSUM_CUH
#define ALIAS_GPU_PREFIXSUM_CUH

#include "SafeArray.cuh"
#include "Utils.cuh"
#include "Adder.cuh"

#define PREFIX_SUM_THREADS_PER_BLOCK 512

class PrefixSum {
    public:
        template<typename T>
        static void exclusivePrefixSum(SafeArray<T> input, SafeArray<T> output);

        template<typename T>
        static void exclusivePrefixSumGpu(SafeArray<T> input, SafeArray<T> output);

        template<typename T>
        static void exclusivePrefixSumWorkEfficient(SafeArray<T> input, T *bufferElement, int threadsPerBlock);

        template<typename T>
        static void exclusivePrefixSumGpuCub(SafeArray<T> input);

        static void unitTest();
};

#endif //ALIAS_GPU_PREFIXSUM_CUH
