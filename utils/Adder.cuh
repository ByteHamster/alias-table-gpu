#ifndef ALIAS_GPU_ADDER_CUH
#define ALIAS_GPU_ADDER_CUH

#include <iostream>
#include <cmath>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>

#include "Timer.cuh"
#include "Utils.cuh"
#include "SafeArray.cuh"

#define ADDER_ITEMS_PER_THREAD 8
#define ADDER_THREADS_PER_BLOCK 512

class Adder {
    public:
        static double addGpu(int N, SafeArray<double> numbers);
        static double addGpuCub(int N, SafeArray<double> numbers);
        static void unitTest();
};

#endif //ALIAS_GPU_ADDER_CUH
