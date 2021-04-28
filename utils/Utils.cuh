#ifndef ALIAS_GPU_UTILS_CUH
#define ALIAS_GPU_UTILS_CUH

#include <string>
#include <thrust/device_vector.h>
#include <csignal>
#include <fstream>

#define RESULT_FOUND 0
#define RESULT_CONTINUE_LEFT 1
#define RESULT_CONTINUE_RIGHT 2

#define ERRCHECK(ans) { \
    cudaError_t code = ans; \
    if (code != cudaSuccess) { \
        fprintf(stderr, "%s:%d: CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(code)); \
        raise(SIGTRAP); \
    } \
}
#define LASTERR { \
    cudaError_t code = cudaGetLastError(); \
    if (code != cudaSuccess) { \
        fprintf(stderr, "%s:%d: CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(code)); \
        raise(SIGTRAP); \
    } \
}

// dest = src, but makes the compiler use a 128 bit load/store operation
#define ASSIGN_128(dest, src) { \
    assert(sizeof(src) == sizeof(int4) && sizeof(dest) == sizeof(int4)); \
    *reinterpret_cast<int4*>(&dest) = *reinterpret_cast<int4*>(&src); \
}

#define OUT_PATH "../data/raw/"
#ifdef NDEBUG
    #define FILEOUT(file, function) { \
        std::cout<<"Writing "<<#function<<" to "<<file<<"..."<<std::endl; \
        int _ = system((std::string("mkdir -p ") + OUT_PATH).c_str()); \
        std::ofstream out((std::string(OUT_PATH) + file).c_str()); \
        auto *backupCout = std::cout.rdbuf(); \
        std::cout.rdbuf(out.rdbuf()); \
        function; \
        std::cout.rdbuf(backupCout); \
    }
#else
    #define FILEOUT(file, function) { \
        std::cout<<"Writing "<<#function<<" to "<<file<<"..."<<std::endl; \
        std::cerr<<"Warning: Trying to generate measurements in debug mode!"<<std::endl; \
        int _ = system((std::string("mkdir -p ") + OUT_PATH).c_str()); \
        std::ofstream out((std::string(OUT_PATH) + file).c_str()); \
        auto *backupCout = std::cout.rdbuf(); \
        std::cout.rdbuf(out.rdbuf()); \
        function; \
        std::cout.rdbuf(backupCout); \
    }
#endif

__host__ __device__ __forceinline__
unsigned long upperPowerOf2(unsigned long v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

#endif //ALIAS_GPU_UTILS_CUH
