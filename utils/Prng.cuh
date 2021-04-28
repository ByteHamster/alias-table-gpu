#ifndef ALIAS_GPU_PRNG_CUH
#define ALIAS_GPU_PRNG_CUH

#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

#include "utils/Utils.cuh"
#include "utils/hashing/Sha1.cuh"
#include "utils/hashing/Md5.cuh"

class MtPrng {
    public:
        void initCpu(int numBlocks) {
            assert(numBlocks <= 200); // https://docs.nvidia.com/cuda/curand/device-api-overview.html#bit-generation-2
            ERRCHECK(cudaMalloc(&state, numBlocks * sizeof(curandStateMtgp32)))
            ERRCHECK(cudaMalloc(&kernel_params, sizeof(mtgp32_kernel_params)))

            if (curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, kernel_params) != CURAND_STATUS_SUCCESS) {
                std::cerr<<"Unable to initialize curand params"<<std::endl;
                raise(SIGTRAP);
            }
            if (curandMakeMTGP32KernelState(state, mtgp32dc_params_fast_11213,
                                            kernel_params, numBlocks, random()) != CURAND_STATUS_SUCCESS) {
                std::cerr<<"Unable to initialize curand state"<<std::endl;
                raise(SIGTRAP);
            }
        }

        __device__
        void initGpu(int threadNumber) {

        }

        void free() {
            cudaFree(state);
            cudaFree(kernel_params);
        }

        __device__ __forceinline__
        double next() {
            assert(threadIdx.x <= 256);
            // curand_uniform may return from 0.0 to 1.0, where 1.0 is included and 0.0 is excluded
            double res = 1.0 - curand_uniform_double(&state[blockIdx.x]);
            assert(0.0 <= res); assert(res < 1.0);
            return res;
        }

        static std::string name() {
            return "MersenneTwister";
        }

    private:
        mtgp32_kernel_params *kernel_params = nullptr;
        curandStateMtgp32 *state = nullptr;
};

class XorWowPrng {
    public:
        curandStateXORWOW state;

        void initCpu(int numBlocks) {

        }

        __device__
        void initGpu(int threadNumber, long seed = -1) {
            if (seed == -1) {
                seed = clock64();
            }
            curand_init(seed, threadNumber, 0, &state);
        }

        void free() {

        }

        __device__ __forceinline__
        double next() {
            // curand_uniform may return from 0.0 to 1.0, where 1.0 is included and 0.0 is excluded
            double res = 1.0 - curand_uniform_double(&state);
            assert(0.0 <= res); assert(res < 1.0);
            return res;
        }

        static std::string name() {
            return "XorWow";
        }

        __device__ __forceinline__
        void skipAhead(int n) {
            skipahead_sequence(n, &state);
        }
};

struct HashNone {
    static __device__ __forceinline__ int perform(int seed) {
        return seed;
    }

    static std::string name() {
        return "NoHash";
    }
};

struct HashMd5 {
    static __device__ __forceinline__ int perform(int seed) {
        const size_t inLength = sizeof(int);
        BYTE in[inLength];
        ((int *) in)[0] = seed;
        BYTE out[16];
        CUDA_MD5_CTX ctx;
        cuda_md5_init(&ctx);
        cuda_md5_update(&ctx, in, inLength);
        cuda_md5_final(&ctx, out);
        return *((int *) out);
    }

    static std::string name() {
        return "Md5";
    }
};

struct HashSha1 {
    static __device__ __forceinline__ int perform(int seed) {
        const size_t inLength = sizeof(int);
        BYTE in[inLength];
        ((int *) in)[0] = seed;
        BYTE out[20];
        CUDA_SHA1_CTX ctx;
        cuda_sha1_init(&ctx);
        cuda_sha1_update(&ctx, in, inLength);
        cuda_sha1_final(&ctx, out);
        return *((int *) out);
    }

    static std::string name() {
        return "Sha1";
    }
};

template <typename Hash>
class XorWowPkPtPrng {
    public:
        curandStateXORWOW state;

        void initCpu(int numBlocks) {

        }

        __device__
        void initGpu(int threadNumber) {
            int seed = threadNumber ^ clock64();
            seed = Hash::perform(seed);
            curand_init(seed, 0, 0, &state);
        }

        void free() {

        }

        __device__ __forceinline__
        double next() {
            // curand_uniform may return from 0.0 to 1.0, where 1.0 is included and 0.0 is excluded
            double res = 1.0 - curand_uniform_double(&state);
            assert(0.0 <= res); assert(res < 1.0);
            return res;
        }

        static std::string name() {
            return "XorWowPkPt" + Hash::name();
        }

        __device__ __forceinline__
        void skipAhead(int n) {
            skipahead_sequence(n, &state);
        }
};

class CpuPrng {
    public:
        float next() {
            // ((double) rand()) is constant but ((double)(int) rand()) is not?! Why?
            float res = 0.9999999f * ((double) (int) rand()) / (double) ((unsigned) RAND_MAX + 1);
            assert(0 <= res); assert(res < 1);
            return res;
        }
};

class Binomial {
    public:
        __device__ __forceinline__
        static int draw(int N, float p, curandStateXORWOW &randomState) {
            float mu_normal = p * N;
            float sigma_normal = sqrt(p * N * (1 - p));
            float normal = curand_normal(&randomState);
            return sigma_normal * normal + mu_normal;
        }
};

#endif //ALIAS_GPU_PRNG_CUH
