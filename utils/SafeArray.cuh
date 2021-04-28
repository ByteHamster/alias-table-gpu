#ifndef ALIAS_GPU_SAFEARRAY_CUH
#define ALIAS_GPU_SAFEARRAY_CUH

#include <curand_kernel.h>
#include <cassert>

#include "Utils.cuh"

enum StorageLocation { HOST, DEVICE };

/**
 * std::vector can not be used on the GPU.
 * In order to execute the same source on host and device, this defines a simple alternative.
 * It provides bound checking and prevents multiple malloc calls.
 * @tparam T Type of the array to create.
 */
template<typename T>
class SafeArray {
    public:
        explicit SafeArray(StorageLocation _location) {
            location = _location;
        }

        explicit SafeArray(size_t n, StorageLocation _location) {
            location = _location;
            malloc(n);
        }

        T *data = nullptr;
        size_t size = 0;
        StorageLocation location;

        __host__ __device__ __forceinline__
        T operator [] (size_t i) const {
            assert(i < size);
            return data[i];
        }

        __host__ __device__ __forceinline__
        T& operator [] (size_t i) {
            assert(i < size);
            return data[i];
        }

        void malloc(size_t n) {
            assert(data == nullptr);
            if (location == HOST) {
                data = (T *) std::malloc(n * sizeof(T));
            } else {
                ERRCHECK(cudaMalloc(&data, n * sizeof(T)))
            }
            size = n;
        }

        void free() {
            assert(data != nullptr);
            if (location == HOST) {
                std::free(data);
            } else {
                cudaFree(data);
            }
            data = nullptr;
            size = 0;
        }

        void copyFrom(SafeArray<T> other) {
            assert(other.size > 0);
            assert(size == other.size);
            cudaMemcpyKind memcopyKind;
            if (other.location == HOST && location == DEVICE) {
                memcopyKind = cudaMemcpyHostToDevice;
            } else if (other.location == DEVICE && location == HOST) {
                memcopyKind = cudaMemcpyDeviceToHost;
            } else {
                assert(false && "Not supported");
            }
            ERRCHECK(cudaMemcpy(data, other.data, size * sizeof(T), memcopyKind))
        }

        void usePreInitialized(T *storage, int partSize) {
            assert(partSize > 0);
            data = storage;
            size = partSize;
        }

        void print(std::string title = "", size_t max = 0) {
            if (max == 0) {
                max = size;
            }
            if (location == HOST) {
                printArray(title, data, std::min(max, size));
            } else {
                T *content = static_cast<T *>(std::malloc(size * sizeof(T)));
                cudaMemcpy(content, data, size * sizeof(T), cudaMemcpyDeviceToHost);
                printArray(title, content, std::min(max, size));
                std::free(content);
            }
        }
    private:
        static void printArray(std::string title, T *array, int N) {
            std::cout<<title<<"\t";
            cudaDeviceSynchronize();
            for (int i = 0; i < N; i++) {
                std::cout<<array[i]<<"\t";
            }
            std::cout<<std::endl;
        }
};

#endif //ALIAS_GPU_SAFEARRAY_CUH
