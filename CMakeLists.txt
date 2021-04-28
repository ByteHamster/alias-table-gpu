cmake_minimum_required(VERSION 3.13)
project(AliasTableGpu CUDA)

set(CMAKE_C_COMPILER /usr/bin/gcc-7)
set(CMAKE_CXX_COMPILER /usr/bin/g++-7)

set(CMAKE_CUDA_STANDARD 14)
if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    message(FATAL_ERROR "CUDA toolkit version must be at least 11.0 to ship cub but is  " ${CMAKE_CUDA_COMPILER_VERSION})
endif()

file(GLOB sourcesTopLevel *.cu)
file(GLOB headersTopLevel *.cuh)
file(GLOB_RECURSE sources construction/**.cu plots/**.cu sampling/**.cu utils/**.cu ${sourcesTopLevel})
file(GLOB_RECURSE headers construction/**.cuh plots/**.cuh sampling/**.cuh utils/**.cuh ${headersTopLevel})

add_executable(AliasTableGpu ${sources} ${headers})
include_directories(.)

if (DEFINED SHARED_MEMORY_WORKER_THREADS)
    set(DEFINES ${DEFINES} -DSHARED_MEMORY_WORKER_THREADS=${SHARED_MEMORY_WORKER_THREADS})
endif()

if (DEFINED PARY_SEARCH_GROUP_SIZE)
    set(DEFINES ${DEFINES} -DPARY_SEARCH_GROUP_SIZE=${PARY_SEARCH_GROUP_SIZE})
endif()

if (DEFINED SPLIT_THREADS_PER_BLOCK)
    set(DEFINES ${DEFINES} -DSPLIT_THREADS_PER_BLOCK=${SPLIT_THREADS_PER_BLOCK})
endif()

if (DEFINED PACK_THREADS_PER_BLOCK)
    set(DEFINES ${DEFINES} -DPACK_THREADS_PER_BLOCK=${PACK_THREADS_PER_BLOCK})
endif()

if (DEFINED PSA_PLUS_THREADS_PER_BLOCK)
    set(DEFINES ${DEFINES} -DPSA_PLUS_THREADS_PER_BLOCK=${PSA_PLUS_THREADS_PER_BLOCK})
endif()

if (DEFINED CHUNKED_GROUP_SIZE)
    set(DEFINES ${DEFINES} -DCHUNKED_GROUP_SIZE=${CHUNKED_GROUP_SIZE})
endif()

if (DEFINED CHUNKED_WORKER_THREADS)
    set(DEFINES ${DEFINES} -DCHUNKED_WORKER_THREADS=${CHUNKED_WORKER_THREADS})
endif()

if (DEFINED CHUNK_SIZE)
    set(DEFINES ${DEFINES} -DCHUNK_SIZE=${CHUNK_SIZE})
endif()

if (DEFINED CHUNK_THRESHOLD_NEXT_PAGE)
    set(DEFINES ${DEFINES} -DCHUNK_THRESHOLD_NEXT_PAGE=${CHUNK_THRESHOLD_NEXT_PAGE})
endif()

# -use_fast_math does not make a difference
if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    target_compile_options(AliasTableGpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-DNDEBUG ${DEFINES}>)
else()
    target_compile_options(AliasTableGpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo ${DEFINES}>)
endif()

set_property(TARGET AliasTableGpu PROPERTY CUDA_ARCHITECTURES 70 75)
set_property(TARGET AliasTableGpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
