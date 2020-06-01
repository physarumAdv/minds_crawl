#include "curand_kernel.h"
#include "common.cuh"

#include "random_generator.cuh"


__device__ curandState_t state;

__global__ void init_rand(unsigned long long seed)
{
    STOP_ALL_THREADS_EXCEPT_FIRST

                curand_init(seed, 0, 0, &state);
}

__device__ double rand0to1()
{
    return curand_uniform_double(&state);
}
