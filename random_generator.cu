#include "curand_kernel.h"

#include "random_generator.cuh"


__device__ curandState_t state;

__global__ void init_rand(unsigned long long seed)
{
    curand_init(seed, 0, 0, &state);
}

__device__ double rand01()
{
    return curand_uniform_double(&state);
}
