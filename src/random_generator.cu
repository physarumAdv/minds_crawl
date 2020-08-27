#include <random>


std::random_device dev;
std::mt19937 engine(dev());
std::uniform_real_distribution<double> distribution(0., 1.);


#ifndef COMPILE_FOR_CPU


#include "curand_kernel.h"
#include "common.cuh"

#include "random_generator.cuh"


__device__ curandState_t state;

__global__ void init_rand(unsigned long long seed)
{
    STOP_ALL_THREADS_EXCEPT_FIRST;

    curand_init(seed, 0, 0, &state);
}


#endif //COMPILE_FOR_CPU


__host__ __device__ double rand0to1()
{
#ifdef __CUDA_ARCH__
    return curand_uniform_double(&state);
#else
    return distribution(engine);
#endif
}
