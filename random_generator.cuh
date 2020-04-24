#ifndef MIND_S_CRAWL_RANDOM_GENERATOR_CUH
#define MIND_S_CRAWL_RANDOM_GENERATOR_CUH

/**
 * Initializes cuRAND
 *
 * Calling it will initialize a `curandState`, which will be used to generate random numbers via `rand0to1`.
 * In fact it just calls `curand_init(seed, 0, 0, <local_variable>)`
 *
 * @param seed The variable to be passed to `curand_init` as an argument
 *
 * @see curand_init
 */
__global__ void init_rand(unsigned long long seed);

/**
 * Generates a pseudo-random `double` in range [0; 1]
 *
 * @returns A pseudo-random `double` in range [0; 1]
 *
 * @warning To use this function you firstly need to initialize random via `init_rand`
 */
__device__ double rand0to1();

#endif //MIND_S_CRAWL_RANDOM_GENERATOR_CUH
