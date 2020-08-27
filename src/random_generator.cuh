#ifndef MIND_S_CRAWL_RANDOM_GENERATOR_CUH
#define MIND_S_CRAWL_RANDOM_GENERATOR_CUH


#ifndef COMPILE_FOR_CPU


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


#endif //COMPILE_FOR_CPU


/**
 * Generates a pseudo-random `double` in range [0; 1]
 *
 * @returns A pseudo-random `double` in range [0; 1]
 *
 * @warning To use this function <b>from device code</b> you must firstly initialize random via `init_rand`. However, if
 *      you are calling this function from host code, you don't have to initialize it
 */
__host__ __device__ double rand0to1();


#endif //MIND_S_CRAWL_RANDOM_GENERATOR_CUH
