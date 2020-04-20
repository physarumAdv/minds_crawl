#ifndef MIND_S_CRAWL_RANDOM_GENERATOR_CUH
#define MIND_S_CRAWL_RANDOM_GENERATOR_CUH

__global__ void init_rand(unsigned long long seed);

__device__ double rand0to1();

#endif //MIND_S_CRAWL_RANDOM_GENERATOR_CUH
