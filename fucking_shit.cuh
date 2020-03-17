/* This file contains all the functions I couldn't place anywhere else.
 * They will be rebased later
 */

#ifndef MIND_S_CRAWL_FUCKING_SHIT_CUH
#define MIND_S_CRAWL_FUCKING_SHIT_CUH

#include "MapPoint.hpp"

typedef long long ll;

__device__ ll get_index(ll x, ll y, ll z, dim3 grid_size);

__device__ void create_particle(MapPoint *p);
__device__ void delete_particle(MapPoint *p);

__device__ void diffuse_trail(MapPoint *grid, ll x, ll y, ll z, dim3 grid_size);

__device__ void random_death_test(MapPoint *p, double death_probability);
__device__ void death_test(MapPoint *grid, ll x, ll y, ll z, dim3 grid_size);
__device__ bool division_test(MapPoint *grid, ll x, ll y, ll z, dim3 grid_size);

#endif //MIND_S_CRAWL_FUCKING_SHIT_CUH
