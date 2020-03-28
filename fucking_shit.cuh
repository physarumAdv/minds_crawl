/* This file contains all the functions I couldn't place anywhere else.
 * They will be rebased later
 */

#ifndef MIND_S_CRAWL_FUCKING_SHIT_CUH
#define MIND_S_CRAWL_FUCKING_SHIT_CUH

#include "MapPoint.hpp"
#include "Polyhedron.cuh"

typedef long long ll;


__device__ void create_particle(MapPoint *p);
__device__ void delete_particle(MapPoint *p);

__device__ void diffuse_trail(MapPoint *m);

__device__ ll get_particle_window(MapPoint *m, int window_size);

__device__ void random_death_test(MapPoint *p);
__device__ void death_test(MapPoint *m);
__device__ void division_test(MapPoint *m);

#endif //MIND_S_CRAWL_FUCKING_SHIT_CUH
