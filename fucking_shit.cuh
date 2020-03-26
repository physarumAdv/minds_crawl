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

__device__ void diffuse_trail(Polyhedron *polyhedron, ll i);

__device__ void random_death_test(MapPoint *p);
__device__ void death_test(Polyhedron *polyhedron, ll i);
__device__ bool division_test(Polyhedron *polyhedron, ll i);

#endif //MIND_S_CRAWL_FUCKING_SHIT_CUH
