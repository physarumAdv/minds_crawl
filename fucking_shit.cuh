/* This file contains all the functions I couldn't place anywhere else.
 * They will be rebased later
 */

#ifndef MIND_S_CRAWL_FUCKING_SHIT_CUH
#define MIND_S_CRAWL_FUCKING_SHIT_CUH

#include "MapNode.cuh"
#include "SimulationMap.cuh"

typedef long long ll;


__device__ void create_particle(MapNode *p, Polyhedron *polyhedron, int polyhedron_face)

__device__ void delete_particle(MapNode *p);

__device__ void diffuse_trail(MapNode *m);

__device__ ll get_particle_window(MapNode *m, int window_size);

__device__ void random_death_test(MapNode *p);

__device__ void death_test(MapNode *m);

__device__ void division_test(MapNode *m);

#endif //MIND_S_CRAWL_FUCKING_SHIT_CUH
