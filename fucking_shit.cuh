/* This file contains all the functions I couldn't place anywhere else.
 * They will be rebased later
 */

#ifndef MIND_S_CRAWL_FUCKING_SHIT_CUH
#define MIND_S_CRAWL_FUCKING_SHIT_CUH

#include "MapNode.cuh"
#include "SimulationMap.cuh"

typedef long long ll;


/**
 * Creates a particle in the given node
 *
 * Creates a new `Particle` object with coordinates of the given node and attaches particle to it
 *
 * @param node The node to create particle at
 *
 * @warning This function must not be called if the node contains a particle already, calling it with a node
 * containing a particle already causes a memory leak
 */
__device__ void create_particle(MapNode *node);

/**
 * Removes a particle from the given node
 *
 * @param node The node to remove particle from
 *
 * @warning This function must only be called if the node contains a particle already, calling it with node
 * not containing a particle already causes undefined behaviour
 */
__device__ void delete_particle(MapNode *node);

// TODO: add detailed description to the following docstring
/**
 * Diffuses trail in the current node
 *
 * <detailed description will appear soon>
 *
 * @param node The node tu diffuse trail at
 */
__device__ void diffuse_trail(MapNode *node);

// TODO: add detailed description to the following docstring
/**
 * Returns number of particles in a particle window around the given node
 *
 * <detailed description will appear soon>
 *
 * @param node The node to get particle window around
 * @param window_size The particle window size
 *
 * @return The number of particles in the particle window
 */
__device__ ll get_particle_window(MapNode *node, int window_size);

// TODO: add detailed description to the following docstring
/**
 * Runs a random death test in the given node
 *
 * <detailed description will appear soon>
 *
 * @param node The node to run random death test in
 */
__device__ void random_death_test(MapNode *node);

// TODO: add detailed description to the following docstring
/**
 * Runs a death test in the given node
 *
 * <detailed description will appear soon>
 *
 * @param node The node to run death test in
 */
__device__ void death_test(MapNode *node);

// TODO: add detailed description to the following docstring
/**
 * Runs a division test in the given node
 *
 * <detailed description will appear soon>
 *
 * @param node The node to run division test in
 */
__device__ void division_test(MapNode *node);

#endif //MIND_S_CRAWL_FUCKING_SHIT_CUH
