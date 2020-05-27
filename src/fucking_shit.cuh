/* This file contains all the functions I couldn't place anywhere else.
 * They will be moved somewhere else later
 */

#ifndef MIND_S_CRAWL_FUCKING_SHIT_CUH
#define MIND_S_CRAWL_FUCKING_SHIT_CUH


#include "MapNode.cuh"
#include "SimulationMap.cuh"
#include "common.cuh"


/**
 * Creates a particle in the given node
 *
 * Creates a new `Particle` object with coordinates of the given node and attaches particle to it
 *
 * @param node The node to create particle at
 *
 * @warning This function must not be called if the node contains a particle already, calling it with a node
 * containing a particle causes a memory leak
 */
__device__ void create_particle(MapNode *node);

/**
 * Removes a particle from the given node
 *
 * @param node The node to remove particle from
 *
 * @warning This function must only be called if the node contains a particle already, calling it with node
 * not containing a particle causes undefined behaviour
 */
__device__ void delete_particle(MapNode *node);

/**
 * Diffuses trail in the given node
 *
 * The diffusion algorithm (developed by Jeff Jones) is pretty simple at first sight. We calculate an average
 * `trail` value in a 3x3 node window around the given one and multiply it by `(1 - jones_constants::diffdamp)`.
 * The new `temp_trail` value in the given node is the value just calculated. This is a natural way to implement the
 * smell spread: on each iteration smell moves more far from the source, but becomes less strong, because
 * `(1 - jones_constants::diffdamp)` < 1
 *
 * @param node The node to diffuse trail at
 */
__device__ void diffuse_trail(MapNode *node);

/**
 * Returns number of particles in a node window around the given node
 *
 * Let's assume a node of the model is a plane point with coordinates (X, Y). Then it's neighbors (even if they are not
 * equidistant) will have (X-1, Y-1), (X-1, Y), (X-1, Y+1), (X, Y-1), us, (X, Y+1), (X+1, Y-1), (X+1, Y), (X+1, Y+1).
 * Obviously, we can extrapolate this to neighbors of neighbors and so on. Then a node window around a node
 * with (X, Y) "coordinates" of size W (odd integer) is a set of nodes in the square with side W and central
 * point (X, Y). More formally a node window around (X, Y) node of size W is a set of nodes in a square with opposite
 * vertices (X - (W-1)/2, Y - (W-1)/2) and (X + (W-1)/2, Y + (W-1)/2). <br><br>
 * This function returns a number of particles in a node window around the given node of the given size
 *
 * @param node The node to get node window around
 * @param window_size The node window size
 *
 * @note `window_size` must be a positive odd integer
 *
 * @returns The number of particles in the node window
 */
__device__ ll count_particles_in_node_window(MapNode *node, int window_size);

/**
 * Runs a random death test in the given node
 *
 * Removes a particle in the given node with the `jones_constants::random_death_probability` probability
 *
 * @param node The node to run random death test in
 *
 * @warning This function must only be called if the node contains a particle already, calling it with node
 * not containing a particle causes undefined behaviour
 */
__device__ void random_death_test(MapNode *node);

/**
 * Runs a death test in the given node
 *
 * Let's assume P is a number of particles in a node window around the given node of size `jones_constants::sw`. If
 * `jones_constants::smin` <= P <= `jones_constants::smax`, then nothing happens. Otherwise the particle is removed
 * from the given node
 *
 * @param node The node to run death test in
 *
 * @warning This function must only be called if the node contains a particle already, calling it with node
 * not containing a particle causes undefined behaviour
 */
__device__ void death_test(MapNode *node);

/**
 * Runs a division test in the given node
 *
 * Let's assume P is a number of particles in a node window around the given node of size `jones_constants::gw`. If
 * `jones_constants::gmin` <= P <= `jones_constants::gmax` expression is FALSE, nothing happens. Otherwise (if the
 * expression is true) the division process begins with the `jones_constants::division_probability` probability. In the
 * division process we try to find a node without particle in a 3x3 node window around the given node. If we succeed,
 * we create a particle in the found node. Otherwise nothing happens
 *
 * @param node The node to run division test in
 *
 * @warning This function must only be called if the node contains a particle already, calling it with node
 * not containing a particle causes undefined behaviour
 */
__device__ void division_test(MapNode *node);


/**
     * Greedily looks for a `MapNode` nearest to a given `dest` point
     *
     * A "current" `MapNode` is initialized with `start` value. On each iteration looks a current node's
     * neighbor, which is nearer to `dest`, than the current node. If found it, it's the new value of current node.
     * If current node is nearer than all the neighbors, it is the answer
     *
     * @param dest Destination space point we are trying to find
     * @param start `MapNode` to start going from
     *
     * @returns `MapNode` found by the algorithm
     *
     * @note It's not guaranteed, that a node found with this function is actually the nearest one. Please, see:
     *
     * @see find_nearest_mapnode
     */
__device__ MapNode *find_nearest_mapnode_greedy(const SpacePoint dest, MapNode *const start);

/**
 * Finds a `MapNode` which is nearest to a given destination `SpacePoint`. Tries to find it next to `start`, if
 * it's provided
 *
 * Calls `find_nearest_mapnode_greedy` from the same arguments, if `start` is provided. If the face returned node
 * is located on and the face `dest` is located on are same, the returned node is the answer. Otherwise (or if
 * `start` is not provided) finds a `Face` of `this->polyhedron` to which `dest` is closest, and calls
 * `find_nearest_mapnode_greedy` from dest and some `MapNode` located on the found face
 *
 * @param polyhedron
 * @param dest `SpacePoint` to find nearest `MapNode` to
 * @param start (optional) `MapNode` to start searching from
 *
 * @returns Pointer to a `MapNode` which is considered to be nearest to the given destination
 *
 * @note If the given destination is not located on the simulation's polyhedron, any `MapNode` of the simulation
 * can be returned
 *
 * @see find_nearest_mapnode_greedy
 */
__device__ MapNode *find_nearest_mapnode(const Polyhedron *polyhedron, const SpacePoint dest, MapNode *start=nullptr);


#endif //MIND_S_CRAWL_FUCKING_SHIT_CUH
