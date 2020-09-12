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
 * @param node Pointer to the node to create particle at
 *
 * @returns `true` if the `MapNode` was not occupied (which means a new `Particle` was created there), otherwise `false`
 *
 * @note This function does allocate new memory for a new `Particle`, but this memory is freed immediately if the
 *      created `Particle` was not attached to the `MapNode`
 */
[[nodiscard]] __device__ bool create_particle(MapNode *node);

/**
 * Removes a particle from the given node
 *
 * @param node Pointer to the node to remove particle from
 *
 * @returns `true`, if successfully deleted particle (which means there was a particle attached to the node and it
 *      wasn't replace by another one during deleting process), otherwise `false`
 *
 * @warning This operation is thread safe and cannot cause memory leaks, however it's highly recommended to run it only
 *      when there's no way for the particle to be updated by other threads
 */
[[nodiscard]] __device__ bool delete_particle(MapNode *node);

/**
 * Diffuses trail in the given node
 *
 * The diffusion algorithm (developed by Jeff Jones) is pretty simple at first sight. We calculate an average
 * `trail` value in a 3x3 node window around the given one and multiply it by `(1 - jones_constants::diffdamp)`.
 * The new `temp_trail` value in the given node is the value just calculated. This is a natural way to implement the
 * smell spread: on each iteration smell moves more far from the source, but becomes less strong, because
 * `(1 - jones_constants::diffdamp)` < 1
 *
 * @param node Pointer to the node to diffuse trail at
 */
__host__ __device__ void diffuse_trail(MapNode *node);

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
 * @param node Pointer to the node to get node window around
 * @param window_size The node window size
 *
 * @note `window_size` must be a positive odd integer
 *
 * @returns The number of particles in the node window
 */
__host__ __device__ int count_particles_in_node_window(MapNode *node, int window_size);

/**
 * Runs a random death test in the given node
 *
 * Removes a particle in the given node with the `jones_constants::random_death_probability` probability
 *
 * @param node Pointer to the node to run random death test in
 *
 * @returns `true` if a particle was deleted from the node, otherwise `false`
 *
 * @note Return value does not tell you whether the operation was successful. It tells you whether the particle was
 *      removed, that's it
 *
 * @warning This function must only be called if the node contains a particle already, calling it with node
 * not containing a particle causes undefined behaviour
 */
__device__ bool random_death_test(MapNode *node);

/**
 * Runs a death test in the given node
 *
 * Let's assume P is a number of particles in a node window around the given node of size `jones_constants::sw`. If
 * `jones_constants::smin` <= P <= `jones_constants::smax`, then nothing happens. Otherwise the particle is removed
 * from the given node
 *
 * @param node Pointer to the node to run death test in
 *
 * @returns `true` if a particle was deleted from the node, otherwise `false`
 *
 * @note Return value does not tell you whether the operation was successful. It tells you whether the particle was
 *      removed, that's it
 *
 * @warning This function must only be called if the node contains a particle already, calling it with node
 * not containing a particle causes undefined behaviour
 */
__device__ bool death_test(MapNode *node);

/**
 * Runs a division test in the given node
 *
 * Let's assume P is a number of particles in a node window around the given node of size `jones_constants::gw`. If
 * `jones_constants::gmin` <= P <= `jones_constants::gmax` expression is FALSE, nothing happens. Otherwise (if the
 * expression is true) the division process begins with the `jones_constants::division_probability` probability. In the
 * division process we try to find a node without particle in a 3x3 node window around the given node. If we succeed,
 * we create a particle in the found node. Otherwise nothing happens
 *
 * @param node Pointer to the node to run division test in
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
 * Uses pointers to neighbor nodes to find the nearest node to `dest` point, so needs completed grid of nodes
 * with all pointers set (which simply means you can use the function if `SimulationMap` is constructed, and can not if
 * it is not)
 *
 * @param dest Destination space point we are trying to find
 * @param start Pointer to a `MapNode` to start going from
 *
 * @returns Pointer to a `MapNode` found by the algorithm
 *
 * @note Works correctly only if grid of nodes is <b>completed</b>. Even if grid is completed, it's not guaranteed,
 * that a node found with this function is actually the nearest one. Please, see:
 *
 * @see find_nearest_mapnode
 */
__host__ __device__ MapNode *find_nearest_mapnode_greedy(const SpacePoint &dest, MapNode *start);

/**
 * Finds a `MapNode` which is nearest to a given destination `SpacePoint`. Tries to find it next to `start`, if
 * it's provided
 *
 * Calls `find_nearest_mapnode_greedy` from the same arguments, if `start` is provided. If the face returned node
 * is located on and the face `dest` is located on are same, the returned node is the answer. Otherwise (or if
 * `start` is not provided) finds a `Face` of `this->polyhedron` to which `dest` is closest, and calls
 * `find_nearest_mapnode_greedy` from dest and some `MapNode` located on the found face
 *
 * @param polyhedron Pointer to the `Polyhedron` object `MapNode`s correspond to
 * @param dest `SpacePoint` to find nearest `MapNode` to
 * @param start (optional) Pointer to a `MapNode` to start searching from
 *
 * @returns Pointer to a `MapNode` which is considered to be nearest to the given destination
 *
 * @note Works correctly only if grid of nodes is <b>completed</b>. If the given destination is not located
 * on the simulation's polyhedron, any `MapNode` of the simulation can be returned
 *
 * @see find_nearest_mapnode_greedy
 */
__host__ __device__ MapNode *find_nearest_mapnode(const Polyhedron *polyhedron, const SpacePoint &dest,
                                                  MapNode *start = nullptr);


typedef unsigned long long base_atomic_type;


#ifdef COMPILE_FOR_CPU


/**
 * Thread-unsafe version of atomicCAS (for CPU)
 *
 * See cuda documentation on atomicCAS for details
 *
 * @param address Pointer to the variable possibly being updated
 * @param compare Expected value of the variable possibly being updated
 * @param val Value to be set if `*address == compare`
 *
 * @returns Value of `*address` before update (doesn't matter was it really updated)
 */
base_atomic_type atomicCAS(base_atomic_type *address, const base_atomic_type compare, const base_atomic_type val)
{
    base_atomic_type ans = *address;

    if(*address == compare)
        *address = val;

    return ans;
}

/**
 * Thread-unsafe version of atomicAdd (for CPU)
 *
 * See cuda documentation on atomicAdd for details
 *
 * @param address Pointer to the variable being updated
 * @param value Value being added to the variable being updated
 */
base_atomic_type atomicAdd(base_atomic_type *address, base_atomic_type value)
{
    base_atomic_type ans = *address;

    *address += value;

    return ans;
}

/**
 * Thread-unsafe version of atomicAdd (for CPU)
 *
 * @overload atomicAdd
 */
double atomicAdd(double *address, double value)
{
    double ans = *address;

    *address += value;

    return ans;
}


#endif //COMPILE_FOR_CPU

/// Cuda-like atomicCAS implementation for `bool`s (see official CUDA documentation for details)
__device__ bool atomicCAS(bool *address, const bool compare, const bool val);

// I (Nikolay Nechaev, @kolayne) have no idea why the fuck the following only works with #else. If you're reading this
// and now why, PLEASE, contact me and tell me
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val);
#endif


#endif //MIND_S_CRAWL_FUCKING_SHIT_CUH
