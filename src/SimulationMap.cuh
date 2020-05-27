#ifndef MIND_S_CRAWL_SIMULATIONMAP_CUH
#define MIND_S_CRAWL_SIMULATIONMAP_CUH


#include "MapNode.cuh"
#include "fucking_shit.cuh"
#include "common.cuh"


/// Object describing a simulation's map
class SimulationMap
{
public:
    /**
     * Creates a `SimulationMap` object
     *
     * This function isn't implemented yet, neither it's ready to be implemented, so the description stays
     * empty for now
     */
    __device__ SimulationMap(...);

    /// Destructs a `SimulationMap` object
    __device__ ~SimulationMap();

    /**
     * Returns the number of nodes in the simulation
     *
     * @returns The number of nodes on the map
     *
     * @note This number is never ever changed since creation of the object
     */
    __device__ ll get_n_of_nodes() const;

    /**
     * Returns the number of nodes in the simulation
     *
     * @overload SimulationMap::get_n_of_nodes
     */
    __global__ friend void get_n_of_nodes(const SimulationMap *simulation_map, ll *return_value);


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
     * @see SimulationMap::find_nearest_mapnode
     */
    static __device__ MapNode *_find_nearest_mapnode_greedy(SpacePoint dest, MapNode *start);

    /**
     * Finds a `MapNode` which is nearest to a given destination `SpacePoint`. Tries to find it next to `start`, if
     * it's provided
     *
     * Calls `_find_nearest_mapnode_greedy` from the same arguments, if `start` is provided. If the face returned node
     * is located on and the face `dest` is located on are same, the returned node is the answer. Otherwise (or if
     * `start` is not provided) finds a `Face` of `this->polyhedron` to which `dest` is closest, and calls
     * `_find_nearest_mapnode_greedy` from dest and some `MapNode` located on the found face
     *
     * @param dest `SpacePoint` to find nearest `MapNode` to
     * @param start (optional) `MapNode` to start searching from
     *
     * @returns Pointer to a `MapNode` which is considered to be nearest to the given destination
     *
     * @note If the given destination is not located on the simulation's polyhedron, any `MapNode` of the simulation
     * can be returned
     *
     * @see SimulationMap::_find_nearest_mapnode_greedy
     */
    __device__ MapNode *find_nearest_mapnode(const SpacePoint dest, MapNode *start=nullptr);


    /// The array of nodes on the map
    MapNode *nodes;

    /// The polyhedron simulation is runned on
    Polyhedron *const polyhedron;

private:
    /// The number of nodes on the map
    ll n_of_nodes;
};

#endif //MIND_S_CRAWL_SIMULATIONMAP_CUH
