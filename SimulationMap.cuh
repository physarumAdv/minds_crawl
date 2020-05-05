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
    // TODO: add a destructor

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
    friend __global__ void get_n_of_nodes(const SimulationMap *simulation_map, ll *return_value);


    /// The array of nodes on the map
    MapNode *nodes;

private:
    /// The number of nodes on the map
    ll n_of_nodes;
};

#endif //MIND_S_CRAWL_SIMULATIONMAP_CUH
