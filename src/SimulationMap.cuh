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
     * Creates a `SimulationMap` object and a grid of mapnodes
     *
     * @param polyhedron The polyhedron in simulation
     */
    __device__ SimulationMap(Polyhedron *polyhedron);

    /// Forbids copying `SimulationMap` objects
    __host__ __device__ SimulationMap(const SimulationMap &) = delete;

    /// Destructs a `SimulationMap` object
    __device__ ~SimulationMap();


    /**
     * Returns the index of neighbor node in `nodes` array if it already exists
     * Creates new node in given coordinates if it does not exists and if it is possible and returns its index
     * Returns `-1` if node cannot be created
     *
     * @param current_node_id Index of the node whose neighbor is searched
     * @param neighbor_coordinates Coordinates of neighbor node
     *
     * @returns The index of neighbor node if it exists, `-1` otherwise
     */
    __device__ int get_neighbor_mapnode_id(int current_node_id, SpacePoint neighbor_coordinates,
                                           SpacePoint **nodes_directions, SpacePoint top_direction);

    /**
     * Finds the nearest node to the given point
     *
     * @param point_coordinates Coordinates of the point
     *
     * @returns The index of the found node in `nodes` array
     */
    __device__ int find_nearest_mapnode_to_point(SpacePoint point_coordinates) const;


    /**
     * Returns the number of nodes in the simulation
     *
     * @returns The number of nodes on the map
     *
     * @note This number is never ever changed during the existence of the object
     */
    __device__ int get_n_of_nodes() const;

    /**
     * Returns the number of nodes in the simulation
     *
     * @overload SimulationMap::get_n_of_nodes
     */
    __global__ friend void get_n_of_nodes(const SimulationMap *simulation_map, int *return_value);


    /// The array of nodes on the map
    MapNode *nodes;

    /// The polyhedron simulation is runned on
    Polyhedron *const polyhedron;

private:
    /// The number of nodes on the map
    int n_of_nodes;
};

#endif //MIND_S_CRAWL_SIMULATIONMAP_CUH
