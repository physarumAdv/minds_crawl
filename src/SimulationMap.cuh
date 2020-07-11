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
     * Creates a `SimulationMap` object and a grid of nodes
     *
     * @param polyhedron The polyhedron in simulation
     */
    __device__ SimulationMap(Polyhedron *polyhedron);

    /// Forbids copying `SimulationMap` objects
    __host__ __device__ SimulationMap(const SimulationMap &) = delete;

    /// Destructs a `SimulationMap` object
    __device__ ~SimulationMap();


    /**
     * Returns the number of nodes in the simulation
     *
     * @returns The number of nodes on the map
     *
     * @note This number is never ever changed during the existence of the object
     */
    __device__ int get_n_of_nodes() const;

    /**
     * Saves the number of nodes in the simulation to the given variable
     *
     * @param simulation_map `SimulationMap` object
     * @param return_value Pointer to save answer to
     */
    __global__ friend void get_n_of_nodes(const SimulationMap *simulation_map, int *return_value);


    /// Pointer-represented array of nodes on the map
    MapNode *nodes;

    /// Pointer to the polyhedron simulation is running on
    Polyhedron *const polyhedron;

private:
    /**
     * Returns coordinates of neighbor node with or without projection on polyhedron
     *
     * @param current_node_id Index of the node whose neighbor is searched
     * @param top_direction Direction vector from current node to its top neighbor
     * @param angle Angle between the top neighbor node and the neighbor node whose index is searched
     *              relative to current node, clockwise is positive direction
     * @param do_projection If `true`, counted coordinates will be projected on polyhedron, otherwise they will not
     *
     * @returns Coordinates of neighbor node projected on polyhedron if `do_projection` is `true`,
     *          coordinates of neighbor node without projection on polyhedron otherwise
     */
    __device__ SpacePoint count_neighbor_node_coordinates(int current_node_id, SpacePoint top_direction, double angle,
                                                          bool do_projection) const;


    /**
     * Returns the index of the nearest node to the given point in `nodes` array
     *
     * Searches in the whole `nodes` array to find the nearest node, not using the pointers to neighbor nodes
     *
     * @param dest Coordinates of the point to find the nearest node to
     *
     * @note It's guaranteed that the result is the closest node to `dest`
     *
     * @returns The index of the found node in `nodes` array
     */
    __device__ int find_index_of_nearest_node(SpacePoint dest) const;


    /**
     * Counts direction vector from neighbor of current node to its top neighbor and sets it to `nodes_direction` array
     *
     * @param current_node_id Index of the node whose neighbor was searched
     * @param neighbor_node_id Index of neighbor node
     * @param nodes_directions Pointer to the array of direction vectors to the top neighbor node from each node
     * @param angle Angle between the top neighbor node and the neighbor node whose index is searched
     *              relative to current node, clockwise is positive direction
     */
    __device__ void set_direction_to_top_neighbor(int current_node_id, int neighbor_node_id,
                                                  SpacePoint **nodes_directions, double angle) const;


    /**
     * Returns the index of found or created neighbor node or `-1` in some cases
     *
     * Returns the index of nearest node to neighbor in `nodes` array, if their coordinates are almost the same
     * or if `create_new_nodes` is `false`
     *
     * Creates new node in given coordinates if it does not exists and if it is possible and returns its index
     *
     * Node cannot be created if on the face it belongs to another node exists and
     * their directions to the top neighbor are not the same
     *
     * Returns `-1` if node cannot be created and `create_new_nodes` is `true`
     *
     * @param current_node_id Index of the node whose neighbor is searched
     * @param nodes_directions Pointer to the array of direction vectors to the top neighbor node from each node
     * @param angle Angle between the top neighbor node and the neighbor node whose index is searched
     *              relative to current node, clockwise is positive direction
     * @param create_new_nodes `true` if new node is allowed to be created, `false` otherwise
     *
     * @returns The index of neighbor node if it has existed or was created, `-1` otherwise
     */
    __device__ int get_neighbor_node_id(int current_node_id, SpacePoint **nodes_directions, double angle,
                                        bool create_new_nodes);


    /// The number of nodes on the map
    int n_of_nodes;
};

#endif //MIND_S_CRAWL_SIMULATIONMAP_CUH
