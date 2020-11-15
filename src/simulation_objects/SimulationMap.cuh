#ifndef MINDS_CRAWL_SIMULATIONMAP_CUH
#define MINDS_CRAWL_SIMULATIONMAP_CUH


#include "MapNode.cuh"
#include "../main_logic/simulation_logic.cuh"
#include "../common.cuh"


/// Object describing a simulation's map
class SimulationMap
{
public:
    /**
     * Creates a `SimulationMap` object and a grid of nodes
     *
     * @param polyhedron The polyhedron in simulation
     */
    __device__ explicit SimulationMap(Polyhedron *polyhedron);

    /**
     * `SimulationMap` object copy assignment operator (deleted)
     *
     * Deleted because despite copying `SimulationMap` makes sense and is possible to implement, accidental copying of a
     * `SimulationMap` may seriously harm performance and need to copy it is a rather special case. However, a special
     * function for copying `SimulationMap` objects will possibly be implemented someday
     */
    __host__ __device__ SimulationMap &operator=(const SimulationMap &other) = delete;

    /**
     * `SimulationMap` object copy constructor (deleted)
     *
     * Deleted because despite copying `SimulationMap` makes sense and is possible to implement, accidental copying of a
     * `SimulationMap` may seriously harm performance and need to copy it is a rather special case. However, a special
     * function for copying `SimulationMap` objects will possibly be implemented someday
     */
    __host__ __device__ SimulationMap(const SimulationMap &) = delete;

    /// `SimulationMap` object move assignment operator
    __host__ __device__ SimulationMap &operator=(SimulationMap &&other) noexcept;

    /// `SimulationMap` object move constructor
    __host__ __device__ SimulationMap(SimulationMap &&other) noexcept;

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

private:
    /**
     * Finds the index of given face in `SimulationMap::polyhedron->faces` array
     *
     * @param face Pointer to a`Face` to find index of
     *
     * @returns Index of given face in `SimulationMap::polyhedron->faces` array or
     *          `-1` if `face` is not a pointer to an element of `SimulationMap::polyhedron->faces` array
     *
     * @warning This function will find the index of given face in `SimulationMap::polyhedron->faces` array
     *          <b>only</b> if `face` is a pointer to an element of this array,
     *          not if `face` is a pointer to a copy of array element
     */
    __device__ long find_face_index(Face *face) const;


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
    __device__ SpacePoint calculate_neighbor_node_coordinates(int current_node_id, SpacePoint top_direction,
                                                              double angle, bool do_projection) const;

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
     * Calculates direction vector from neighbor of current node to its top neighbor and stores it in the
     * `top_neighbor_directions_for_faces` array, unless it is already calculated
     *
     * @param current_node_id Index of the node whose neighbor was searched
     * @param neighbor_node_id Index of neighbor node
     * @param top_neighbor_directions_for_faces Array of direction vectors to top neighbor nodes for each
     *      polyhedron face
     * @param angle Angle between the top neighbor node and the neighbor node whose index is searched
     *              relative to current node, clockwise is positive direction
     */
    __device__ void set_direction_to_top_neighbor(int current_node_id, int neighbor_node_id,
                                                  SpacePoint *top_neighbor_directions_for_faces, double angle) const;


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
     * @param top_neighbor_directions_for_faces Array of direction vectors to top neighbor nodes for each
     *      polyhedron face
     * @param angle Angle between the top neighbor node and the neighbor node whose index is searched
     *              relative to current node, clockwise is positive direction
     * @param does_face_have_nodes Boolean array whether the faces have nodes or not
     * @param create_new_nodes `true` if new node is allowed to be created, `false` otherwise
     *
     * @returns The index of neighbor node if it has existed or was created, `-1` otherwise
     */
    __device__ int get_neighbor_node_id(int current_node_id, SpacePoint *top_neighbor_directions_for_faces,
                                        double angle, bool *does_face_have_nodes, bool create_new_nodes);


    /// Pointer to the polyhedron simulation is running on
    Polyhedron *polyhedron;

    /// The number of nodes on the map
    int n_of_nodes;
};

#endif //MINDS_CRAWL_SIMULATIONMAP_CUH
