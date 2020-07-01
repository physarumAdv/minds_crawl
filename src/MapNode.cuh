#ifndef MIND_S_CRAWL_MAPNODE_CUH
#define MIND_S_CRAWL_MAPNODE_CUH


#include "SpacePoint.cuh"

class Particle;

class Polyhedron;


// TODO: add @see to the modified model description to the following docstring
/**
 * Object describing a node of `SimulationMap`
 *
 * This structure describes a node of a simulation map in the Jones' model modified for 3d space
 */
class MapNode
{
public:
    /**
     * Creates a `MapNode` object
     *
     * @param polyhedron The polyhedron to create node on
     * @param polyhedron_face_id The polyhedron's face to create node on
     * @param coordinates The coordinates of node to create node at
     */
    __device__ MapNode(Polyhedron *polyhedron, int polyhedron_face_id, SpacePoint coordinates);

    /// Forbids copying `MapNode` objects
    __host__ __device__ MapNode(const MapNode &) = delete;

    /// Destructs a `MapNode` object
    __device__ ~MapNode();


    /**
     * Sets the left neighbor, only if it was not set already
     *
     * @param value Pointer to new left neighbor
     *
     * @returns `true`, if the neighbor was not set already (so it is updated by the given value), otherwise `false`
     *
     * @note If the given value is `nullptr`, nothing happens, `false` is returned
     *
     * @note This operation is thread-safe
     */
    __device__ bool set_left(MapNode *value);

    /**
     * Sets the top neighbor, only if it was not set already
     *
     * @param value Pointer to new top neighbor
     *
     * @returns `true`, if the neighbor was not set already (so it is updated by the given value), otherwise `false`
     *
     * @note If the given value is `nullptr`, nothing happens, `false` is returned
     *
     * @note This operation is thread-safe
     */
    __device__ bool set_top(MapNode *value);

    /**
     * Sets the right neighbor, only if it was not set already
     *
     * @param value Pointer to new right neighbor
     *
     * @returns `true`, if the neighbor was not set already (so it is updated by the given value), otherwise `false`
     *
     * @note If the given value is `nullptr`, nothing happens, `false` is returned
     *
     * @note This operation is thread-safe
     */
    __device__ bool set_right(MapNode *value);

    /**
     * Sets the bottom neighbor, only if it was not set already
     *
     * @param value Pointer to new bottom neighbor
     *
     * @returns `true`, if the neighbor was not set already (so it is updated by the given value), otherwise `false`
     *
     * @note If the given value is `nullptr`, nothing happens, `false` is returned
     *
     * @note This operation is thread-safe
     */
    __device__ bool set_bottom(MapNode *value);


    /**
     * Returns a pointer to the left neighbor
     *
     * @returns Pointer to the left neighbor
     */
    __device__ MapNode *get_left() const;

    /**
     * Returns a pointer to the top neighbor
     *
     * @returns Pointer to the top neighbor
     */
    __device__ MapNode *get_top() const;

    /**
     * Returns a pointer to the right neighbor
     *
     * @returns Pointer to the right neighbor
     */
    __device__ MapNode *get_right() const;

    /**
     * Returns a pointer to the bottom neighbor
     *
     * @returns Pointer to the bottom neighbor
     */
    __device__ MapNode *get_bottom() const;


    /**
     * Returns the node's coordinates
     *
     * @returns The coordinates of the node
     *
     * @note This parameter is never ever changed during the existence of the object
     */
    __device__ SpacePoint get_coordinates() const;

    /**
     * Returns the polyhedron the node is laying on
     *
     * @returns The pointer to the polyhedron
     *
     * @note This parameter is never ever changed during the existence of the object
     */
    __device__ Polyhedron *get_polyhedron() const;

    /**
     * Returns the id of face the node is laying on
     *
     * @returns The id of face the node belongs to
     *
     * @note This parameter is never ever changed during the existence of the object
     */
    __device__ int get_face_id() const;


    /**
     * Returns whether the node contains food or not
     *
     * @returns True if the node does contain food, False otherwise
     */
    __device__ bool does_contain_food() const;


    /**
     * Attaches the given `Particle` to the node, if it is not occupied already
     *
     * @param value Pointer to the particle to be attached
     *
     * @returns `true`, if the particle was successfully attached (which means the node was not occupied before),
     *      otherwise `false`
     *
     * @note This operation is thread-safe
     *
     * @see MapNode::get_particle, MapNode::remove_particle
     */
    [[nodiscard]] __device__ bool set_particle(Particle *value);

    /**
     * Returns a pointer to the attached particle
     *
     * @returns Pointer to the attached particle, if there is any, otherwise `nullptr`
     *
     * @see mapNode::set_particle, MapNode::remove_particle
     */
    __device__ Particle *get_particle() const;

    /**
     * Marks the node as not occupied (not containing a particle)
     *
     * @note The operation is thread-safe
     *
     * @warning This function <b>does not</b> free memory, allocated for particle, so if you want to free the particle,
     *      you have to do it <b>before</b> calling `remove_particle`. You can obtain a pointer you can use for freeing
     *      memory via `get_particle()`
     *
     * @see MapNode::set_particle, MapNode::get_particle
     */
    __device__ void remove_particle();


    /// Trail value in the node
    double trail;

    /// Temporary trail value in the node (implementation-level field)
    double temp_trail;

private:
    /// Pointer to a neighbor from the corresponding side
    MapNode *left, *top, *right, *bottom;


    /// Polyhedron containing the node
    Polyhedron *polyhedron;

    /// Polyhedron's face the node is located on
    int polyhedron_face_id;

    /// The node's coordinates
    SpacePoint coordinates;


    /// Whether there is food in the current node
    bool contains_food;


    /// Pointer to a particle attached to the node if it exists or TO WHATEVER otherwise
    Particle *particle;
};

#endif //MIND_S_CRAWL_MAPNODE_CUH
