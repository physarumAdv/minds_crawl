#ifndef MINDS_CRAWL_MAPNODE_CUH
#define MINDS_CRAWL_MAPNODE_CUH


#include "../common.cuh"
#include "geometric/SpacePoint.cuh"
#include "geometric/Face.cuh"

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
     * @param polyhedron Pointer to the polyhedron to create node on
     * @param polyhedron_face The polyhedron's face to create node on
     * @param coordinates Coordinates of node to create node at
     */
    __host__ __device__ MapNode(Polyhedron *polyhedron, Face *polyhedron_face, SpacePoint coordinates);

    /**
     * `MapNode` object copy assignment operator (deleted)
     *
     * Deleted because the need to copy a `MapNode` is a rather special case and can be done with other methods, while
     * an accidental copying of a `MapNode` can result into unexpected things (for example, when copying a `MapNode`
     * the `Particle` from the original node shouldn't be copied, and unwanted copy may create an impression that
     * there is no particle in a node, but in fact it will be checking a copy of an original node)
     */
    __host__ __device__ MapNode &operator=(const MapNode &other) = delete;

    /**
     * `MapNode` object copy constructor (deleted)
     *
     * Deleted because the need to copy a `MapNode` is a rather special case and can be done with other methods, while
     * an accidental copying of a `MapNode` can result into unexpected things (for example, when copying a `MapNode`
     * the `Particle` from the original node shouldn't be copied, and unwanted copy may create an impression that
     * there is no particle in a node, but in fact it will be checking a copy of an original node)
     */
    __host__ __device__ MapNode(const MapNode &) = delete;

    /// `MapNode` object move assignment operator
    __host__ __device__ MapNode &operator=(MapNode &&other) noexcept;

    /// `MapNode` object move constructor
    __host__ __device__ MapNode(MapNode &&other) noexcept;

    /**
     * Creates a <b>fake</b> `MapNode` object
     *
     * Creates a `MapNode` object such that it can be destructed safely, but must not be used for any purposes. All the
     * fields, which don't take part in the destruction, stay uninitialized.
     *
     * @warning The behaviour is undefined if you use the object constructed with this constructor in any way but
     *      destructing it (the object can be destructed safely)
     */
    __host__ __device__ MapNode();

    /// Destructs a `MapNode` object
    __host__ __device__ ~MapNode();


    /**
     * Sets the left neighbor, only if it was not set already
     *
     * @param value Pointer to new left neighbor
     *
     * @returns `true`, if the neighbor was not set already (so it is updated by the given value), otherwise `false`
     *
     * @note If the given value is `nullptr`, nothing happens, `false` is returned
     *
     * @note This operation is thread-safe when compiled as CUDA code, thread-unsafe when compiled as C++
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
     * @note This operation is thread-safe when compiled as CUDA code, thread-unsafe when compiled as C++
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
     * @note This operation is thread-safe when compiled as CUDA code, thread-unsafe when compiled as C++
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
     * @returns Pointer to the left neighbor (`nullptr` if it is not set)
     */
    __host__ __device__ MapNode *get_left() const;

    /**
     * Returns a pointer to the top neighbor
     *
     * @returns Pointer to the top neighbor (`nullptr` if it is not set)
     */
    __host__ __device__ MapNode *get_top() const;

    /**
     * Returns a pointer to the right neighbor
     *
     * @returns Pointer to the right neighbor (`nullptr` if it is not set)
     */
    __host__ __device__ MapNode *get_right() const;

    /**
     * Returns a pointer to the bottom neighbor
     *
     * @returns Pointer to the bottom neighbor (`nullptr` if it is not set)
     */
    __host__ __device__ MapNode *get_bottom() const;


    __host__ __device__ bool does_contain_particle() const;

    /**
     * Returns the node's coordinates
     *
     * @returns The coordinates of the node
     *
     * @note This parameter is never ever changed during the existence of the object
     */
    __host__ __device__ SpacePoint get_coordinates() const;

    /**
     * Returns the polyhedron the node is laying on
     *
     * @returns The pointer to the polyhedron
     *
     * @note This parameter is never ever changed during the existence of the object
     */
    __host__ __device__ Polyhedron *get_polyhedron() const;

    /**
     * Returns pointer to the face the node is laying on
     *
     * @returns Pointer to the face the node belongs to
     *
     * @note This parameter is never ever changed during the existence of the object
     */
    __host__ __device__ Face *get_face() const;


    /**
     * Returns whether the node contains food or not
     *
     * @returns True if the node does contain food, False otherwise
     */
    __host__ __device__ bool does_contain_food() const;


    /**
     * Attaches the given `Particle` to the node, if it is not occupied already
     *
     * @param p Pointer to the particle to be attached
     *
     * @returns `true`, if the particle was successfully attached (which means the node was not occupied before),
     *      otherwise `false`
     *
     * @note This operation is thread-safe
     *
     * @see MapNode::get_particle, MapNode::detach_particle
     */
    [[nodiscard]] __device__ bool attach_particle(Particle *p);

    /**
     * Returns a pointer to the attached particle
     *
     * @returns Pointer to the attached particle, if there is any, otherwise `nullptr`
     *
     * @see mapNode::attach_particle, MapNode::detach_particle
     */
    __host__ __device__ Particle *get_particle() const;

    /**
     * Marks the node as not occupied (not containing a particle) / Detaches particle from the node
     *
     * @note The operation is thread-safe
     *
     * @warning Detaching particle from a map node <b>does not</b> free memory, allocated for `Particle`, so if you want
     *      to free memory, you have to firstly obtain a pointer to the `Particle` if you don't have it yet (can be done
     *      via `get_particle()`), then detach the particle from it's node (call `detach_particle()`), and then free
     *      memory.
     *
     * @warning Remember about thread-safety: `MapNode` does not guarantee that the `Particle` being removed didn't change
     *      since calling `get_particle()`
     *
     * @see MapNode::attach_particle, MapNode::get_particle
     */
    __device__ void detach_particle();

    /**
     * Detaches the given `Particle` from the `MapNode`, if it is attached
     *
     * @param p Pointer to the `Particle` to be detached
     *
     * @returns `true`, if the given `Particle` was attached to the node (which means it was successfully removed),
     *      otherwise `false`
     *
     * @note This operation is thread-safe
     */
    __device__ bool detach_particle(Particle *p);


    /**
     * Checks whether two `MapNode`s are same (checked using coordinates)
     *
     * @param a `MapNode` object
     * @param b `MapNode` object
     *
     * @returns `true` if two mapnodes have same coordinates, `false` otherwise
     */
    __host__ __device__ friend bool operator==(const MapNode &a, const MapNode &b);


    /// Trail value in the node
    double trail;

    /// Temporary trail value in the node (implementation-level field)
    double temp_trail;

private:
    /// Pointer to a neighbor from the corresponding side
    MapNode *left, *top, *right, *bottom;


    /// Polyhedron containing the node
    Polyhedron *polyhedron;

    /// Pointer to the polyhedron's face the node is located on
    Face *polyhedron_face;

    /// The node's coordinates
    SpacePoint coordinates;


    /// Whether there is food in the current node
    bool contains_food;


    /// Pointer to a particle attached to the node if it exists or TO WHATEVER otherwise
    Particle *particle;
};


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


#endif //MINDS_CRAWL_MAPNODE_CUH
