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

    /// Destructs a `MapNode` object
    __device__ ~MapNode();

    /// Polyhedron containing the node
    Polyhedron *const polyhedron;

    /// Polyhedron's face the node is located on
    const int polyhedron_face_id;

    /**
     * Whether the node has foreign neighbors or not
     * (foreign neighbors are neighbors with different `polyhedron_face_id` value)
     */
    bool is_near_edge;


    /// Trail value in the node
    double trail;

    /// Temporary trail value in the node (implementation-level field)
    double temp_trail;

    /// Whether there is food in the current node
    bool contains_food;


    /// The node's coordinates
    const SpacePoint coordinates;


    /// Pointer to a neighbor from the corresponding side
    MapNode *left, *top, *right, *bottom;


    /// Whether there is a particle attached to the node
    bool contains_particle;

    /// Pointer to a particle attached to the node if it exists or TO WHATEVER otherwise
    Particle *particle;
};

#endif //MIND_S_CRAWL_MAPNODE_CUH
