#ifndef MIND_S_CRAWL_MAPNODE_CUH
#define MIND_S_CRAWL_MAPNODE_CUH


#include "Polyhedron.cuh"
#include "SpacePoint.hpp"
#include "Particle.cuh"


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
     * @param polyhedron_face The polyhedron's face to create node on
     * @param coordinates The coordinates of node to create node at
     */
    __device__ MapNode(const Polyhedron *polyhedron, int polyhedron_face, SpacePoint coordinates);
    // TODO: add a destructor

    /// Polyhedron containing the node
    const Polyhedron *const polyhedron;

    /// Polyhedron's face the node is located on
    const int polyhedron_face;

    /**
     * Whether the node has foreign neighbors or not
     * (foreign neighbors are neighbors with different `polyhedron_face` value)
     */
    bool is_on_edge;


    /// Trail value in the node
    double trail;

    /// Temporary trail value in the node (implementation-level field)
    double temp_trail;

    /// Food value in the node
    double food;


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
