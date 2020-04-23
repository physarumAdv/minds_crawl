#include "MapNode.cuh"

__device__ MapNode::MapNode(Polyhedron *polyhedron, int polyhedron_face, SpacePoint coordinates) :
        polyhedron(polyhedron), trail(0), temp_trail(0), food(0),
        coordinates(coordinates), contains_particle(false), particle(nullptr),
        left(nullptr), top(nullptr), right(nullptr), bottom(nullptr),
        polyhedron_face(polyhedron_face)
{
    // `is_on_edge` will be defined here
}
