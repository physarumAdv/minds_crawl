#include "MapNode.cuh"


__device__ MapNode::MapNode(const Polyhedron *const polyhedron, int polyhedron_face, SpacePoint coordinates) :
        polyhedron(polyhedron), trail(0), food(0), coordinates(coordinates), contains_particle(false),
        polyhedron_face(polyhedron_face)
{

}

__device__ MapNode::~MapNode()
{
    delete particle;
}
