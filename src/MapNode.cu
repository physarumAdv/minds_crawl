#include "MapNode.cuh"
#include "Particle.cuh"
#include "Polyhedron.cuh"


__device__ MapNode::MapNode(const Polyhedron *const polyhedron, int polyhedron_face_id, SpacePoint coordinates) :
        polyhedron(polyhedron), trail(0), contains_food(0), coordinates(coordinates), contains_particle(false),
        polyhedron_face_id(polyhedron_face_id)
{

}

__device__ MapNode::~MapNode()
{
    delete particle;
}
