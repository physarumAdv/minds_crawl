#include "MapNode.cuh"


__device__ MapNode::MapNode(Polyhedron *polyhedron, int polyhedron_face, SpacePoint coordinates) :
        polyhedron(polyhedron), trail(0), temp_trail(0), food(0),
        coordinates(coordinates), contains_particle(false), particle(nullptr),
        polyhedron_face(polyhedron_face)
{

}
