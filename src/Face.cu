#include "Face.cuh"
#include "MapNode.cuh"
#include "Polyhedron.cuh"


__device__ SpacePoint get_normal(const SpacePoint *vertices)
{
    SpacePoint normal = (vertices[2] - vertices[0]) % (vertices[1] - vertices[0]);
    return normal / get_distance(normal, origin);
}

__device__ Face::Face(int id, const SpacePoint *vertices, int n_of_vertices) :
        id(id), vertices(malloc_and_copy(vertices, n_of_vertices)), n_of_vertices(n_of_vertices),
        normal(get_normal(vertices)), node(nullptr)
{

}

__device__ Face::~Face()
{
    free((void *) vertices);
}


__device__ void Face::set_node(MapNode *node, Polyhedron *polyhedron)
{
    if(polyhedron->find_face_id_by_point(node->get_coordinates()) == id)
    {
        this->node = node;
    }
}

__device__ MapNode *Face::get_node() const
{
    return node;
}


bool operator==(const Face &a, const Face &b)
{
    return a.id == b.id;
}
