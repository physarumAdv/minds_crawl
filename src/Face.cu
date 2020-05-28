#include "Face.cuh"
#include "MapNode.cuh"


__device__ SpacePoint get_normal(const SpacePoint *vertices)
{
    SpacePoint normal = (vertices[2] - vertices[0]) % (vertices[1] - vertices[0]);
    return normal / get_distance(normal, origin);
}

__device__ Face::Face(int id, const SpacePoint *vertices, int n_of_vertices, MapNode *node) :
        id(id), vertices(malloc_and_copy(vertices, n_of_vertices)), n_of_vertices(n_of_vertices),
        normal(get_normal(vertices)), node(node)
{

}

__device__ Face::~Face()
{
    free((void *)vertices);
}


bool operator==(const Face &a, const Face &b)
{
    return a.id == b.id;
}
