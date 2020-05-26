#include "Face.cuh"
#include "MapNode.cuh"
#include "common.cuh"


__device__ Face::Face(int id, const int *vertices, int n_of_vertices, MapNode *node) :
        id(id), vertices(malloc_and_copy(vertices, n_of_vertices)), n_of_vertices(n_of_vertices),
        normal(get_normal(vertices, n_of_vertices)), node(node)
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
