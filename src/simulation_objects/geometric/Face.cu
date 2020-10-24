#include <utility>

#include "Face.cuh"
#include "../MapNode.cuh"
#include "Polyhedron.cuh"
#include "../../common.cuh"


__host__ __device__ SpacePoint calculate_normal(const SpacePoint *vertices)
{
    SpacePoint normal = (vertices[2] - vertices[0]) % (vertices[1] - vertices[0]);
    return normal / get_distance(normal, origin);
}


__host__ __device__ Face::Face(const SpacePoint *vertices, int n_of_vertices) :
        vertices(malloc_and_copy(vertices, n_of_vertices)), n_of_vertices(n_of_vertices),
        normal(calculate_normal(vertices)), node(nullptr)
{

}

__host__ __device__ Face &Face::operator=(const Face &other)
{
    if(this != &other)
    {
        vertices = malloc_and_copy(other.vertices, other.n_of_vertices);
        n_of_vertices = other.n_of_vertices;
        normal = other.normal;
        node = nullptr;
    }
    return *this;
}

__host__ __device__ Face::Face(const Face &other)
{
    *this = other;
}

__host__ __device__ Face &Face::operator=(Face &&other) noexcept
{
    if(this != &other)
    {
        swap(vertices, other.vertices);
        swap(n_of_vertices, other.n_of_vertices);
        swap(normal, other.normal);
        swap(node, other.node);
    }

    return *this;
}

__host__ __device__ Face::Face(Face &&other) noexcept
{
    vertices = nullptr;

    *this = std::move(other);
}

__host__ __device__ Face::~Face()
{
    free((void *)vertices);
}


__host__ __device__ void Face::set_node(MapNode *node, Polyhedron *polyhedron)
{
    if(*this == *polyhedron->find_face_by_point(node->get_coordinates()))
    {
        this->node = node;
    }
}

__host__ __device__ MapNode *Face::get_node() const
{
    return node;
}

__host__ __device__ const SpacePoint *Face::get_vertices() const
{
    return vertices;
}

__host__ __device__ int Face::get_n_of_vertices() const
{
    return n_of_vertices;
}

__host__ __device__ SpacePoint Face::get_normal() const
{
    return normal;
}


__host__ __device__ bool operator==(const Face &a, const Face &b)
{
    if(a.n_of_vertices != b.n_of_vertices)
        return false;

    for(int i = 0; i < a.n_of_vertices; ++i)
    {
        if(a.vertices[i] != b.vertices[i])
            return false;
    }
    return true;
}

__host__ __device__ bool operator!=(const Face &a, const Face &b)
{
    return !(a == b);
}
