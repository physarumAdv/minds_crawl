#include <utility>

#ifdef COMPILE_FOR_CPU
#include <cmath>
#endif //COMPILE_FOR_CPU

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
        vertices(newalloc_and_copy(vertices, n_of_vertices)), n_of_vertices(n_of_vertices),
        normal(calculate_normal(vertices)), node(nullptr)
{

}

__host__ __device__ Face &Face::operator=(const Face &other)
{
    if(this != &other)
    {
        vertices = newalloc_and_copy(other.vertices, other.n_of_vertices);
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

__host__ __device__ Face::Face()
{
    vertices = nullptr;
}

__host__ __device__ Face::~Face()
{
    delete[] vertices;
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

__host__ __device__ bool Face::contains_point(SpacePoint p)
{
    /*
     * Calculate the area of the face as sum of areas of triangles with vertices `vertices[0]`, `vertices[i]`,
     * `vertices[i + 1]` for all `i` > 1.
     * WARNING: only works for convex polyhedrons.
     */
    double face_area_by_triangles = 0;
    for(int i = 1; i < n_of_vertices - 1; ++i)
    {
        SpacePoint a = vertices[i] - vertices[0];
        SpacePoint b = vertices[i + 1] - vertices[0];
        SpacePoint scalar_product = a % b; // Its length is equal to the area of the parallelogram built on a, b
        face_area_by_triangles += get_distance(origin, scalar_product) / 2;
    }

    /*
     * Calculate the area of the face as sum of areas of triangles with vertices `p` (point to be checked),
     * `vertices[i]`, `vertices[i + 1]` for all `i`
     */
    double face_area_with_point = 0;
    for(int i = 0; i < n_of_vertices; ++i)
    {
        SpacePoint this_vertex = vertices[i], next_vertex = vertices[(i + 1) % n_of_vertices];
        SpacePoint a = this_vertex - p, b = next_vertex - p;
        double triangle_area = 1. / 2 * sqrt((a * a) * (b * b) - (a * b) * (a * b));
        face_area_with_point += triangle_area;
    }

    // The face contains point if and only if the two areas are equal to each other
    return std::abs(face_area_by_triangles - face_area_with_point) < eps;
}


__host__ __device__ bool operator!=(const Face &a, const Face &b)
{
    return !(a == b);
}
