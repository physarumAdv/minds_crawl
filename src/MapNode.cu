#include "MapNode.cuh"
#include "Particle.cuh"
#include "Polyhedron.cuh"
#include "fucking_shit.cuh"


__host__ __device__ MapNode::MapNode(Polyhedron *const polyhedron, Face *polyhedron_face, SpacePoint coordinates) :
        trail(0), temp_trail(0), left(nullptr), top(nullptr), right(nullptr), bottom(nullptr), polyhedron(polyhedron),
        polyhedron_face(polyhedron_face), coordinates(coordinates), contains_food(false), particle(nullptr)
{}

__host__ __device__ MapNode &MapNode::operator=(MapNode &&other) noexcept
{
    if(this != &other)
    {
        particle = nullptr;

        swap(polyhedron, other.polyhedron);
        swap(trail, other.trail);
        swap(temp_trail, other.temp_trail);
        swap(left, other.left);
        swap(top, other.top);
        swap(right, other.right);
        swap(bottom, other.bottom);
        swap(polyhedron_face, other.polyhedron_face);
        swap(coordinates, other.coordinates);
        swap(contains_food, other.contains_food);
        swap(particle, other.particle);
    }

    return *this;
}

__host__ __device__ MapNode::MapNode(MapNode &&other) noexcept
{
    *this = std::move(other);
}

__host__ __device__ MapNode::~MapNode()
{
    delete particle;
}


/**
 * Updates `MapNode`'s neighbor pointer with the given value
 *
 * If the given value is `nullptr` (neighbor is set already) or the `target`'s value is not `nullptr` (trying to set
 * "no neighbor"), nothing happens, otherwise the neighbor value is set
 *
 * @param target Pointer to the neighbor field
 * @param value Neighbor to be set
 *
 * @returns `true`, if the neighbor is updated, otherwise `false`
 *
 * @note This operation is thread-safe when compiled as CUDA code, thread-unsafe when compiled as C++
 */
__device__ inline bool set_neighbor(MapNode **target, MapNode *value)
{
    static_assert(sizeof(target) <= sizeof(unsigned long long *), "I think, I can't safely cast `MapNode **` to "
                                                                  "`unsigned long long *`");

    if(value == nullptr)
        return false;

    return nullptr == (MapNode *)atomicCAS((unsigned long long *)target, (unsigned long long)nullptr,
                                           (unsigned long long)value);
}


__device__ bool MapNode::set_left(MapNode *value)
{
    return set_neighbor(&left, value);
}

__device__ bool MapNode::set_top(MapNode *value)
{
    return set_neighbor(&top, value);
}

__device__ bool MapNode::set_right(MapNode *value)
{
    return set_neighbor(&right, value);
}

__device__ bool MapNode::set_bottom(MapNode *value)
{
    return set_neighbor(&bottom, value);
}


__host__ __device__ MapNode *MapNode::get_left() const
{
    return left;
}

__host__ __device__ MapNode *MapNode::get_top() const
{
    return top;
}

__host__ __device__ MapNode *MapNode::get_right() const
{
    return right;
}

__host__ __device__ MapNode *MapNode::get_bottom() const
{
    return bottom;
}


__host__ __device__ SpacePoint MapNode::get_coordinates() const
{
    return coordinates;
}

__host__ __device__ Polyhedron *MapNode::get_polyhedron() const
{
    return polyhedron;
}

__host__ __device__ Face *MapNode::get_face() const
{
    return polyhedron_face;
}


__host__ __device__ bool MapNode::does_contain_food() const
{
    return contains_food;
}

__host__ __device__ bool MapNode::does_contain_particle() const
{
    return particle != nullptr;
}


[[nodiscard]] __device__ bool MapNode::attach_particle(Particle *p)
{
    static_assert(sizeof(&particle) <= sizeof(unsigned long long *), "I think, I can't safely cast `Particle **` to "
                                                                     "`unsigned long long *`");

    return nullptr == (Particle *)atomicCAS((unsigned long long *)&particle, (unsigned long long)nullptr,
                                            (unsigned long long)p);
}

__host__ __device__ Particle *MapNode::get_particle() const
{
    return particle;
}

__device__ void MapNode::detach_particle()
{
    particle = nullptr;
}

__device__ bool MapNode::detach_particle(Particle *p)
{
    static_assert(sizeof(&particle) <= sizeof(unsigned long long *), "I think, I can't safely cast `Particle **` to "
                                                                     "`unsigned long long *`");

    return p == (Particle *)atomicCAS((unsigned long long *)&particle, (unsigned long long)p,
                                      (unsigned long long)nullptr);
}


__host__ __device__ bool operator==(const MapNode &a, const MapNode &b)
{
    return a.coordinates == b.coordinates;
}
