#include "MapNode.cuh"
#include "Particle.cuh"
#include "Polyhedron.cuh"


__device__ MapNode::MapNode(Polyhedron *polyhedron, Face *polyhedron_face, SpacePoint coordinates) :
        polyhedron(polyhedron), trail(0), contains_food(false), coordinates(coordinates),
        polyhedron_face(polyhedron_face), left(nullptr), top(nullptr), right(nullptr), bottom(nullptr)
{}

__device__ MapNode::~MapNode()
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
 */
__device__ inline bool set_mapnode_neighbor(MapNode **target, MapNode *value)
{
    static_assert(sizeof(target) <= sizeof(unsigned long long *), "I think, I can't safely cast `MapNode **` to"
                                                                  "`unsigned long long *`");

    if(value == nullptr)
        return false;

    return nullptr == (MapNode *)atomicCAS((unsigned long long *)target, (unsigned long long)nullptr,
                                           (unsigned long long)value);
}


__device__ bool MapNode::set_left(MapNode *value)
{
    return set_mapnode_neighbor(&left, value);
}

__device__ bool MapNode::set_top(MapNode *value)
{
    return set_mapnode_neighbor(&top, value);
}

__device__ bool MapNode::set_right(MapNode *value)
{
    return set_mapnode_neighbor(&right, value);
}

__device__ bool MapNode::set_bottom(MapNode *value)
{
    return set_mapnode_neighbor(&bottom, value);
}


__device__ MapNode *MapNode::get_left() const
{
    return left;
}

__device__ MapNode *MapNode::get_top() const
{
    return top;
}

__device__ MapNode *MapNode::get_right() const
{
    return right;
}

__device__ MapNode *MapNode::get_bottom() const
{
    return bottom;
}


__device__ SpacePoint MapNode::get_coordinates() const
{
    return coordinates;
}

__device__ Polyhedron *MapNode::get_polyhedron() const
{
    return polyhedron;
}

__device__ Face *MapNode::get_face() const
{
    return polyhedron_face;
}


__device__ bool MapNode::does_contain_food() const
{
    return contains_food;
}

__device__ bool MapNode::does_contain_particle() const
{
    return particle != nullptr;
}


__device__ bool MapNode::attach_particle(Particle *p)
{
    static_assert(sizeof(&particle) <= sizeof(unsigned long long *), "I think, I can't safely cast `Particle **` to"
                                                                     "`unsigned long long *`");

    return nullptr == (Particle *)atomicCAS((unsigned long long *)&particle, (unsigned long long)nullptr,
                                            (unsigned long long)p);
}

__device__ Particle *MapNode::get_particle() const
{
    return particle;
}

__device__ void MapNode::detach_particle()
{
    particle = nullptr;
}

__device__ bool MapNode::detach_particle(Particle *p)
{
    static_assert(sizeof(&particle) <= sizeof(unsigned long long *), "I think, I can't safely cast `Particle **` to"
                                                                     "`unsigned long long *`");

    return p == (Particle *)atomicCAS((unsigned long long *)&particle, (unsigned long long)p,
                                      (unsigned long long)nullptr);
}


__host__ __device__ bool operator==(const MapNode &a, const MapNode &b)
{
    return a.coordinates == b.coordinates;
}
