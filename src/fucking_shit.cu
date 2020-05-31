#include <cstdio>
#include <initializer_list>

#include "fucking_shit.cuh"
#include "random_generator.cuh"
#include "jones_constants.hpp"
#include "Particle.cuh"
#include "jones_constants.hpp"

namespace jc = jones_constants;


[[nodiscard]] __device__ bool create_particle(MapNode *node)
{
    auto p = new Particle(node, rand0to1() * 360);

    if(node->attach_particle(p))
        return true;

    delete p;
    return false;
}

[[nodiscard]] __device__ bool delete_particle(MapNode *node)
{
    Particle *p = node->get_particle();

    if(!node->detach_particle(p))
        return false;

    delete p;
    return true;
}


__device__ void diffuse_trail(MapNode *node)
{
    auto get_trail = [](MapNode *m)
    { return m->trail; };

    auto left = node->get_left(), top = node->get_top(), right = node->get_right(), bottom = node->get_bottom();

    double sum = get_trail(top->get_left()) + get_trail(top) + get_trail(top->get_right()) +
                 get_trail(left) + get_trail(node) + get_trail(right) +
                 get_trail(bottom->get_left()) + get_trail(bottom) + get_trail(bottom->get_right());

    node->temp_trail = (1 - jc::diffdamp) * (sum / 9.0);
}


__device__ int count_particles_in_node_window(MapNode *node, int window_size)
{
    for(int i = 0; i < window_size / 2; ++i)
        node = node->get_top()->get_left();

    MapNode *row = node;
    int ans = 0;
    for(int i = 0; i < window_size; ++i)
    {
        MapNode *cur = row;
        for(int j = 0; j < window_size; ++j)
        {
            if(cur->contains_particle())
                ++ans;
            cur = cur->get_right();
        }
        row = row->get_bottom();
    }

    return ans;
}


__device__ bool random_death_test(MapNode *node)
{
    if(rand0to1() < jc::random_death_probability)
    {
        if(!delete_particle(node))
        {
            // This is what called "undefined behaviour" in the docs :)
            printf("%s:%d - this line should never be reached", __FILE__, __LINE__);
            return false; // Particle was not removed
        }
        return true; // Particle was removed
    }
    return false; // Particle was not removed
}

__device__ bool death_test(MapNode *node)
{
    int particles_in_window = count_particles_in_node_window(node, jc::sw);
    if(jc::smin <= particles_in_window && particles_in_window <= jc::smax)
    {/* if in survival range, then stay alive */}
    else
    {
        if(!delete_particle(node))
        {
            // This is what called "undefined behaviour" in the docs :)
            printf("%s:%d - this line should never be reached", __FILE__, __LINE__ - 1);
            return false; // Particle was not removed
        }
        return true; // Particle was removed
    }
    return false; // Particle was not removed
}

__device__ void division_test(MapNode *node)
{
    int particle_window = count_particles_in_node_window(node, jc::gw);
    if(jc::gmin <= particle_window && particle_window <= jc::gmax)
    {
        if(rand0to1() <= jc::division_probability)
        {
            MapNode *row = node->get_top()->get_left();
            for(int i = 0; i < 3; ++i)
            {
                MapNode *cur = row;
                for(int j = 0; j < 3; ++j)
                {
                    if(create_particle(cur)) // If new particle was successfully created
                        return;
                    cur = cur->get_right();
                }
                row = row->get_bottom();
            }
        }
    }
}


__device__ MapNode *find_nearest_mapnode_greedy(const SpacePoint dest, MapNode *const start)
{
    MapNode *current = start;
    double current_dist = get_distance(dest, current->coordinates);
    while(true)
    {
        bool found_better = false;
        for(auto next : {current->get_left(), current->get_top(), current->get_right(), current->get_bottom()})
        {
            double next_dist = get_distance(dest, next->coordinates);
            if(next_dist < current_dist)
            {
                current = next;
                current_dist = next_dist;
                found_better = true;
                break;
            }
        }
        if(!found_better)
            break;
    }
    return current;
}

__device__ MapNode *find_nearest_mapnode(const Polyhedron *const polyhedron, SpacePoint dest, MapNode *start)
{
    int dest_face = polyhedron->find_face_id_by_point(dest);

    if(start != nullptr)
    {
        MapNode *ans = find_nearest_mapnode_greedy(dest, start);
        if(ans->polyhedron_face_id == dest_face)
            return ans;
    }

    return find_nearest_mapnode_greedy(dest, polyhedron->faces[polyhedron->find_face_id_by_point(dest)].node);
}


// `address` CANNOT be pointer to const, because we are trying to edit memory by it's address
__device__ bool atomicCAS(bool *const address, const bool compare, const bool val)
{
    auto addr = (unsigned long long)address;
    unsigned pos = addr & 3;  // byte position within the int
    auto *int_addr = (unsigned *)(addr - pos);  // int-aligned address
    unsigned old = *int_addr, assumed, ival;

    bool current_value;

    do
    {
        current_value = (bool)(old & ((0xFFU) << (8 * pos)));

        if(current_value != compare) // If we expected that bool to be different, then
            break; // stop trying to update it and just return it's current value

        assumed = old;
        if(val)
            ival = old | (1U << (8 * pos));
        else
            ival = old & (~((0xFFU) << (8 * pos)));
        old = atomicCAS(int_addr, assumed, ival);
    } while(assumed != old);

    return current_value;
}

