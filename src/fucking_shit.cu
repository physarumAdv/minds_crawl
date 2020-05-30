#include <initializer_list>

#include "fucking_shit.cuh"
#include "random_generator.cuh"
#include "jones_constants.hpp"
#include "Particle.cuh"
#include "jones_constants.hpp"

namespace jc = jones_constants;


__device__ void create_particle(MapNode *node)
{
    node->particle = new Particle(node, node->coordinates, rand0to1() * 360);

    node->contains_particle = true;
}

__device__ void delete_particle(MapNode *node)
{
    delete node->particle;

    node->contains_particle = false;
}


__device__ void diffuse_trail(MapNode *node)
{
    auto get_trail = [](MapNode *m)
    { return m->trail; };

    double sum = get_trail(node) + get_trail(node->top->left) +
                 get_trail(node->top) + get_trail(node->top->right) +
                 get_trail(node->left) + get_trail(node->right) +
                 get_trail(node->bottom->left) + get_trail(node->bottom) +
                 get_trail(node->bottom->right);

    node->temp_trail = (1 - jc::diffdamp) * (sum / 9.0);
}


__device__ int count_particles_in_node_window(MapNode *node, int window_size)
{
    for(int i = 0; i < window_size / 2; ++i)
        node = node->top->left;

    MapNode *row = node;
    int ans = 0;
    for(int i = 0; i < window_size; ++i)
    {
        MapNode *cur = row;
        for(int j = 0; j < window_size; ++j)
        {
            ans += cur->contains_particle;
            cur = cur->right;
        }
        row = row->bottom;
    }

    return ans;
}


__device__ void random_death_test(MapNode *node)
{
    if(rand0to1() < jc::random_death_probability)
    {
        delete_particle(node);
    }
}

__device__ void death_test(MapNode *node)
{
    int particles_in_window = count_particles_in_node_window(node, jc::sw);
    if(jc::smin <= particles_in_window && particles_in_window <= jc::smax)
    {/* if in survival range, then stay alive */}
    else
        delete_particle(node);
}

__device__ void division_test(MapNode *node)
{
    int particle_window = count_particles_in_node_window(node, jc::gw);
    if(jc::gmin <= particle_window && particle_window <= jc::gmax)
    {
        if(rand0to1() <= jc::division_probability)
        {
            MapNode *row = node->top->left;
            for(int i = 0; i < 3; ++i)
            {
                MapNode *cur = row;
                for(int j = 0; j < 3; ++j)
                {
                    if(!cur->contains_particle)
                    {
                        create_particle(cur);
                        return;
                    }
                    cur = cur->right;
                }
                row = row->bottom;
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
        for(auto next : {current->left, current->top, current->right, current->bottom})
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

