#include "random_generator.cuh"
#include "model_constants.hpp"
#include "fucking_shit.cuh"
#include "model_constants.hpp"

namespace jc = jones_constants;


__device__ void diffuse_trail(MapNode *node)
{
    auto get_trail = [](MapNode *m)
    { return m->trail; };

    ll sum = get_trail(node) + get_trail(node->top->left) +
             get_trail(node->top) + get_trail(node->top->right) +
             get_trail(node->left) + get_trail(node->right) +
             get_trail(node->bottom->left) + get_trail(node->bottom) +
             get_trail(node->bottom->right);

    node->temp_trail = (double) sum / (1 - jc::diffdamp);
}

__device__ void create_particle(MapNode *node)
{
    node->particle = new Particle(node->coordinates, node->polyhedron, node->polyhedron_face);
    /* Please, note that we're using `new` and `delete` operators for allocating and deallocating Particles,
     * and it doesn't matter if we're running on cpu or gpu
     */

    node->contains_particle = true;
}

__device__ void delete_particle(MapNode *node)
{
    delete node->particle;
    /* Please, note that we're using `new` and `delete` operators for allocating and deallocating Particles,
     * and it doesn't matter if we're running on cpu or gpu
     */

    node->contains_particle = false;
}


__device__ ll get_particle_window(MapNode *node, int window_size)
{
    to_be_rewritten; // TODO: Rewrite this function in a more efficient way

    for(int i = 0; i < window_size / 2; ++i)
        node = node->top->left;

    MapNode *row = node;
    ll ans = 0;
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
    if(rand0to1() < jc::death_random_probability)
    {
        delete_particle(node);
    }
}

__device__ void death_test(MapNode *node)
{
    ll particle_window = get_particle_window(node, jc::sw);
    if(jc::smin <= particle_window && particle_window <= jc::smax)
    {/* if in survival range, then stay alive */}
    else
        delete_particle(node);
}

__device__ void division_test(MapNode *node)
{
    ll particle_window = get_particle_window(node, jc::gw);
    if(jc::gmin <= particle_window && particle_window <= jc::gmax)
    {
        if(rand0to1() <= jc::division_probability)
        {
            MapNode *row = node->top->left;
            for(ll i = 0; i < 3; ++i)
            {
                MapNode *cur = row;
                for(ll j = 0; j < 3; ++j)
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
