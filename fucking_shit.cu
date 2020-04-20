#include "random_generator.cuh"
#include "model_constants.hpp"
#include "fucking_shit.cuh"

namespace jc = jones_constants;


__device__ void diffuse_trail(MapPoint *m)
{
    auto get_trail = [](MapPoint *m)
    { return m->trail; };

    ll sum = get_trail(m) + get_trail(m->top->left) +
             get_trail(m->top) + get_trail(m->top->right) +
             get_trail(m->left) + get_trail(m->right) +
             get_trail(m->bottom->left) + get_trail(m->bottom) +
             get_trail(m->bottom->right);

    // 9.0 is the count of `MapPoint`s in a window (implementation-level constant)
    m->temp_trail = (double) sum / 9.0;
}

void create_particle(MapPoint *p)
{
    // TODO: this may cause a memoery leak, if it containts a particle already. Think about it and probably add a check
    p->particle = new Particle;
    /* Please, note that we're using `new` and `delete` operators for allocating and deallocating Particles,
     * and it doesn't matter if we're running on cpu or gpu
     */

    p->contains_particle = true;
}

__device__ void delete_particle(MapPoint *p)
{
    delete p->particle;
    /* Please, note that we're using `new` and `delete` operators for allocating and deallocating Particles,
     * and it doesn't matter if we're running on cpu or gpu
     */

    p->contains_particle = false;
}


__device__ ll get_particle_window(MapPoint *m, int window_size)
{
    to_be_rewritten; // TODO: Rewrite this function in a more efficient way

    for(int i = 0; i < window_size / 2; ++i)
        m = m->top->left;

    MapPoint *row = m;
    ll ans = 0;
    for(int i = 0; i < window_size; ++i)
    {
        MapPoint *cur = row;
        for(int j = 0; j < window_size; ++j)
        {
            ans += cur->contains_particle;
            cur = cur->right;
        }
        row = row->bottom;
    }

    return ans;
}


__device__ void random_death_test(MapPoint *p)
{
    if(rand0to1() < jc::death_random_probability)
    {
        delete_particle(p);
    }
}

__device__ void death_test(MapPoint *m)
{
    ll particle_window = get_particle_window(m, jc::sw);
    if(jc::smin <= particle_window && particle_window <= jc::smax)
    {/* if in survival range, then stay alive */}
    else
        delete_particle(m);
}

__device__ void division_test(MapPoint *m)
{
    ll particle_window = get_particle_window(m, jc::gw);
    if(jc::gmin <= particle_window && particle_window <= jc::gmax)
    {
        if(rand0to1() <= jc::division_probability)
        {
            MapPoint *row = m->top->left;
            for(ll i = 0; i < 3; ++i)
            {
                MapPoint *cur = row;
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
