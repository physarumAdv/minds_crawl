#include "random_generator.cuh"
#include "model_constants.hpp"
#include "fucking_shit.cuh"
namespace jc = jones_constants;


__device__ ll get_index(ll x, ll y, ll z, dim3 grid_size)
{
    return x * grid_size.y * grid_size.z + y * grid_size.y + z;
}


__device__ void diffuse_trail(MapPoint *grid, ll x, ll y, ll z, dim3 grid_size)
{
    ll sum, cnt;
    sum = cnt = 0;
    for(int dx = -1; dx <= 1; ++dx)
        for(int dy = -1; dy <= 1; ++dy)
            for(int dz = -1; dz <= 1; ++dz)
                if(x + dx > 0 && x + dx < grid_size.x)
                    if(y + dy > 0 && y + dy < grid_size.y)
                        if(z + dz > 0 && z + dz < grid_size.z)
                        {
                            sum += grid[get_index(x + dx, y + dy, z + dz, grid_size)].trail;
                            ++cnt;
                        }
    grid[get_index(x, y, z, grid_size)].temp_trail = sum / cnt;
}


__device__ void delete_particle(MapPoint *p)
{
/*#ifdef COMPILE_FOR_CPU
    free(p->particle);
#else
    cudaFree(p->particle);
#endif*/
    delete p->particle;
    /* Please, note that we're using `new` and `delete` operators for allocating and deallocating Particles,
     * and it doesn't matter if we're running on cpu or gpu */

    p->contains_particle = false;
}


__device__ void random_death_test(MapPoint *p, double death_probability)
{
    if(rand01() < death_probability)
    {
        delete_particle(p);
    }
}

__device__ ll get_particle_window(MapPoint *grid, ll x, ll y, ll z, dim3 grid_size, ll w)
{
    ll ans = 0;
    for(ll dx = -(w/2); dx <= w/2; ++dx)
        for(ll dy = -(w/2); dy <= w/2; ++dy)
            for(ll dz = -(w/2); dz <= w/2; ++dz)
                ans += grid[get_index(x, y, z, grid_size)].contains_particle;
    return ans;
}

__device__ void death_test(MapPoint *grid, ll x, ll y, ll z, dim3 grid_size)
{
    ll particle_window = get_particle_window(grid, x, y, z, grid_size, jc::sw);
    if(jc::smin <= particle_window && particle_window <= jc::smax)
    {/* if in survival range, then stay alive */}
    else
        delete_particle(&grid[get_index(x, y, z, grid_size)]);
}

__device__ bool division_test(MapPoint *grid, ll x, ll y, ll z, dim3 grid_size)
{
    ll particle_window = get_particle_window(grid, x, y, z, grid_size, jc::gw);
    if(jc::gmin <= particle_window && particle_window <= jc::gmax)
        if(rand01() <= jc::division_probability)
            for(ll dx = -1; dx <= 1; ++dx)
                for(ll dy = -1; dy <= 1; ++dy)
                    for(ll dz = -1; dz <= 1; ++dz)
                    {
                        ll i = get_index(x + dx, y + dy, z + dz, grid_size);
                        if(!grid[i].contains_particle)
                        {
                            create_particle(&grid[i]);
                            return true;
                        }
                    }
    return false;
}
