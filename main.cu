#define CUDA


#include <iostream>

#include "MapPoint.h"
#include "Polyhedron.h"
#include "Particle.h"
#include "grid_processing.h"
#include "fucking_shit.h"
#include "model_constants.h"
#include "random_generator.h"
namespace jc = jones_constants;
typedef long long ll;


const ll block_size_by_dim = 8;


__global__ void init_food(...)
{
    if(threadIdx.x || threadIdx.y || threadIdx.z || \
            blockIdx.x || blockIdx.y || blockIdx.z)
        return;
    // <initialization here>
}

__global__ void init_particles(...)
{
    if(threadIdx.x || threadIdx.y || threadIdx.z || \
            blockIdx.x || blockIdx.y || blockIdx.z)
        return;
    // <initialization here>
}

__global__ void run_iteration(MapPoint *grid, dim3 grid_size, Polyhedron *polyhedron, ll *iteration_number)
{
    ll x = blockIdx.x * blockDim.x + threadIdx.x;
    ll y = blockIdx.y * blockDim.y + threadIdx.y;
    ll z = blockIdx.z * blockDim.z + threadIdx.z;

    ll mx = grid_size.x, my = grid_size.y, mz = grid_size.z;

    // The grid index of this MapPoint
    ll i = x * my * mz + y * mz + z;

    if(it's time to project food)
        grid[i].trail += food[i];
    if(it's time to diffuse trail)
        diffuse_trail(grid, x, y, z, grid_size); // Diffuses trail in currect point
    if(grid[i].contains_particle)
    {
        do_motor_behaviours(grid, grid_size, mx, my, mz);
        do_sensory_behaviours(grid, grid_size, mx, my, mz);

        if(jc::do_random_death_test && jc::death_random_probability > 0 &&
                *iteration_number > jc::startprojecttime)
            random_death_test(&grid[i],
                    jc::death_random_probability);
        if(max(x, max(y, z)) > jc::divisionborder &&
                min(mx - x, min(my - y, mz - z)) > jc::divisionborder)
        {
            if(*iteration_number % jc::death_frequency_test == 0)
                death_test(grid, x, y, z, grid_size);
            if(*iteration_number % jc::division_frequency_test == 0)
                division_test(grid, x, y, z, grid_size);
        }
    }
}

__host__ int main()
{
    // Initializing cuRAND:
    init_rand<<<1, 1>>>(time(nullptr));

    Polyhedron *polyhedron;
    cudaMallocManaged((void **)&polyhedron, sizeof(Polyhedron));
    // <Create polyhedron here>

    MapPoint *grid;
    ll mx = polyhedron->get_max_x(), my = polyhedron->get_max_y(),
            mz = polyhedron->get_max_z();
    dim3 grid_size(polyhedron->get_max_x(), polyhedron->get_max_y(),
                   polyhedron->get_max_z());
    cudaMallocManaged((void **)&grid, grid_size.x * grid_size.y *
            grid_size.z * sizeof(MapPoint));
    
    // <Precalculations (cos, sin, ...) here>
    init_food<<<1, 1>>>(...);
    init_particles<<<1, 1>>>(...);
    
    ll *iteration_number;
    cudaMallocManaged((void **)&iteration_number, sizeof(int));

    dim3 cuda_block_size(block_size_by_dim, block_size_by_dim, block_size_by_dim);
    dim3 cuda_grid_size((mx + block_size_by_dim - 1) / block_size_by_dim,
            (my + block_size_by_dim - 1) / block_size_by_dim,
            (mz + block_size_by_dim - 1) / block_size_by_dim);
    for(*iteration_number = 0; ; ++*iteration_number)
    {
        run_iteration<<<cuda_grid_size, cuda_block_size>>>(grid, grid_size, polyhedron, iteration_number);
        // <redrawing here>
    }
}
