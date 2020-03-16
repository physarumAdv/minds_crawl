#include <iostream>

#include "MapPoint.h"
#include "Polyhedron.h"
#include "Particle.h"
#include "grid_processing.h"
#include "fucking_shit.h"
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

__global__ void run_iteration(MapPoint *grid, int mx, int my, int mz, Polyhedron *polyhedron, int *iteration_number)
{
    ll x = blockIdx.x * blockDim.x + threadIdx.x;
    ll y = blockIdx.y * blockDim.y + threadIdx.y;
    ll z = blockIdx.z * blockDim.z + threadIdx.z;

    // The grid index of this MapPoint
    ll i = x * my * mz + y * mz + z;

    if(it's time to project food)
        grid[i].trail += food[i];
    if(it's time to diffuse trail)
        diffuse_trail(grid, x, y, z, mx, my, mz); // Diffuses trail in currect point
    if(grid[i].contains_particle)
    {
        do_motor_behaviours(grid, x, y, z, mx, my, mz);
        do_sensory_behaviours(grid, x, y, z, mx, my, mz);

        // <random_death_test here>
        // <death_test here>
        // <division_test here>
    }
}

__host__ int main()
{
    Polyhedron *polyhedron;
    cudaMallocManaged((void **)&polyhedron, sizeof(Polyhedron));
    // <Create polyhedron here>

    ll grid_size = polyhedron->max_x * polyhedron->max_y * polyhedron->max_z;
    MapPoint *grid;
    cudaMallocManaged((void **)&grid, grid_size * sizeof(MapPoint));
    
    // <Precalculations (cos, sin, ...) here>
    init_food<<<1, 1>>>(...);
    init_particles<<<1, 1>>>(...);
    
    ll *iteration_number;
    cudaMallocManaged((void **)&iteration_number, sizeof(int));

    dim3 block_size(block_size_by_dim, block_size_by_dim, block_size_by_dim);
    int mx = polyhedron->get_max_x(), my = polyhedron->get_max_y(),
             mz = polyhedron->get_max_z();
    dim3 grid_size((mx + block_size_by_dim - 1) / block_size_by_dim,
            (my + block_size_by_dim - 1) / block_size_by_dim,
            (mz + block_size_by_dim - 1) / block_size_by_dim);
    for(*iteration_number = 0; ; ++*iteration_number)
    {
        run_iteration<<<grid_size, block_size>>>(grid, mx, my, mz, polyhedron, iteration_number);
        // <redrawing here>
    }
}
