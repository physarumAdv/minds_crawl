#include <iostream>
typedef long long ll;


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

__global__ void run_iteration(MapPoint *grid, Polyhedron *polyhedron, int *iteration_number)
{
    ll x = blockIdx.x * blockDim.x + threadIdx.x;
    ll y = blockIdx.y * blockDim.y + threadIdx.y;
    ll z = blockIdx.z * blockDim.z + threadIdx.z;

    // The grid index of this MapPoint
    ll i = x * polyhedron.y * polyhedron.z + y * polyhedron.z + z;

    if(it's time to diffuse trail)
        diffuse_trail(x, y, z); // Diffuses trail in currect point
    if(grid[i].contains_particle)
    {
        Particle p = grid[i].particle;
        p.do_motor_behaviours();
        p.do_sensor_behaviours();

        // <random_death_test here>
        // <death_test here>
        // <division_test here>
    }
}

__host__ int main()
{
    Polyhedron *polyhedron;
    cudaMallocManaged((void**)&polyhedron, sizeof(Polyhedron));
    // <Create polyhedron here>

    MapPoint *grid;
    cudaMallocManaged((void **)&grid, grid_size * sizeof(MapPoint));
    
    // <Precalculations (cos, sin, ...) here>
    init_food<<<1, 1>>>(...);
    init_particles<<<1, 1>>>(...);
    
    int *iteration_number;
    cudaMallocManaged((void**)&iteration_number, sizeof(int));
    for(*iteration_number = 0; ; ++*iteration_number)
    {
        // <running an iteration here>
        // <redrawing here>
    }
}
