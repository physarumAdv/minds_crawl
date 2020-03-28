#include <iostream>

#include "MapPoint.hpp"
#include "Polyhedron.cuh"
#include "Particle.hpp"
#include "fucking_shit.cuh"
#include "model_constants.hpp"
#include "random_generator.cuh"
namespace jc = jones_constants;
typedef long long ll;

#define force_one_threaded_kernel() if(threadIdx.x || threadIdx.y || threadIdx.z || \
        blockIdx.x || blockIdx.y || blockIdx.z) return


const ll cuda_block_size = 256;


__global__ void init_food(...)
{
    force_one_threaded_kernel();

    // <initialization here>
}

__global__ void init_polyhedron(Polyhedron *polyhedron, ...)
{
    force_one_threaded_kernel();

    /* WARNING!!! As you can see, we're creating a new `Polyhedron` object
     * and _copying_ it to `*polyhedron`, not assigning the pointer. This
     * is done in purpose, to make it possible to copy `*polyhedron` back to
     * host code.
     */
    *polyhedron = *(new Polyhedron(...));
}

__global__ void run_iteration(const Polyhedron *polyhedron, const ll *iteration_number)
{
    ll i = blockIdx.x * blockDim.x + threadIdx.x;

    if(jc::projectnutrients && *iteration_number >= jc::startprojecttime)
        // Projecting food:
        polyhedron->points[i].trail += polyhedron->points[i].food;

    // Diffuses trail in current point
    diffuse_trail(polyhedron, i);
    polyhedron->points[i].trail = polyhedron->points[i].temp_trail;

    if(polyhedron->points[i].contains_particle)
    {
        do_motor_behaviours(polyhedron->points, i);
        do_sensory_behaviours(polyhedron->points, i);

        if(jc::do_random_death_test && jc::death_random_probability > 0 &&
                *iteration_number > jc::startprojecttime)
            random_death_test(&polyhedron->points[i]);
        if(*iteration_number % jc::death_frequency_test == 0)
            death_test(&polyhedron->points[i]);
        if(*iteration_number % jc::division_frequency_test == 0)
            division_test(&polyhedron->points[i]);
    }
}

__host__ int main()
{
    // Initializing cuRAND:
    init_rand<<<1, 1>>>(time(nullptr));

    Polyhedron *polyhedron;
    cudaMallocManaged((void **)&polyhedron, sizeof(Polyhedron));
    init_polyhedron<<<1, 1>>>(polyhedron);
    
    // <Precalculations (cos, sin, ...) here>
    init_food<<<1, 1>>>(...);
    
    ll *iteration_number;
    cudaMallocManaged((void **)&iteration_number, sizeof(ll));

    const ll cuda_grid_size = (polyhedron->get_n_of_points() + cuda_block_size - 1) /
            cuda_block_size;
    for(*iteration_number = 0; ; ++*iteration_number)
    {
        run_iteration<<<cuda_grid_size, cuda_block_size>>>(polyhedron, iteration_number);
        // <redrawing here>
    }
}
