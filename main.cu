#include <iostream>

#include "MapPoint.hpp"
#include "SimulationMap.cuh"
#include "Particle.cuh"
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

__global__ void init_polyhedron(SimulationMap *polyhedron, ...)
{
    force_one_threaded_kernel();

    /* WARNING!!! As you can see, we're creating a new `SimulationMap` object
     * and _copying_ it to `*polyhedron`, not assigning the pointer. This
     * is done in purpose, to make it possible to copy `*polyhedron` back to
     * host code.
     */
    *polyhedron = *(new SimulationMap(...));
}

__global__ void run_iteration(const SimulationMap *polyhedron, ll *iteration_number)
{
    MapPoint *self = &polyhedron->points[blockIdx.x * blockDim.x + threadIdx.x];

    if(jc::projectnutrients && *iteration_number >= jc::startprojecttime)
        // Projecting food:
        self->trail += self->food;

    // Diffuses trail in current point
    diffuse_trail(self);

    if(self->contains_particle)
    {
        do_motor_behaviours(polyhedron, self);
        do_sensory_behaviours(polyhedron, self);

        if(jc::do_random_death_test && jc::death_random_probability > 0 &&
           *iteration_number > jc::startprojecttime)
            random_death_test(self);
        if(*iteration_number % jc::death_frequency_test == 0)
            death_test(self);
        if(*iteration_number % jc::division_frequency_test == 0)
            division_test(self);
    }

    ++*iteration_number;
}

__global__ void iteration_post_triggers(const SimulationMap *polyhedron)
{
    MapPoint *self = &polyhedron->points[blockIdx.x * blockDim.x + threadIdx.x];

    self->trail = self->temp_trail;
}

__host__ int main()
{
    // Initializing cuRAND:
    init_rand<<<1, 1>>>(time(nullptr));

    SimulationMap *polyhedron;
    cudaMallocManaged((void **) &polyhedron, sizeof(SimulationMap));
    init_polyhedron<<<1, 1>>>(polyhedron);

    // <Precalculations (like cos, sin, ...) here>
    init_food<<<1, 1>>>(...);

    ll *iteration_number;
    cudaMallocManaged((void **) &iteration_number, sizeof(ll));

    const ll cuda_grid_size = (polyhedron->get_n_of_points() + cuda_block_size - 1) /
                              cuda_block_size;
    for(*iteration_number = 0;; /* iteration_number is updated inside run_iteration,
                                  * because we're going to run this as a stream/graph later
                                  * and don't want cpu to do anything between runs within a group */)
    {
        run_iteration<<<cuda_grid_size, cuda_block_size>>>(polyhedron, iteration_number);
        iteration_post_triggers<<<cuda_grid_size, cuda_block_size>>>(polyhedron);
        // <redrawing here>
    }
}
