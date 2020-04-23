#include <iostream>

#include "MapNode.cuh"
#include "SimulationMap.cuh"
#include "Particle.cuh"
#include "fucking_shit.cuh"
#include "model_constants.hpp"
#include "random_generator.cuh"

namespace jc = jones_constants;
typedef long long ll;

#define stop_all_threads_except_first if(threadIdx.x || threadIdx.y || threadIdx.z || \
        blockIdx.x || blockIdx.y || blockIdx.z) return


const ll cuda_block_size = 256;


__global__ void init_food(...)
{
    stop_all_threads_except_first;

    // <initialization here>
}

__global__ void init_simulation_objects(SimulationMap *simulation_map, Polyhedron *polyhedron, ...)
{
    stop_all_threads_except_first;

    /* WARNING!!! As you can see, we're creating a new `SimulationMap` object
     * and _copying_ it to `*simulation_map`, not assigning the pointer. This
     * is done in purpose, to make it possible to copy `*simulation_map` back to
     * host code.
     */
    *simulation_map = *(new SimulationMap(...));
}

__global__ void run_iteration(const SimulationMap *simulation_map, ll *iteration_number)
{
    MapNode *self = &simulation_map->points[blockIdx.x * blockDim.x + threadIdx.x];

    if(jc::projectnutrients && *iteration_number >= jc::startprojecttime)
        // Projecting food:
        self->trail += self->food;

    // Diffuses trail in the current node
    diffuse_trail(self);

    if(self->contains_particle)
    {
        do_motor_behaviours(simulation_map, self);
        do_sensory_behaviours(simulation_map, self);

        if(jc::do_random_death_test && jc::death_random_probability > 0 &&
           *iteration_number > jc::startprojecttime)
            random_death_test(self);
        if(*iteration_number % jc::death_frequency_test == 0)
            death_test(self);
        if(*iteration_number % jc::division_frequency_test == 0)
            division_test(self);
    }
}

__global__ void iteration_post_triggers(const SimulationMap *simulation_map, int *iteration_number)
{
    MapNode *self = &simulation_map->points[blockIdx.x * blockDim.x + threadIdx.x];
    self->trail = self->temp_trail;

    stop_all_threads_except_first;
    ++*iteration_number;
}

template<class T>
__global__ inline void set_variable_to_value(T *variable, T value)
{
    stop_all_threads_except_first;
    *variable = value;
}

__host__ int main()
{
    // Initializing cuRAND:
    init_rand<<<1, 1>>>(time(nullptr));

    SimulationMap *simulation_map;
    cudaMallocManaged((void **) &simulation_map, sizeof(SimulationMap));
    Polyhedron *polyhedron;
    cudaMallocManaged((void **) &polyhedron, sizeof(Polyhedron));
    init_simulation_objects<<<1, 1>>>(simulation_map, polyhedron);

    init_food<<<1, 1>>>(...);

    ll *iteration_number;
    cudaMalloc((void **) &iteration_number, sizeof(ll));
    set_variable_to_value<<<1, 1>>>(iteration_number, 0LL);

    const ll cuda_grid_size = (simulation_map->get_n_of_points() + cuda_block_size - 1) /
                              cuda_block_size;
    for(;; /* iteration_number is updated inside `run_iteration`,
            * because we're going to run this all as a cuda stream/graph later
            * and don't want cpu to do anything between runs within a group */)
    {
        run_iteration<<<cuda_grid_size, cuda_block_size>>>(simulation_map, iteration_number);
        iteration_post_triggers<<<cuda_grid_size, cuda_block_size>>>(simulation_map);
        // <redrawing here>
    }
}
