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


/**
 * Initializes food on the Polyhedron's surface
 *
 * This function isn't implemented yet, neither it's ready to be implemented, so the description stays empty for now
 */
__global__ void init_food(...)
{
    stop_all_threads_except_first;

    // <initialization here>
}

/**
 * Initializes the simulation's objects (simulation map, polyhedron, probably something else)
 *
 * This function isn't implemented yet, neither it's ready to be implemented, so the description stays empty for now
 */
__global__ void init_simulation_objects(SimulationMap *simulation_map, Polyhedron *polyhedron, ...)
{
    stop_all_threads_except_first;

    // <initialization here>
}

/**
 * Runs an iteration of the simulation (not self-sufficient, see the warning)
 *
 * Runs an iteration of the simulation on cuda device if compiled for gpu. Projects food, diffuses trail,
 * moves particles, runs division/death tests and so on
 *
 * @param simulation_map The simulation map to run iteration on
 * @param iteration_number The number of current iteration, used to run division/death tests with needed frequencies
 *      (recent are declared in the `jones_constants` namespace)
 *
 * @note Run the function with `<<<gridDim, blockDim>>>` such that total number of runned threads
 *      is greater or equal to `simulation_map->get_n_of_nodes()`
 *
 * @warning This function is not self-sufficient. After running it you also need to run `iteration_post_triggers`
 */
__global__ void run_iteration(const SimulationMap *simulation_map, const ll *const iteration_number)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= simulation_map->get_n_of_nodes())
        return;
    MapNode *self = &simulation_map->nodes[i];

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

/**
 * Runs some operations which must be performed after running the simulation iteration before running the next one
 *
 * Applies trail changes (sets `trail`s to `temp_trail`s) and increases `iteration_number` by 1
 *
 * @param simulation_map The simulation map to run iteration on
 * @param iteration_number The number of current iteration, used to run division/death tests with needed frequencies
 *      (recent are declared in the `jones_constants` namespace)
 *
 * @note Run the function with `<<<gridDim, blockDim>>>` such that total number of runned threads
 *      is greater or equal to `simulation_map->get_n_of_nodes()`
 */
__global__ void iteration_post_triggers(const SimulationMap *simulation_map, ll *const iteration_number)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= simulation_map->get_n_of_nodes())
        return;
    MapNode *self = &simulation_map->nodes[i];

    self->trail = self->temp_trail;

    stop_all_threads_except_first;
    ++*iteration_number;
}


/**
 * Sets a device-allocated variable to a given value from host code
 *
 * Simply `cudaMemcpy`s the given value to the given pointer
 *
 * @tparam T Type of the value being copied
 *
 * @param destination Device memory pointer to copy value to
 * @param value The value to be copied
 *
 * @see get_cuda_variable_value
 */
template<class T>
__host__ inline void set_cuda_variable_value(T *destination, T value)
{
    cudaMemcpy(destination, &value, sizeof(T), cudaMemcpyHostToDevice);
}


/**
 * Returns a value from the given pointer to device memory
 *
 * @tparam T Type of a value being copied
 *
 * @param source Device-allocated pointer to copy from
 *
 * @returns The value from device memory
 *
 * @see set_cuda_variable_value
 */
template<class T>
__host__ inline T get_cuda_variable_value(T *source)
{
    T ans;
    cudaMemcpy(&ans, source, sizeof(T), cudaMemcpyDeviceToHost);
    return ans;
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
    set_cuda_variable_value(iteration_number, 0LL);

    // Obtaining `n_of_nodes`:
    ll *_temporary;
    cudaMalloc((void **) &_temporary, sizeof(ll));
    simulation_map->get_n_of_nodes(_temporary);
    const ll n_of_nodes = get_cuda_variable_value(_temporary);
    cudaFree(_temporary);

    const ll cuda_grid_size = (n_of_nodes + cuda_block_size - 1) / cuda_block_size;
    for(;;)
    {
        run_iteration<<<cuda_grid_size, cuda_block_size>>>(simulation_map, iteration_number);
        iteration_post_triggers<<<cuda_grid_size, cuda_block_size>>>(simulation_map, iteration_number);
        // <redrawing here>
    }
}
