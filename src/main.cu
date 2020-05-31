#include <iostream>

#include "MapNode.cuh"
#include "SimulationMap.cuh"
#include "Particle.cuh"
#include "fucking_shit.cuh"
#include "jones_constants.hpp"
#include "random_generator.cuh"
#include "Polyhedron.cuh"
#include "common.cuh"

namespace jc = jones_constants;

#define run_iteration_set_self_or_return(self) { \
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
            if(i >= simulation_map->get_n_of_nodes()) \
            return; \
            self = &simulation_map->nodes[i]; \
        }


const int cuda_block_size = 256;


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
 * Initializes food on the `SimulationMap`
 *
 * This function isn't implemented yet, neither it's ready to be implemented, so the description stays empty for now
 */
__global__ void init_food(...)
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
 * @param iteration_number The current iteration number, used to run division/death tests with needed frequencies
 *      (recent are declared in the `jones_constants` namespace)
 *
 * @note Run the function with `<<<gridDim, blockDim>>>` such that total number of runned threads
 *      is greater or equal to `simulation_map->get_n_of_nodes()` and all of four `gridDim.y`, `gridDim.z`,
 *      `blockDim.y`, `blockDim.z` are equal to 1
 *
 * @warning This function is not self-sufficient. After running it you also need to run `iteration_post_triggers`
 *      with the same arguments
 */
__global__ void run_iteration(const SimulationMap *simulation_map, const int *const iteration_number)
{
    /// Pointer to the `MapNode` being processed by current thread
    MapNode *self;
    run_iteration_set_self_or_return(self)

    if(jc::projectnutrients && *iteration_number >= jc::startprojecttime)
    {
        if(self->contains_food)
        {
            double trail_value;
            if(count_particles_in_node_window(self, 3) > 0)
                trail_value = jc::suppressvalue;
            else
                trail_value = jc::projectvalue;

            // Add trail to 3x3 node window
            MapNode *left = self->get_left();
            for(MapNode *node : {left->get_top(), left,
                                 left->get_bottom()}) // for each leading node of rows of 3x3 square
            {
                for(int i = 0; i < 3; ++i)
                {
                    node->trail += trail_value; // add trail
                    node = node->get_right(); // move to next node in row
                }
            }
        }
    }

    // Diffuses trail in the current node
    diffuse_trail(self);

    if(self->contains_particle())
    {
        self->get_particle()->do_motor_behaviours();
        self->get_particle()->do_sensory_behaviours();

        if(*iteration_number % jc::division_test_frequency == 0)
            division_test(self);
        if(jc::do_random_death_test && jc::random_death_probability > 0 &&
           *iteration_number > jc::startprojecttime)
            if(random_death_test(self)) // If the particle died
                return;
        if(*iteration_number % jc::death_test_frequency == 0)
            if(death_test(self)) // If the particle died
                return;
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
__global__ void iteration_post_triggers(const SimulationMap *simulation_map, int *const iteration_number)
{
    /// Pointer to the `MapNode` being processed by current thread
    MapNode *self;
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= simulation_map->get_n_of_nodes())
            return;
        self = &simulation_map->nodes[i];
    }

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
 * @param source Device memory pointer to copy from
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
    cudaMallocManaged((void **)&simulation_map, sizeof(SimulationMap));
    Polyhedron *polyhedron;
    cudaMallocManaged((void **)&polyhedron, sizeof(Polyhedron));
    init_simulation_objects<<<1, 1>>>(simulation_map, polyhedron);

    init_food<<<1, 1>>>(...);

    int *iteration_number;
    cudaMalloc((void **)&iteration_number, sizeof(int));
    set_cuda_variable_value(iteration_number, 0);

    // Obtaining `n_of_nodes`:
    int *_temporary;
    cudaMalloc((void **)&_temporary, sizeof(int));
    get_n_of_nodes<<<1, 1>>>(simulation_map, _temporary);
    const int n_of_nodes = get_cuda_variable_value(_temporary);
    cudaFree(_temporary);

    const int cuda_grid_size = (n_of_nodes + cuda_block_size - 1) / cuda_block_size;
    for(;;)
    {
        run_iteration<<<cuda_grid_size, cuda_block_size>>>(simulation_map, iteration_number);
        iteration_post_triggers<<<cuda_grid_size, cuda_block_size>>>(simulation_map, iteration_number);
        // <redrawing here>
    }
}
