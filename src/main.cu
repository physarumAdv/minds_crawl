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

#define RUN_ITERATION_SET_SELF(self) { \
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
            if(i >= simulation_map->get_n_of_nodes()) \
            return; \
            self = &simulation_map->nodes[i]; \
        }

typedef void (*RunIterationFunc)(SimulationMap *, int *);


const int cuda_block_size = 256;


/**
 * Initializes the simulation's objects (simulation map, polyhedron, probably something else)
 *
 * This function isn't implemented yet, neither it's ready to be implemented, so the description stays empty for now
 */
__global__ void init_simulation_objects(SimulationMap *simulation_map, Polyhedron *polyhedron, ...)
{
    STOP_ALL_THREADS_EXCEPT_FIRST;

    // <initialization here>
}

/**
 * Initializes food on the `SimulationMap`
 *
 * This function isn't implemented yet, neither it's ready to be implemented, so the description stays empty for now
 */
__global__ void init_environment(...)
{
    STOP_ALL_THREADS_EXCEPT_FIRST;

    // <initialization here>
}


/**
 * Performs a part of an iteration (not self-sufficient, see note below)
 *
 * Projection: for each node containing food, the amount of trail being added is either `jc::suppressvalue` if
 * there is at least one particle in its 3x3 node window (which is also called node neighborhood), or `jc::projectvalue`
 * if there are no. This trail is added not only to the node containing food, but <b>to its neighborhood either</b>
 *
 * @param simulation_map Simulation map to run iteration on
 * @param iteration_number Serial number of current iteration
 *
 * @see run_iteration_diffuse_trail, run_iteration_process_particles, run_iteration_cleanup
 *
 * @note This function is a part of run_iteration kernels set. Because at some moments while running an iteration we
 *      have to synchronize all the threads (including inside blocks), the "run_iteration" operation is split into
 *      multiple kernel functions. Other functions (hopefully, all of them) are mentioned in the "see" block above
 */
__global__ void run_iteration_project_nutrients(SimulationMap *simulation_map, const int *const iteration_number)
{
    /// Pointer to the `MapNode` being processed by current thread
    MapNode *self;
    RUN_ITERATION_SET_SELF(self)

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
                    atomicAdd(&node->trail, trail_value); // add trail
                    node = node->get_right(); // move to next node in row
                }
            }
        }
    }
}

/**
 * Performs a part of an iteration (not self-sufficient, see note below
 *
 * Diffusion: the diffusion algorithm (developed by Jeff Jones) is pretty simple at first sight. We calculate an average
 * `trail` value in a 3x3 node window around the given one and multiply it by `(1 - jones_constants::diffdamp)`.
 * The new `temp_trail` value in the given node is the value just calculated. This is a natural way to implement the
 * smell spread: on each iteration smell moves more far from the source, but becomes less strong, because
 * `(1 - jones_constants::diffdamp)` < 1
 *
 * @param simulation_map Simulation map to run iteration on
 * @param iteration_number Serial number of current iteration
 *
 * @see run_iteration_project_nutrients, run_iteration_process_particles, run_iteration_cleanup
 *
 * @note This function is a part of run_iteration kernels set. Because at some moments while running an iteration we
 *      have to synchronize all the threads (including inside blocks), the "run_iteration" operation is split into
 *      multiple kernel functions. Other functions (hopefully, all of them) are mentioned in the "see" block above
 */
__global__ void run_iteration_diffuse_trail(SimulationMap *simulation_map, const int *const iteration_number)
{
    MapNode *self;
    RUN_ITERATION_SET_SELF(self);

    auto left = self->get_left(), top = self->get_top(), right = self->get_right(), bottom = self->get_bottom();

    double sum = top->get_left()->trail + top->trail + top->get_right()->trail +
                 left->trail + self->trail + right->trail +
                 bottom->get_left()->trail + bottom->trail + bottom->get_right()->trail;

    self->temp_trail = (1 - jc::diffdamp) * (sum / 9.0);
}

/**
 * Performs a part of an iteration
 *
 * Processing particles: For each node containing particles, the following operations are performed on the particles:
 * motor behaviours, sensory behaviours, division test, random death test, death test (order saved). You can read
 * details about these in the corresponding functions' docs
 *
 * @param simulation_map Simulation map to run iteration on
 * @param iteration_number Serial number of current iteration
 *
 * @see run_iteration_project_nutrients, run_iteration_diffuse_trail, run_iteration_cleanup
 *
 * @note This function is a part of run_iteration kernels set. Because at some moments while running an iteration we
 *      have to synchronize all the threads (including inside blocks), the "run_iteration" operation is split into
 *      multiple kernel functions. Other functions (hopefully, all of them) are mentioned in the "see" block above
 */
__global__ void run_iteration_process_particles(SimulationMap *simulation_map, const int *const iteration_number)
{
    /// Pointer to the `MapNode` being processed by current thread
    MapNode *self;
    RUN_ITERATION_SET_SELF(self)

    if(!self->contains_particle() || !self->get_particle()->capture())
        return;

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

/**
 * Runs some operations which must be performed after running the simulation iteration before running the next one
 *
 * Applies trail changes (sets `trail`s to `temp_trail`s), releases captured particles and increases `iteration_number`
 * by 1
 *
 * @param simulation_map Simulation map to run iteration on
 * @param iteration_number Serial number of current iteration
 *
 * @see run_iteration_project_nutrients, run_iteration_diffuse_trail, run_iteration_process_particles
 *
 * @note This function is a part of run_iteration kernels set. Because at some moments while running an iteration we
 *      have to synchronize all the threads (including inside blocks), the "run_iteration" operation is split into
 *      multiple kernel functions. Other functions (hopefully, all of them) are mentioned in the "see" block above
 */
__global__ void run_iteration_cleanup(SimulationMap *simulation_map, int *const iteration_number)
{
    /// Pointer to the `MapNode` being processed by current thread
    MapNode *self;
    RUN_ITERATION_SET_SELF(self)

    self->trail = self->temp_trail;
    if(self->contains_particle())
        self->get_particle()->release();

    STOP_ALL_THREADS_EXCEPT_FIRST;
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
 * @param value Value to be copied
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
 * @returns Value from device memory
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

    // Initializing simulation objects
    SimulationMap *simulation_map;
    cudaMallocManaged((void **)&simulation_map, sizeof(SimulationMap));
    Polyhedron *polyhedron;
    cudaMalloc((void **)&polyhedron, sizeof(Polyhedron));
    init_simulation_objects<<<1, 1>>>(simulation_map, polyhedron, ...);

    init_environment<<<1, 1>>>(...);

    int *iteration_number;
    cudaMalloc((void **)&iteration_number, sizeof(int));
    set_cuda_variable_value(iteration_number, 0);

    // Obtaining `n_of_nodes`
    int n_of_nodes;
    {
        int *_temporary;
        cudaMalloc((void **)&_temporary, sizeof(int));
        get_n_of_nodes<<<1, 1>>>(simulation_map, _temporary);
        n_of_nodes = get_cuda_variable_value(_temporary);
        cudaFree(_temporary);
    }

    // Obtaining `nodes`
    MapNode *nodes, *nodes_d = simulation_map->nodes;
    cudaMallocHost((void **)&nodes, sizeof(MapNode) * n_of_nodes);


    RunIterationFunc iteration_runners[] = {(RunIterationFunc)run_iteration_project_nutrients,
                                            (RunIterationFunc)run_iteration_diffuse_trail,
                                            (RunIterationFunc)run_iteration_process_particles,
                                            run_iteration_cleanup};


    // Creating cuda stream
    cudaStream_t iterations_stream;
    cudaStreamCreate(&iterations_stream);


    const int cuda_grid_size = (n_of_nodes + cuda_block_size - 1) / cuda_block_size;

    for(;;)
    {
        if(cudaErrorNotReady == cudaStreamQuery(iterations_stream))
            printf("I: Redrawed, but kernel haven't finished yet\n");

        // (implicit synchronization)
        cudaMemcpy((void *)nodes, (void *)nodes_d, sizeof(MapNode) * n_of_nodes, cudaMemcpyDeviceToHost);

        for(RunIterationFunc f : iteration_runners)
        {
            f<<<cuda_grid_size, cuda_block_size, 0, iterations_stream>>>(simulation_map,
                                                                         iteration_number);
        }

        // <redrawing here>
    }
}
