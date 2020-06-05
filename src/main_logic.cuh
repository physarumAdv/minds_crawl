#ifndef MIND_S_CRAWL_MAIN_LOGIC_CUH
#define MIND_S_CRAWL_MAIN_LOGIC_CUH


#include <initializer_list>

#include "SimulationMap.cuh"
#include "Polyhedron.cuh"
#include "MapNode.cuh"
#include "Particle.cuh"
#include "fucking_shit.cuh"
#include "jones_constants.hpp"

namespace jc = jones_constants;

#define RUN_ITERATION_SET_SELF(self, i) { \
            if(i >= simulation_map->get_n_of_nodes()) \
                return; \
            self = &simulation_map->nodes[i]; \
        }

typedef void (*RunIterationFunc)(SimulationMap *, int *);


// TODO: Add the opposite (destructing) function
/**
 * Initializes the simulation's objects (simulation map, polyhedron, probably something else)
 *
 * This function isn't implemented yet, neither it's ready to be implemented, so the description stays empty for now
 */
__device__ inline void init_simulation_objects(SimulationMap *const simulation_map, Polyhedron *const polyhedron, ...)
{
    // <initialization here>
}

/**
 * Initializes food on the `SimulationMap`
 *
 * This function isn't implemented yet, neither it's ready to be implemented, so the description stays empty for now
 */
__device__ inline void init_environment(...)
{
    // <initialization here>
}


/**
 * Performs a part of an iteration <b>for one node</b> (not self-sufficient, see note below)
 *
 * Projection: for each node containing food, the amount of trail being added is either `jc::suppressvalue` if
 * there is at least one particle in its 3x3 node window (which is also called node neighborhood), or `jc::projectvalue`
 * if there are no. This trail is added not only to the node containing food, but <b>to its neighborhood either</b>
 *
 * @param simulation_map Simulation map to run iteration on
 * @param iteration_number Serial number of current iteration
 * @param node_index Index of `SimulationMap`'s node to run iteration on. Can be out of bounds
 *
 * @see run_iteration_diffuse_trail, run_iteration_process_particles, run_iteration_cleanup
 *
 * @note Again, this function only processes just one node. To use it, you'll need
 *
 * @note This function is a part of run_iteration kernels set. Because at some moments while running an iteration we
 *      have to synchronize all the threads (including inside blocks), the "run_iteration" operation is split into
 *      multiple kernel functions. Other functions (hopefully, all of them) are mentioned in the "see" block above
 */
__device__ inline void run_iteration_project_nutrients(SimulationMap *const simulation_map,
                                                       const int *const iteration_number, const unsigned int node_index)
{
    /// Pointer to the `MapNode` being processed by current thread
    MapNode *self;
    RUN_ITERATION_SET_SELF(self, node_index)

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
 * Performs a part of an iteration <b>for one node</b> (not self-sufficient, see note below
 *
 * Diffusion: the diffusion algorithm (developed by Jeff Jones) is pretty simple at first sight. We calculate an average
 * `trail` value in a 3x3 node window around the given one and multiply it by `(1 - jones_constants::diffdamp)`.
 * The new `temp_trail` value in the given node is the value just calculated. This is a natural way to implement the
 * smell spread: on each iteration smell moves more far from the source, but becomes less strong, because
 * `(1 - jones_constants::diffdamp)` < 1
 *
 * @param simulation_map Simulation map to run iteration on
 * @param iteration_number Serial number of current iteration
 * @param node_index Index of `SimulationMap`'s node to run iteration on. Can be out of bounds
 *
 * @see run_iteration_project_nutrients, run_iteration_process_particles, run_iteration_cleanup
 *
 * @note This function is a part of run_iteration kernels set. Because at some moments while running an iteration we
 *      have to synchronize all the threads (including inside blocks), the "run_iteration" operation is split into
 *      multiple kernel functions. Other functions (hopefully, all of them) are mentioned in the "see" block above
 */
__device__ inline void run_iteration_diffuse_trail(SimulationMap *const simulation_map,
                                                   const int *const iteration_number, const unsigned int node_index)
{
    MapNode *self;
    RUN_ITERATION_SET_SELF(self, node_index)

    auto left = self->get_left(), top = self->get_top(), right = self->get_right(), bottom = self->get_bottom();

    double sum = top->get_left()->trail + top->trail + top->get_right()->trail +
                 left->trail + self->trail + right->trail +
                 bottom->get_left()->trail + bottom->trail + bottom->get_right()->trail;

    self->temp_trail = (1 - jc::diffdamp) * (sum / 9.0);
}

/**
 * Performs a part of an iteration <b>for one node</b> (not self-sufficient, see note below
 *
 * Processing particles: For each node containing particles, the following operations are performed on the particles:
 * motor behaviours, sensory behaviours, division test, random death test, death test (order saved). You can read
 * details about these in the corresponding functions' docs
 *
 * @param simulation_map Simulation map to run iteration on
 * @param iteration_number Serial number of current iteration
 * @param node_index Index of `SimulationMap`'s node to run iteration on. Can be out of bounds
 *
 * @see run_iteration_project_nutrients, run_iteration_diffuse_trail, run_iteration_cleanup
 *
 * @note This function is a part of run_iteration kernels set. Because at some moments while running an iteration we
 *      have to synchronize all the threads (including inside blocks), the "run_iteration" operation is split into
 *      multiple kernel functions. Other functions (hopefully, all of them) are mentioned in the "see" block above
 */
__device__ inline void run_iteration_process_particles(SimulationMap *const simulation_map,
                                                       const int *const iteration_number, const unsigned int node_index)
{
    /// Pointer to the `MapNode` being processed by current thread
    MapNode *self;
    RUN_ITERATION_SET_SELF(self, node_index)

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
 * Performs a part of an iteration <b>for one node</b> (not self-sufficient, see note below
 *
 * Applies trail changes (sets `trail`s to `temp_trail`s), releases captured particles and increases `iteration_number`
 * by 1
 *
 * @param simulation_map Simulation map to run iteration on
 * @param iteration_number Serial number of current iteration
 * @param node_index Index of `SimulationMap`'s node to run iteration on. Can be out of bounds
 *
 * @see run_iteration_project_nutrients, run_iteration_diffuse_trail, run_iteration_process_particles
 *
 * @note This function is a part of run_iteration kernels set. Because at some moments while running an iteration we
 *      have to synchronize all the threads (including inside blocks), the "run_iteration" operation is split into
 *      multiple kernel functions. Other functions (hopefully, all of them) are mentioned in the "see" block above
 */
__device__ inline void run_iteration_cleanup(SimulationMap *const simulation_map, int *const iteration_number,
                                             const unsigned int node_index)
{
    /// Pointer to the `MapNode` being processed by current thread
    MapNode *self;
    RUN_ITERATION_SET_SELF(self, node_index)

    self->trail = self->temp_trail;
    if(self->contains_particle())
        self->get_particle()->release();

    if(node_index == 0)
        ++*iteration_number;
}


#endif //MIND_S_CRAWL_MAIN_LOGIC_CUH
