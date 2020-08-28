#ifndef MIND_S_CRAWL_MAIN_LOGIC_CUH
#define MIND_S_CRAWL_MAIN_LOGIC_CUH


#include <initializer_list>

#include "SimulationMap.cuh"
#include "Polyhedron.cuh"
#include "MapNode.cuh"
#include "Particle.cuh"
#include "fucking_shit.cuh"
#include "jones_constants.hpp"
#include "common.cuh"

namespace jc = jones_constants;


#define RUN_ITERATION_SET_SELF(self, i) { \
            if(i >= simulation_map->get_n_of_nodes()) \
                return; \
            self = &simulation_map->nodes[i]; \
        }

typedef void (*RunIterationFunc)(SimulationMap *, int *);


// TODO: Add the opposite (destructing) function
/**
 * Initializes the simulation's objects (at the moment only `SimulationMap`)
 *
 * This function is used to initialize simulation objects on device. It at the moment only initializes `SimulationMap`,
 * but can be extended to initialize more objects
 *
 * @param simulation_map Pointer to `SimulationMap`. A constructed simulation map will be moved there
 * @param polyhedron Pointer to a polyhedron to be used to initialize `SimulationMap` (constructor's parameter)
 *
 * @warning While `polyhedron` parameter must point to a real `Polyhedron` object, `simulation_map` might contain an
 *      existing map already, but it will be destructed in this case. Note that if the pointer doesn't contain an
 *      object the destructor will be called anyway, but it should be safe (???)
 */
__global__ void init_simulation_objects(SimulationMap *const simulation_map, Polyhedron *const polyhedron)
{
    STOP_ALL_THREADS_EXCEPT_FIRST;

    *simulation_map = SimulationMap(polyhedron);
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
        if(self->does_contain_food())
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

    if(!self->does_contain_particle() || !self->get_particle()->capture())
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
    if(self->does_contain_particle())
        self->get_particle()->release();

    if(node_index == 0)
        ++*iteration_number;
}


/**
 * Creates a cube and returns it
 *
 * Will be only used in the early stages of development, later will be replaced with a universal function building
 * arbitrary polyhedrons
 * 
 * @param edge_length Length of the cube's edge
 *
 * @returns Cube represented wth a `Polyhedron` object
 */
__host__ inline Polyhedron generate_cube(double edge_length = 200)
{
    using std::move;


    Face face1((SpacePoint[]){
            {0,           0, 0},
            {0,           0, edge_length},
            {edge_length, 0, edge_length},
            {edge_length, 0, 0}
    }, 4);
    Face face2((SpacePoint[]){
            {0, 0,           0},
            {0, edge_length, 0},
            {0, edge_length, edge_length},
            {0, 0,           edge_length}
    }, 4);
    Face face3((SpacePoint[]){
            {0,           0,           0},
            {edge_length, 0,           0},
            {edge_length, edge_length, 0},
            {0,           edge_length, 0}
    }, 4);
    Face face4((SpacePoint[]){
            {edge_length, 0,           edge_length},
            {edge_length, edge_length, edge_length},
            {edge_length, edge_length, 0},
            {edge_length, 0,           0}
    }, 4);
    Face face5((SpacePoint[]){
            {0,           0,           edge_length},
            {0,           edge_length, edge_length},
            {edge_length, edge_length, edge_length},
            {edge_length, 0,           edge_length}
    }, 4);
    Face face6((SpacePoint[]){
            {edge_length, edge_length, 0},
            {edge_length, edge_length, edge_length},
            {0,           edge_length, edge_length},
            {0,           edge_length, 0}
    }, 4);

    Polyhedron cube(
            (Face[]){move(face1), move(face2), move(face3), move(face4), move(face5), move(face6)},
            6
    );
    return cube;
}


#endif //MIND_S_CRAWL_MAIN_LOGIC_CUH
