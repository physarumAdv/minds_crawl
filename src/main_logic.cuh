#ifndef MIND_S_CRAWL_MAIN_LOGIC_CUH
#define MIND_S_CRAWL_MAIN_LOGIC_CUH


#ifdef COMPILE_FOR_CPU
#include <initializer_list>
#include <utility>
#endif //COMPILE_FOR_CPU

#include <cstdio>

#include "SimulationMap.cuh"
#include "Polyhedron.cuh"
#include "MapNode.cuh"
#include "Particle.cuh"
#include "fucking_shit.cuh"
#include "random_generator.cuh"
#include "jones_constants.hpp"
#include "common.cuh"

namespace jc = jones_constants;


#define RUN_ITERATION_SET_SELF(self, i) { \
            if(i >= simulation_map->get_n_of_nodes()) \
                return; \
            self = &simulation_map->nodes[i]; \
        }

typedef void (*RunIterationFunc)(SimulationMap *, int *);


/**
 * Initializes the simulation's objects (at the moment only a `SimulationMap`)
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
 *
 * @see destruct_simulation_objects
 */
__global__ void init_simulation_objects(SimulationMap *const simulation_map, Polyhedron *const polyhedron)
{
#ifndef COMPILE_FOR_CPU
    STOP_ALL_THREADS_EXCEPT_FIRST;
#endif

    *simulation_map = SimulationMap(polyhedron);
}

/**
 * Initializes environment (food, first particles) of the `SimulationMap`
 *
 * Purpose of this function is to flexibly perform additional configuration of a simulation map such as placing food
 * and first particles on it. However, at the moment (on the most early stages of development) it performs the most
 * basic configuration you can't control from outside of the function. Later the function is of course going to take
 * more parameters
 *
 * @param simulation_map Pointer to `SimulationMap` object to be configured
 */
__global__ void init_environment(SimulationMap *const simulation_map)
{
#ifndef COMPILE_FOR_CPU
    STOP_ALL_THREADS_EXCEPT_FIRST;
#endif

    int random_node_index = (int)(rand0to1() * (simulation_map->get_n_of_nodes() - 1));
    MapNode *random_node = simulation_map->nodes + random_node_index;
    if(!create_particle(random_node))
    {
        printf("%s:%d - something went REALLY wrong at ", __FILE__, __LINE__);
    }
}

/**
 * The opposite of `init_simulation_objects`
 *
 * This function destructs the objects constructed in `init_simulation_objects` (at the moment only a `SimulationMap`)
 * to let you safely free memory without breaking any invariants etc
 *
 * @param simulation_map Pointer to the `SimulationMap` to be destructed
 *
 * @see init_simulation_objects
 */
__global__ void destruct_simulation_objects(SimulationMap *const simulation_map)
{
#ifndef COMPILE_FOR_CPU
    STOP_ALL_THREADS_EXCEPT_FIRST;
#endif

    simulation_map->~SimulationMap();
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

    diffuse_trail(self);
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
    SpacePoint vertices1[] = {
            {0,           0, 0},
            {0,           0, edge_length},
            {edge_length, 0, edge_length},
            {edge_length, 0, 0},
            {0,           0, 0}
    };
    SpacePoint vertices2[] = {
            {0, 0,           0},
            {0, edge_length, 0},
            {0, edge_length, edge_length},
            {0, 0,           edge_length},
            {0, 0,           0}
    };
    SpacePoint vertices3[] = {
            {0,           0,           0},
            {edge_length, 0,           0},
            {edge_length, edge_length, 0},
            {0,           edge_length, 0},
            {0,           0,           0}
    };
    SpacePoint vertices4[] = {
            {edge_length, 0,           edge_length},
            {edge_length, edge_length, edge_length},
            {edge_length, edge_length, 0},
            {edge_length, 0,           0},
            {edge_length, 0,           edge_length}
    };
    SpacePoint vertices5[] = {
            {0,           0,           edge_length},
            {0,           edge_length, edge_length},
            {edge_length, edge_length, edge_length},
            {edge_length, 0,           edge_length},
            {0,           0,           edge_length}
    };
    SpacePoint vertices6[] = {
            {edge_length, edge_length, 0},
            {edge_length, edge_length, edge_length},
            {0,           edge_length, edge_length},
            {0,           edge_length, 0},
            {edge_length, edge_length, 0}
    };

    Face faces[] = {
            Face(vertices1, 5),
            Face(vertices2, 5),
            Face(vertices3, 5),
            Face(vertices4, 5),
            Face(vertices5, 5),
            Face(vertices6, 5)
    };

    Polyhedron cube(std::move(faces), 6);
    return cube;
}


#endif //MIND_S_CRAWL_MAIN_LOGIC_CUH
