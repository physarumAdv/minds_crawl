#ifndef MIND_S_CRAWL_MAIN_LOGIC_CUH
#define MIND_S_CRAWL_MAIN_LOGIC_CUH


#ifdef COMPILE_FOR_CPU
#include <initializer_list>
#include <utility>
#endif //COMPILE_FOR_CPU

#include <cstdio>
#include <fstream>
#include <string>

#include "../lib/HTTPRequest/include/HTTPRequest.hpp"

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
 * Reads the first line from the file containing the visualization endpoint url and returns it
 *
 * Path to the file (relative to the executable's location) is `local/visualization_endpoint.txt`
 *
 * @returns Url to send data to be visualized to
 */
__host__ std::string get_visualization_endpoint()
{
    std::ifstream f("local/visualization_endpoint.txt", std::ios::in);
    std::string url;
    getline(f, url);
    return url;
}

/**
 * Converts a `vector` of `double` to a JSON array
 *
 * `std::to_string(double)` is used to create array elements
 *
 * @param v Array to be converted to JSON
 *
 * @returns `std::string` with JSON array of numbers with floating point
 */
__host__ std::string vector_double_to_json_array(const std::vector<double> &v)
{
    if(v.empty())
        return "[]";

    std::string ans = "[";
    for(double i : v)
    {
        ans += std::to_string(i) + ',';
    }
    ans.back() = ']';

    return ans;
}

/**
 * Send information about the particles of a simulation to visualization appliction
 *
 * The information about the existing particles is collected from the `nodes` array. Their coordinates are extracted.
 * The following JSON is generated: `{"x": [x coordinates of the particles (double)], "y": [y coordinates], "z": [z
 * coordinates]}`. For example, to represent two particles with coordinates (0, 0, 0) and (0.5, 1, 2), the following
 * JSON is generated: `{"x": [0.000000, 0.500000], "y": [0.000000, 1.000000], "z": [0.000000, 2.000000]}`. Then it is
 * sent to the given url as an HTTP POST request with `Content-Type: application/json`
 *
 * @param url HTTP url to send data to (protocols different from http are not supported)
 * @param nodes Pointer-represented array of nodes from the simulation (probably) containing particles
 * @param n_of_nodes Number of elements in the `nodes` array
 *
 * @returns `true` if the request was successfully sent, `false` if there was an exception from the http lib (it will
 *      be caught and printed to stderr)
 *
 * @note This function was not designed to be used in various contexts different from the simulation's main flow, were
 *      it's needed to send the data to visualization and only know, whether it was sent successfully or not (the
 *      end-user will see the error in stderr if it exists, but the function caller won't be able to handle it, because
 *      it's handled inside of `send_particles_to_visualization`). The reason for this decision is to reduce code
 *      duplication (because there are two main functions for cpp and cuda), which might mean for you that you don't
 *      want to use this function but want to write your own request sender
 */
__host__ bool send_particles_to_visualization(const std::string &url, MapNode *nodes, int n_of_nodes)
{
    std::vector<double> x, y, z;
    x.reserve(n_of_nodes);
    y.reserve(n_of_nodes);
    z.reserve(n_of_nodes);

    for(int i = 0; i < n_of_nodes; ++i)
    {
        if(!nodes[i].does_contain_particle())
            continue;

        SpacePoint coords = nodes[i].get_coordinates();
        x.push_back(coords.x);
        y.push_back(coords.y);
        z.push_back(coords.z);
    }

    std::string body = "{";
    body += "\"x\":" + vector_double_to_json_array(x) +
           ",\"y\":" + vector_double_to_json_array(y) +
           ",\"z\":" + vector_double_to_json_array(z) + "}";

    http::Request request(url);

    try
    {
        const http::Response response = request.send("POST", body, {"Content-Type: application/json"});

        if(response.status < 200 || 300 <= response.status)
            throw http::ResponseError("Response status is not OK");
    }
    catch(const std::exception &e)
    {
        std::cerr << "Request failed, error: " << e.what() << std::endl;
        return false;
    }

    return true;
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
            {edge_length, 0, 0}
    };
    SpacePoint vertices2[] = {
            {0, 0,           0},
            {0, edge_length, 0},
            {0, edge_length, edge_length},
            {0, 0,           edge_length}
    };
    SpacePoint vertices3[] = {
            {0,           0,           0},
            {edge_length, 0,           0},
            {edge_length, edge_length, 0},
            {0,           edge_length, 0}
    };
    SpacePoint vertices4[] = {
            {edge_length, 0,           edge_length},
            {edge_length, edge_length, edge_length},
            {edge_length, edge_length, 0},
            {edge_length, 0,           0}
    };
    SpacePoint vertices5[] = {
            {0,           0,           edge_length},
            {0,           edge_length, edge_length},
            {edge_length, edge_length, edge_length},
            {edge_length, 0,           edge_length}
    };
    SpacePoint vertices6[] = {
            {edge_length, edge_length, 0},
            {edge_length, edge_length, edge_length},
            {0,           edge_length, edge_length},
            {0,           edge_length, 0}
    };

    Face faces[] = {
            Face(vertices1, 4),
            Face(vertices2, 4),
            Face(vertices3, 4),
            Face(vertices4, 4),
            Face(vertices5, 4),
            Face(vertices6, 4)
    };

    Polyhedron cube(std::move(faces), 6);
    return cube;
}


#endif //MIND_S_CRAWL_MAIN_LOGIC_CUH
