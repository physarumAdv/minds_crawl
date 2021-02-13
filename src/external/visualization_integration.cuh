#ifndef MINDS_CRAWL_VISUALIZATION_INTEGRATION_CUH
#define MINDS_CRAWL_VISUALIZATION_INTEGRATION_CUH


#include "../../lib/HTTPRequest/include/HTTPRequest.hpp"
#include <string>
#include <vector>

#include "../simulation_objects/MapNode.cuh"


/**
 * Reads the first 2 lines from the file containing the visualization endpoint urls and returns it
 * We have 2 urls (one in each line), one for sending particles to and other for sending model to
 *
 * Path to the file (relative to the executable's location) is `config/visualization_endpoint.txt`
 *
 * @returns std::pair<std::string, std::string> with each string being a url to send data to be visualized to
 */
__host__ std::pair<std::string, std::string> get_visualization_endpoints();

/**
 * Converts a `vector` of `double` to a JSON array
 *
 * `std::to_string(double)` is used to create array elements
 *
 * @param v Array to be converted to JSON
 *
 * @returns `std::string` with JSON array of numbers with floating point
 */
__host__ std::string vector_double_to_json_array(const std::vector<double> &v);


/**
 * Send information about the particles of a simulation to visualization application
 *
 * The information about the existing particles is collected from the `nodes` array. Their coordinates are extracted.
 * The following JSON is generated: `{"x": [x coordinates of the particles (double)], "y": [y coordinates], "z": [z
 * coordinates]}`. For example, to represent two particles with coordinates (0, 0, 0) and (0.5, 1, 2), the following
 * JSON is generated: `{"x": [0.000000, 0.500000], "y": [0.000000, 1.000000], "z": [0.000000, 2.000000]}`. Then it is
 * sent to the given url as an HTTP POST request with `Content-Type: application/json`
 *
 * @param urls HTTP two urls each to send data to. One for particles array and one for polyhedron. (protocols different from http are not supported)
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
__host__ bool send_poly_to_visualization(const std::pair<std::string, std::string> &urls, const Polyhedron *polyhedron);

__host__ bool send_particles_to_visualization(const std::pair<std::string, std::string> &urls,
                                              const MapNodeReflection *nodes, int n_of_nodes);


#endif //MINDS_CRAWL_VISUALIZATION_INTEGRATION_CUH
