#include "SimulationMap.cuh"
#include "Polyhedron.cuh"
#include "Face.cuh"
#include "jones_constants.hpp"
#include "random_generator.cuh"
#include <cmath>

namespace jc = jones_constants;


__device__ double const mapnode_dist = 2 * jc::speed;


__device__ SimulationMap::SimulationMap(Polyhedron *polyhedron) :
        polyhedron(polyhedron)
{
    bool create_new_nodes = true;  // New nodes are allowed to be created

    Face &start_face = polyhedron->faces[0];
    SpacePoint start_node_coordinates = (start_face.vertices[0] + start_face.vertices[1] +
                                         start_face.vertices[2]) / 3;
    nodes = (MapNode *)malloc(sizeof(MapNode));
    nodes[0] = MapNode(polyhedron, start_face.id, start_node_coordinates);
    n_of_nodes = 1;

    // Direction vector from first node to its top neighbor sets randomly
    SpacePoint direction_vector = relative_point_rotation(start_node_coordinates, start_face.vertices[0],
                                                          start_face.normal, M_PI * 2 * rand0to1());
    // Array of direction vectors from nodes by index to their top neighbors
    auto *nodes_directions = (SpacePoint *)malloc(sizeof(SpacePoint));
    nodes_directions[0] = direction_vector * mapnode_dist / get_distance(direction_vector, origin);

    // Creating new nodes until it can be done, some nodes may have less neighbors than four
    for(int current_node_id = 0; current_node_id < n_of_nodes; ++current_node_id)
    {
        MapNode &current_node = nodes[current_node_id];
        if(current_node.get_top() == nullptr)
        {
            int top_node_id = get_neighbor_node_id(current_node_id, &nodes_directions, 0, create_new_nodes);
            if (top_node_id != -1)
            {
                current_node.set_top(&nodes[top_node_id]);
                nodes[top_node_id].set_bottom(&current_node);
            }
        }
        if(current_node.get_left() == nullptr)
        {
            int left_node_id = get_neighbor_node_id(current_node_id, &nodes_directions, -M_PI_2, create_new_nodes);
            if (left_node_id != -1)
            {
                current_node.set_left(&nodes[left_node_id]);
                nodes[left_node_id].set_right(&current_node);
            }
        }
        if(current_node.get_bottom() == nullptr)
        {
            int bottom_node_id = get_neighbor_node_id(current_node_id, &nodes_directions, M_PI, create_new_nodes);
            if (bottom_node_id != -1)
            {
                current_node.set_bottom(&nodes[bottom_node_id]);
                nodes[bottom_node_id].set_top(&current_node);
            }
        }
        if(current_node.get_right() == nullptr)
        {
            int right_node_id = get_neighbor_node_id(current_node_id, &nodes_directions, M_PI_2, create_new_nodes);
            if (right_node_id != -1)
            {
                current_node.set_right(&nodes[right_node_id]);
                nodes[right_node_id].set_left(&current_node);
            }
        }
    }

    // All nodes were created
    create_new_nodes = false;

    // Setting all neighbors that were not set earlier, neighbor is the closest node to neighbor coordinates
    for (int current_node_id = 0; current_node_id < n_of_nodes; ++current_node_id)
    {
        MapNode &current_node = nodes[current_node_id];
        if(current_node.get_top() == nullptr)
        {
            int top_mapnode_id = get_neighbor_node_id(current_node_id, &nodes_directions, 0, create_new_nodes);
            current_node.set_top(&nodes[top_mapnode_id]);
        }
        if(current_node.get_left() == nullptr)
        {
            int left_mapnode_id = get_neighbor_node_id(current_node_id, &nodes_directions, -M_PI_2, create_new_nodes);
            current_node.set_left(&nodes[left_mapnode_id]);
        }
        if(current_node.get_bottom() == nullptr)
        {
            int bottom_mapnode_id = get_neighbor_node_id(current_node_id, &nodes_directions, M_PI, create_new_nodes);
            current_node.set_bottom(&nodes[bottom_mapnode_id]);
        }
        if(current_node.get_right() == nullptr)
        {
            int right_mapnode_id = get_neighbor_node_id(current_node_id, &nodes_directions, M_PI_2, create_new_nodes);
            current_node.set_right(&nodes[right_mapnode_id]);
        }
    }
}

__device__ SimulationMap::~SimulationMap()
{
    free(nodes);
}


__device__ SpacePoint SimulationMap::count_neighbor_node_coordinates(int current_node_id, SpacePoint top_direction,
                                                                     double angle, bool do_projection) const
{
    MapNode *current_node = &nodes[current_node_id];
    int current_face_id = current_node->get_face_id();
    SpacePoint neighbor_coordinates = relative_point_rotation(current_node->get_coordinates(),
                                                              current_node->get_coordinates() + top_direction,
                                                              polyhedron->faces[current_face_id].normal, angle);
    if (do_projection)
    {
        return get_projected_vector_end(current_node->get_coordinates(), neighbor_coordinates, current_face_id,
                                        polyhedron);
    }
    else
    {
        return neighbor_coordinates;
    }
}


__device__ int SimulationMap::find_index_of_nearest_node(SpacePoint dest) const
{
    int nearest_mapnode_id = 0;
    for(int neighbor = 0; neighbor < n_of_nodes; ++neighbor)
    {
        if (get_distance(nodes[neighbor].get_coordinates(), dest) <
            get_distance(nodes[nearest_mapnode_id].get_coordinates(), dest))
        {
            nearest_mapnode_id = neighbor;
        }
    }
    return nearest_mapnode_id;
}


__device__ void SimulationMap::set_direction_to_top_neighbor(int current_node_id, int neighbor_node_id,
                                                             SpacePoint **nodes_directions, double angle) const
{
    MapNode *neighbor_node = &nodes[neighbor_node_id];
    MapNode *current_node = &nodes[current_node_id];

    if(neighbor_node->get_face_id() == current_node->get_face_id())
    {
        (*nodes_directions)[neighbor_node_id] = (*nodes_directions)[current_node_id];
    }
    else
    {
        SpacePoint new_direction = neighbor_node->get_coordinates() -
                find_intersection_with_edge(current_node->get_coordinates(),
                        count_neighbor_node_coordinates(current_node_id, (*nodes_directions)[current_node_id],
                                                           angle, false),
                        &polyhedron->faces[current_node->get_face_id()]);
        new_direction = relative_point_rotation(neighbor_node->get_coordinates(),
                                                neighbor_node->get_coordinates() + new_direction,
                                                polyhedron->faces[neighbor_node->get_face_id()].normal,
                                                -angle) -
                        neighbor_node->get_coordinates();
        (*nodes_directions)[neighbor_node_id] = new_direction * mapnode_dist / get_distance(new_direction, origin);
    }
}


__device__ int SimulationMap::get_neighbor_node_id(int current_node_id, SpacePoint **nodes_directions, double angle,
                                                   bool create_new_nodes)
{
    MapNode *current_node = &nodes[current_node_id];
    int current_face_id = current_node->get_face_id();

    // Hypothetical coordinates of neighbor node counted using direction to the top neighbor and `angle`
    SpacePoint neighbor_coordinates = count_neighbor_node_coordinates(current_node_id,
                                                                      (*nodes_directions)[current_node_id], angle,
                                                                      true);
    int next_face_id = polyhedron->find_face_id_by_point(neighbor_coordinates);
    int nearest_node_id = find_index_of_nearest_node(neighbor_coordinates);
    if (!create_new_nodes || (current_face_id == nodes[nearest_node_id].get_face_id() &&
            get_distance(nodes[nearest_node_id].get_coordinates(), neighbor_coordinates) < eps))
    {
        // Neighbor node has already existed
        return nearest_node_id;
    }
    else if(current_face_id == next_face_id || polyhedron->faces[next_face_id].get_node() == nullptr)
    {
        // Neighbor node does not exist, but it can be created
        nodes = device_realloc(nodes, n_of_nodes, n_of_nodes + 1);
        *nodes_directions = device_realloc(*nodes_directions, n_of_nodes, n_of_nodes + 1);
        nodes[n_of_nodes] = MapNode(polyhedron, next_face_id, neighbor_coordinates);
        polyhedron->faces[next_face_id].set_node(&nodes[n_of_nodes], polyhedron);
        set_direction_to_top_neighbor(current_node_id, n_of_nodes, nodes_directions, angle);
        n_of_nodes++;
        return n_of_nodes - 1;
    }
    return -1;
}


__device__ int SimulationMap::get_n_of_nodes() const
{
    return this->n_of_nodes;
}

__global__ void get_n_of_nodes(const SimulationMap *const simulation_map, int *return_value)
{
    stop_all_threads_except_first;

    *return_value = simulation_map->get_n_of_nodes();
}

