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
    Face &start_face = polyhedron->faces[0];
    SpacePoint start_mapnode_coordinates = (start_face.vertices[0] + start_face.vertices[1] +
                                            start_face.vertices[2]) / 3;
    nodes = (MapNode *)malloc(sizeof(MapNode));
    nodes[0] = MapNode(polyhedron, start_face.id, start_mapnode_coordinates);
    n_of_nodes = 1;

    SpacePoint direction_vector = relative_point_rotation(start_mapnode_coordinates, start_face.vertices[0],
                                                          start_face.normal, M_PI * 2 * rand0to1());
    SpacePoint *nodes_directions = (SpacePoint *)malloc(sizeof(SpacePoint));
    nodes_directions[0] = direction_vector * mapnode_dist / get_distance(direction_vector, origin);

    for(int current_node_id = 0; current_node_id < n_of_nodes; ++current_node_id)
    {
        SpacePoint top_direction = nodes_directions[current_node_id];
        MapNode &current_node = nodes[current_node_id];
        Face &current_face = polyhedron->faces[current_node.get_face_id()];

        if(current_node.get_top() == nullptr)
        {
            SpacePoint top_coordinates = current_node.get_coordinates() + top_direction;
            int top_mapnode_id = get_neighbor_mapnode_id(current_node_id, top_coordinates,
                                                         &nodes_directions, top_direction);
            if (top_mapnode_id != -1)
            {
                current_node.set_top(&nodes[top_mapnode_id]);
                nodes[top_mapnode_id].set_bottom(&current_node);
            }
            if (top_mapnode_id == n_of_nodes - 1)
            {
                if (nodes[top_mapnode_id].get_face_id() == current_face.id)
                {
                    nodes_directions[n_of_nodes] = top_direction;
                }
                else
                {
                    SpacePoint new_direction = get_projected_vector_end(current_node.get_coordinates(),
                            top_coordinates, current_face.id, polyhedron) -
                            find_intersection_with_edge(current_node.get_coordinates(), top_coordinates, &current_face);
                    nodes_directions[n_of_nodes] = new_direction * mapnode_dist / get_distance(new_direction, origin);
                }
            }
        }
        if(current_node.get_left() == nullptr)
        {
            SpacePoint left_coordinates = relative_point_rotation(current_node.get_coordinates(),
                                                                  current_node.get_coordinates() + top_direction,
                                                                  current_face.normal, -M_PI / 2);
            int left_mapnode_id = get_neighbor_mapnode_id(current_node_id, left_coordinates,
                                                          &nodes_directions, top_direction);
            if (left_mapnode_id != -1)
            {
                current_node.set_left(&nodes[left_mapnode_id]);
                nodes[left_mapnode_id].set_right(&current_node);
            }
            if (left_mapnode_id == n_of_nodes - 1)
            {
                if (nodes[left_mapnode_id].get_face_id() == current_face.id)
                {
                    nodes_directions[n_of_nodes] = top_direction;
                }
                else
                {
                    SpacePoint new_direction = get_projected_vector_end(current_node.get_coordinates(),
                            left_coordinates, current_face.id, polyhedron) -
                            find_intersection_with_edge(current_node.get_coordinates(),
                            left_coordinates, &current_face);
                    new_direction = relative_point_rotation(left_coordinates, left_coordinates + new_direction,
                            polyhedron->faces[nodes[left_mapnode_id].get_face_id()].normal, M_PI / 2) -
                            left_coordinates;
                    nodes_directions[n_of_nodes] = new_direction * mapnode_dist / get_distance(new_direction, origin);
                }
            }
        }
        if(current_node.get_bottom() == nullptr)
        {
            SpacePoint bottom_coordinates = current_node.get_coordinates() - top_direction;
            int bottom_mapnode_id = get_neighbor_mapnode_id(current_node_id, bottom_coordinates,
                                                            &nodes_directions, top_direction);
            if (bottom_mapnode_id != -1)
            {
                current_node.set_bottom(&nodes[bottom_mapnode_id]);
                nodes[bottom_mapnode_id].set_top(&current_node);
            }
            if (bottom_mapnode_id == n_of_nodes - 1)
            {
                if (nodes[bottom_mapnode_id].get_face_id() == current_face.id)
                {
                    nodes_directions[n_of_nodes] = top_direction;
                }
                else
                {
                    SpacePoint new_direction = get_projected_vector_end(current_node.get_coordinates(),
                            bottom_coordinates, current_face.id, polyhedron) -
                            find_intersection_with_edge(current_node.get_coordinates(),
                            bottom_coordinates, &current_face);
                    nodes_directions[n_of_nodes] = (-1) * new_direction * mapnode_dist /
                                                   get_distance(new_direction, origin);
                }
            }
        }
        if(current_node.get_right() == nullptr)
        {
            SpacePoint right_coordinates = relative_point_rotation(current_node.get_coordinates(),
                                                                   current_node.get_coordinates() + top_direction,
                                                                   current_face.normal, M_PI / 2);
            int right_mapnode_id = get_neighbor_mapnode_id(current_node_id, right_coordinates,
                                                           &nodes_directions, top_direction);
            if (right_mapnode_id != -1)
            {
                current_node.set_right(&nodes[right_mapnode_id]);
                nodes[right_mapnode_id].set_left(&current_node);
            }
            if (right_mapnode_id == n_of_nodes - 1)
            {
                if (nodes[right_mapnode_id].get_face_id() == current_face.id)
                {
                    nodes_directions[n_of_nodes] = top_direction;
                }
                else
                {
                    SpacePoint new_direction = get_projected_vector_end(current_node.get_coordinates(),
                            right_coordinates, current_face.id, polyhedron) -
                            find_intersection_with_edge(current_node.get_coordinates(),
                            right_coordinates, &current_face);
                    new_direction = relative_point_rotation(right_coordinates, right_coordinates + new_direction,
                            polyhedron->faces[nodes[right_mapnode_id].get_face_id()].normal, -M_PI / 2) -
                            right_coordinates;
                    nodes_directions[n_of_nodes] = new_direction * mapnode_dist / get_distance(new_direction, origin);
                }
            }
        }
    }

    for (int current_node_id = 0; current_node_id < n_of_nodes; ++current_node_id)
    {
        SpacePoint top_direction = nodes_directions[current_node_id];
        MapNode &current_node = nodes[current_node_id];
        Face &current_face = polyhedron->faces[current_node.get_face_id()];

        if(current_node.get_top() == nullptr)
        {
            SpacePoint top_coordinates = current_node.get_coordinates() + top_direction;
            top_coordinates = get_projected_vector_end(nodes[current_node_id].get_coordinates(),
                                                       top_coordinates, current_face.id, polyhedron);
            int top_mapnode_id = find_nearest_mapnode_to_point(top_coordinates);
            current_node.set_top(&nodes[top_mapnode_id]);
        }
        if(current_node.get_left() == nullptr)
        {
            SpacePoint left_coordinates = relative_point_rotation(current_node.get_coordinates(),
                                                                  current_node.get_coordinates() + top_direction,
                                                                  current_face.normal, -M_PI / 2);
            left_coordinates = get_projected_vector_end(nodes[current_node_id].get_coordinates(),
                                                        left_coordinates, current_face.id, polyhedron);
            int left_mapnode_id = find_nearest_mapnode_to_point(left_coordinates);
            current_node.set_left(&nodes[left_mapnode_id]);
        }
        if(current_node.get_bottom() == nullptr)
        {
            SpacePoint bottom_coordinates = current_node.get_coordinates() - top_direction;
            bottom_coordinates = get_projected_vector_end(nodes[current_node_id].get_coordinates(),
                                                          bottom_coordinates, current_face.id, polyhedron);
            int bottom_mapnode_id = find_nearest_mapnode_to_point(bottom_coordinates);
            current_node.set_bottom(&nodes[bottom_mapnode_id]);
        }
        if(current_node.get_right() == nullptr)
        {
            SpacePoint right_coordinates = relative_point_rotation(current_node.get_coordinates(),
                                                                   current_node.get_coordinates() + top_direction,
                                                                   current_face.normal, M_PI / 2);
            right_coordinates = get_projected_vector_end(nodes[current_node_id].get_coordinates(),
                                                         right_coordinates, current_face.id, polyhedron);
            int right_mapnode_id = find_nearest_mapnode_to_point(right_coordinates);
            current_node.set_right(&nodes[right_mapnode_id]);
        }
    }
}

__device__ SimulationMap::~SimulationMap()
{
    free(nodes);
}


__device__ int SimulationMap::find_nearest_mapnode_to_point(SpacePoint point_coordinates) const
{
    int nearest_mapnode_id = 0;
    for(int neighbor = 0; neighbor < n_of_nodes; ++neighbor)
    {
        if (get_distance(nodes[neighbor].get_coordinates(), point_coordinates) <
                get_distance(nodes[nearest_mapnode_id].get_coordinates(), point_coordinates))
        {
            nearest_mapnode_id = neighbor;
        }
    }
    return nearest_mapnode_id;
}

__device__ int SimulationMap::get_neighbor_mapnode_id(int current_node_id, SpacePoint neighbor_coordinates,
                                                      SpacePoint **nodes_directions, SpacePoint top_direction)
{
    int current_face_id = nodes[current_node_id].get_face_id();
    neighbor_coordinates = get_projected_vector_end(nodes[current_node_id].get_coordinates(), neighbor_coordinates,
                                                    current_face_id, polyhedron);
    int next_face_id = polyhedron->find_face_id_by_point(neighbor_coordinates);
    int neighbor_mapnode_id = find_nearest_mapnode_to_point(neighbor_coordinates);
    if (current_face_id == nodes[neighbor_mapnode_id].get_face_id() &&
        get_distance(nodes[neighbor_mapnode_id].get_coordinates(), neighbor_coordinates) < mapnode_dist / 100)
    {
        return neighbor_mapnode_id;
    }
    else if(current_face_id == next_face_id || polyhedron->faces[next_face_id].node == nullptr)
    {
        nodes = device_realloc(nodes, n_of_nodes, n_of_nodes + 1);
        nodes[n_of_nodes] = MapNode(polyhedron, next_face_id, neighbor_coordinates);
        polyhedron->faces[next_face_id].node = &nodes[n_of_nodes];
        *nodes_directions = device_realloc(*nodes_directions, n_of_nodes, n_of_nodes + 1);
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

