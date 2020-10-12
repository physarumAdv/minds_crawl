#include "Particle.cuh"
#include "MapNode.cuh"
#include "random_generator.cuh"
#include "jones_constants.hpp"

namespace jc = jones_constants;


__host__ __device__ Particle::Particle(MapNode *map_node, double angle) :
        coordinates(map_node->get_coordinates()), map_node(map_node)
{
    Face *current_face = map_node->get_face();
    normal = current_face->get_normal();

    SpacePoint radius = current_face->get_vertices()[0] - coordinates;
    radius = radius * jc::speed / get_distance(radius, origin);
    direction_vector = rotate_point_from_agent(radius, angle, false) - coordinates;
}


__host__ __device__ SpacePoint Particle::rotate_point_from_agent(SpacePoint radius, double angle,
                                                                 bool do_projection) const
{
    SpacePoint after_rotation = relative_point_rotation(coordinates, coordinates + radius, normal, angle);
    if(do_projection)
        after_rotation = get_projected_vector_end(coordinates, after_rotation,
                                                  map_node->get_face(),
                                                  map_node->get_polyhedron());
    return after_rotation;
}


__device__ void Particle::do_motor_behaviours()
{
    if(rand0to1() < jc::pcd)
    {
        rotate(rand0to1() * 2 * M_PI);
    }

    SpacePoint end = coordinates + direction_vector; // New coordinates if particle doesn't change face
    MapNode *node_neighbors[] = {map_node->get_left(), map_node->get_top(), map_node->get_right(),
                                 map_node->get_bottom()};
    for(MapNode *neighbor : node_neighbors)
    {
        if(neighbor->get_face() != map_node->get_face()) // If any of the neighbor nodes is on a different face
        {
            // Then particle might move to a different face, so project coordinates to the polyhedron's surface
            end = get_projected_vector_end(coordinates, end, map_node->get_face(), map_node->get_polyhedron());
            break;
        }
    }

    MapNode *new_node = find_nearest_mapnode(map_node->get_polyhedron(), end, map_node);
    if(new_node->attach_particle(this)) // If can reattach myself to that node
    {
        map_node->detach_particle(this);
        map_node = new_node;
        normal = map_node->get_face()->get_normal();
    }
    if(*new_node == *map_node) // If either just reattached myself successfully or trying to move to the same node
    {
        map_node->trail += jc::dept;
        coordinates = end;
    }
    else if(!jc::osc)
    {
        rotate(rand0to1() * 2 * M_PI);
    }
}

__host__ __device__ void Particle::do_sensory_behaviours()
{
    Polyhedron *p = map_node->get_polyhedron();
    SpacePoint m_sensor_direction = direction_vector * jc::so / jc::speed;

    double trail_l = find_nearest_mapnode(p, rotate_point_from_agent(m_sensor_direction, -jc::sa,
                                                                     true), map_node)->trail;
    // Rotate 0 radians is not useless! The `rotate_point_from_agent` also projects vector to polyhedron's surface
    double trail_m = find_nearest_mapnode(p, rotate_point_from_agent(m_sensor_direction, 0,
                                                                     true), map_node)->trail;
    double trail_r = find_nearest_mapnode(p, rotate_point_from_agent(m_sensor_direction, jc::sa,
                                                                     true), map_node)->trail;

    if((trail_m > trail_l) && (trail_m > trail_r)) // m > l, r
        return;
    if((trail_m < trail_l) && (trail_m < trail_r)) // m < l, r
    {
        if(trail_l < trail_r) // m < l < r
            rotate(jc::ra);
        else // m < r <= l
            rotate(-jc::ra);

        return;
    }
    if(trail_l < trail_r) // l < m < r
        rotate(jc::ra);
    else // r < m < l
        rotate(-jc::ra);
}

__host__ __device__ void Particle::rotate(double angle)
{
    direction_vector = rotate_point_from_agent(direction_vector, angle, false) - coordinates;
}


[[nodiscard]] __device__ bool Particle::capture()
{
    return !atomicCAS(&is_captured, false, true);
}

__device__ void Particle::release()
{
    is_captured = false;
}
