#include "Particle.cuh"
#include "fucking_shit.cuh"
#include "random_generator.cuh"
#include "jones_constants.hpp"

namespace jc = jones_constants;


__device__ Particle::Particle(MapNode *map_node, double angle) :
        coordinates(map_node->get_coordinates()), map_node(map_node)
{
    Face &current_face = map_node->get_polyhedron()->faces[map_node->get_face_id()];
    normal = current_face.get_normal();

    SpacePoint radius = current_face.get_vertices()[0] - coordinates;
    radius = radius * jc::speed / get_distance(radius, origin);
    direction_vector = rotate_point_from_agent(radius, angle);
}


__device__ SpacePoint Particle::rotate_point_from_agent(SpacePoint radius, double angle) const
{
    return relative_point_rotation(coordinates, coordinates + radius, normal, angle);
}


__device__ void Particle::do_motor_behaviours()
{
    if(rand0to1() < jc::pcd)
    {
        rotate(rand0to1() * 2 * M_PI);
    }

    SpacePoint end = get_projected_vector_end(coordinates, coordinates + direction_vector,
                                              map_node->get_face_id(), map_node->get_polyhedron());
    MapNode *new_node = find_nearest_mapnode(map_node->get_polyhedron(), end, map_node);
    if(new_node->attach_particle(this)) // If can reattach myself to that node
    {
        map_node->detach_particle(this);
        map_node = new_node;
        normal = map_node->get_polyhedron()->faces[map_node->get_face_id()].get_normal();
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

__device__ void Particle::do_sensory_behaviours()
{
    Polyhedron *p = map_node->get_polyhedron();
    SpacePoint m_sensor_direction = direction_vector * jc::so / jc::speed;

    double trail_l = find_nearest_mapnode(p, rotate_point_from_agent(m_sensor_direction, -jc::sa), map_node)->trail;
    double trail_m = find_nearest_mapnode(p, m_sensor_direction, map_node)->trail;
    double trail_r = find_nearest_mapnode(p, rotate_point_from_agent(m_sensor_direction, jc::sa), map_node)->trail;

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

__device__ void Particle::rotate(double angle)
{
    direction_vector = rotate_point_from_agent(direction_vector, angle);
}


[[nodiscard]] __device__ bool Particle::capture()
{
    return !atomicCAS(&is_captured, false, true);
}

__device__ void Particle::release()
{
    is_captured = false;
}


__device__ SpacePoint get_projected_vector_end(SpacePoint a, SpacePoint b, int current_face_id, Polyhedron *polyhedron)
{
    Face &current_face = polyhedron->faces[current_face_id];
    for(int i = 0; i < current_face.get_n_of_vertices() - 1; ++i)
    {
        SpacePoint intersection = line_intersection(current_face.get_vertices()[i], current_face.get_vertices()[i + 1],
                                                    a, b);
        if(intersection != origin && is_in_segment(a, b, intersection) &&
           is_in_segment(current_face.get_vertices()[i], current_face.get_vertices()[i + 1], intersection) &&
           get_distance(intersection, a) > eps)
        {
            Face &next_face = polyhedron->faces[find_face_next_to_edge(a, b, current_face_id, polyhedron)];
            SpacePoint normal_before = current_face.get_normal();
            SpacePoint normal_after = next_face.get_normal();
            SpacePoint moving_vector = (b - a) / get_distance(a, b);
            double phi_cos = normal_after * normal_before;
            double phi_sin = sin(acos(phi_cos));
            double alpha_cos = moving_vector * (normal_before % normal_after);

            SpacePoint faced_vector_direction =
                    (normal_before + normal_after * phi_cos) * sin(acos(alpha_cos)) /
                    phi_sin + (normal_before % normal_after) * alpha_cos / phi_sin;

            return intersection + faced_vector_direction * (get_distance(a, b) - get_distance(intersection, a));
        }
    }
    return b;
}


__device__ int find_face_next_to_edge(SpacePoint a, SpacePoint b, int current_face_id, Polyhedron *polyhedron)
{
    for(int i = 0; i < polyhedron->n_of_faces; ++i)
        if(polyhedron->faces[i].get_id() != current_face_id && does_edge_belong_to_face(a, b, &polyhedron->faces[i]))
            return i;
    return current_face_id;
}

__device__ bool is_edge_belongs_face(SpacePoint a, SpacePoint b, const Face *const face)
{
    bool flag1 = false, flag2 = false;
    for(int i = 0; i < face->get_n_of_vertices(); ++i)
    {
        if(face->get_vertices()[i] == a)
            flag1 = true;
        if(face->get_vertices()[i] == b)
            flag2 = true;
    }
    return flag1 && flag2;
}
