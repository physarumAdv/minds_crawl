#include "Particle.cuh"
#include "fucking_shit.cuh"
#include "jones_constants.hpp"

namespace jc = jones_constants;


__device__ Particle::Particle(MapNode *map_node, SpacePoint coordinates, double angle) :
    coordinates(coordinates), map_node(map_node)
{
    Face current_face = map_node->polyhedron->faces[map_node->polyhedron_face_id];
    normal = current_face.normal;

    SpacePoint radius = current_face.vertices[0] - coordinates;
    radius = radius * jc::speed / get_distance(radius, origin);
    direction_vector = rotate_point_angle(radius, angle);
}


__device__ SpacePoint Particle::rotate_point_angle(SpacePoint radius, double angle) const
{
    double angle_cos = cos(angle);
    return (1 - angle_cos) * (this->normal * radius) * this->normal + angle_cos * radius +
                      sin(angle) * (this->normal % radius) + this->coordinates;
}


__device__ void Particle::do_sensory_behaviours()
{
    Polyhedron *p = map_node->polyhedron;

    double trail_l = find_nearest_mapnode(p, rotate_point_angle(direction_vector, -jc::sa), map_node)->trail;
    double trail_m = find_nearest_mapnode(p, direction_vector, map_node)->trail;
    double trail_r = find_nearest_mapnode(p, rotate_point_angle(direction_vector, jc::ra), map_node)->trail;

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
    direction_vector = rotate_point_angle(direction_vector, angle);
}

__device__ SpacePoint get_projected_vector_end(SpacePoint a, SpacePoint b, int current_face_id, Polyhedron *polyhedron)
{
    Face current_face = polyhedron->faces[current_face_id];
    for (int i = 0; i < current_face.n_of_vertices - 1; ++i)
    {
        SpacePoint intersection = line_intersection(current_face.vertices[i], current_face.vertices[i + 1], a, b);
        if (intersection != origin && is_in_segment(a, b, intersection) &&
            is_in_segment(current_face.vertices[i], current_face.vertices[i + 1], intersection) &&
            get_distance(intersection, a) > eps)
        {
            Face next_face = polyhedron->faces[find_face_next_to_edge(a, b, current_face_id, polyhedron)];
            SpacePoint normal_before = current_face.normal;
            SpacePoint normal_after = next_face.normal;
            SpacePoint moving_vector = (b - a) / get_distance(a, b);
            double phi_cos = normal_after * normal_before;
            double phi_sin = sin(acos(phi_cos));
            double alpha_cos = moving_vector * (normal_before % normal_after);

            SpacePoint faced_vector_direction = (normal_before + normal_after * phi_cos) * sin(acos(alpha_cos)) / phi_sin +
                    (normal_before % normal_after) * alpha_cos / phi_sin;

            return intersection + faced_vector_direction * (get_distance(a, b) - get_distance(intersection, a));
        }
    }
    return b;
}

__device__ int find_face_next_to_edge(SpacePoint a, SpacePoint b, int current_face_id, Polyhedron *polyhedron)
{
    for(int i = 0; i < polyhedron->n_of_faces; ++i)
        if (polyhedron->faces[i].id != current_face_id && is_edge_belongs_face(a, b, &polyhedron->faces[i]))
            return i;
    return current_face_id;
}

__device__ bool is_edge_belongs_face(SpacePoint a, SpacePoint b, const Face *const face)
{
    bool flag1 = false, flag2 = false;
    for (int i = 0; i < face->n_of_vertices; ++i)
    {
        if (face->vertices[i] == a)
            flag1 = true;
        if (face->vertices[i] == b)
            flag2 = true;
    }
    return flag1 && flag2;
}
