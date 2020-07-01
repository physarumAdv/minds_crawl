#include "Particle.cuh"
#include "fucking_shit.cuh"
#include "jones_constants.hpp"

namespace jc = jones_constants;


__device__ Particle::Particle(MapNode *map_node, SpacePoint coordinates, double angle) :
        coordinates(coordinates), map_node(map_node)
{
    Face &current_face = map_node->get_polyhedron()->faces[map_node->get_face_id()];
    normal = current_face.normal;

    SpacePoint radius = current_face.vertices[0] - coordinates;
    radius = radius * jc::speed / get_distance(radius, origin);
    direction_vector = rotate_point_from_agent(radius, angle);
}


__device__ SpacePoint Particle::rotate_point_from_agent(SpacePoint radius, double angle) const
{
    return relative_point_rotation(coordinates, coordinates + radius, normal, angle);
}


__device__ void Particle::do_sensory_behaviours()
{
    Polyhedron *p = map_node->get_polyhedron();

    double trail_l = find_nearest_mapnode(p, rotate_point_from_agent(direction_vector, -jc::sa), map_node)->trail;
    double trail_m = find_nearest_mapnode(p, direction_vector, map_node)->trail;
    double trail_r = find_nearest_mapnode(p, rotate_point_from_agent(direction_vector, jc::ra), map_node)->trail;

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
