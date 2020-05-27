#include <initializer_list>

#include "Particle.cuh"
#include "Polyhedron.cuh"
#include "Face.cuh"
#include "jones_constants.hpp"

namespace jc = jones_constants;


__device__ Particle::Particle(const Polyhedron *polyhedron, int polyhedron_face_id,
                              SpacePoint coordinates, double angle)
{
    this->coordinates = coordinates;
    this->polyhedron_face_id = polyhedron_face_id;

    Face current_face = polyhedron->faces[polyhedron_face_id];
    this->normal = current_face.normal;

    SpacePoint radius = current_face.vertices[0] - coordinates;
    radius = radius * jc::so / get_distance(radius, origin);
    this->middle_sensor = this->rotate_point_angle(radius, angle);
    this->init_left_right_sensors();
}


__device__ SpacePoint Particle::rotate_point_angle(SpacePoint radius, double angle) const
{
    double angle_cos = cos(angle);
    return (1 - angle_cos) * (this->normal * radius) * this->normal + angle_cos * radius +
                      sin(angle) * (this->normal % radius) + this->coordinates;
}

__device__ void Particle::init_left_right_sensors()
{
    this->left_sensor = this->rotate_point_angle(this->middle_sensor - this->coordinates, jc::sa);
    this->right_sensor = this->rotate_point_angle(this->middle_sensor - this->coordinates, -jc::sa);
    if(this->normal * ((this->right_sensor - this->coordinates) % (this->left_sensor - this->coordinates)) < 0)
    {
        SpacePoint p = this->right_sensor;
        this->right_sensor = this->left_sensor;
        this->left_sensor = p;
    }
}


__device__ void Particle::do_sensory_behaviours()
{
    double trail_l = find_nearest_mapnode(left_sensor, map_node)->trail;
    double trail_m = find_nearest_mapnode(middle_sensor, map_node)->trail;
    double trail_r = find_nearest_mapnode(right_sensor, map_node)->trail;

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
    left_sensor = rotate_point_angle(left_sensor, angle);
    middle_sensor = rotate_point_angle(middle_sensor, angle);
    right_sensor = rotate_point_angle(right_sensor, angle);
}
