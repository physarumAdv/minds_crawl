#include "Particle.cuh"
#include "Polyhedron.cuh"
#include "Face.cuh"
#include "jones_constants.hpp"

namespace jc = jones_constants;
SpacePoint origin = {0, 0, 0};

__device__ Particle::Particle(const Polyhedron *const polyhedron, int polyhedron_face_id,
                              SpacePoint coordinates, double angle)
{
    this->coordinates = coordinates;
    this->polyhedron_face_id = polyhedron_face_id;

    Face current_face = polyhedron->faces[polyhedron_face_id];
    this->normal = current_face.normal;

    SpacePoint radius = polyhedron->vertices[current_face.vertices[0]] - coordinates;
    radius = radius * jc::so / get_distance(radius, origin);
    this->middle_sensor = this->rotate_point_angle(radius, angle);
}

__device__ SpacePoint Particle::rotate_point_angle(SpacePoint radius, double angle)
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
