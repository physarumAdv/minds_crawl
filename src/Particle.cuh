#ifndef MIND_S_CRAWL_PARTICLE_CUH
#define MIND_S_CRAWL_PARTICLE_CUH

#include "SpacePoint.cuh"


class Polyhedron;

/// Object describing a particle in the model (also called "agent" - from the original Jones' book)
class Particle
{
public:
    /**
     * Creates a `Particle` object
     *
     * @param polyhedron The polyhedron to create particle on
     * @param polyhedron_face_id The polyhedron's face to create particle on
     * @param coordinates The coordinates to create particle at
     * @param angle Initial direction of the particle
     */
    __device__ Particle(const Polyhedron *polyhedron, int polyhedron_face_id, SpacePoint coordinates, double angle);

    /// The particle's location
    SpacePoint coordinates;

    /// The corresponding sensor's location
    SpacePoint left_sensor, middle_sensor, right_sensor;

    /// A number of the current face
    int polyhedron_face_id;

    /// A normal to the current face
    SpacePoint normal;

private:
    /**
     * Rotates given point at the angle relative to agent's coordinates
     *
     * @param radius Vector from agent to point to rotate
     * @param angle Rotation angle
     *
     * @returns New coordinates of the point
     */
    __device__ SpacePoint rotate_point_angle(SpacePoint radius, double angle);

    /// Initializes left and right sensors relative to the middle sensor
    __device__ void init_left_right_sensors();


};

#endif //MIND_S_CRAWL_PARTICLE_CUH
