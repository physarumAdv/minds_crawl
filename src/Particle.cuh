#ifndef MIND_S_CRAWL_PARTICLE_CUH
#define MIND_S_CRAWL_PARTICLE_CUH

#include "Polyhedron.cuh"
#include "SpacePoint.hpp"


/// Object describing a particle in the model
class Particle
{
public:
    /**
     * Creates a `Particle` object
     *
     * @param polyhedron The polyhedron to create particle on
     * @param polyhedron_face The polyhedron's face to create particle on
     * @param coordinates The coordinates to create particle at
     */
    __device__ Particle(const Polyhedron *const polyhedron, int polyhedron_face, SpacePoint coordinates);

    /// The particle's location
    SpacePoint coordinates;

    /// The corresponding sensor's location
    SpacePoint left_sensor, middle_sensor, right_sensor;

    /// Normal vector to the current face
    SpacePoint polyhedron_face_normal;
};

#endif //MIND_S_CRAWL_PARTICLE_CUH
