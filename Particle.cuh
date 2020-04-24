#ifndef MIND_S_CRAWL_PARTICLE_CUH
#define MIND_S_CRAWL_PARTICLE_CUH

#include "Polyhedron.cuh"
#include "SpacePoint.hpp"


/// Object describing a particle in the model (also called "agent" - from the original Jones' book)
class Particle
{
public:
    /**
     * Creates a `Particle` object
     *
     * @param coordinates The coordinates to create particle at
     * @param polyhedron The polyhedron to create particle on
     * @param polyhedron_face The polyhedron's face to create particle on
     */
    __device__ Particle(SpacePoint coordinates, Polyhedron *polyhedron, int polyhedron_face);

    /// The particle's location
    SpacePoint coordinates;

    /// The corresponding sensor's location
    SpacePoint left_sensor, middle_sensor, right_sensor;

    /// Normal vector to the current face
    SpacePoint polyhedron_face_normal;
};

#endif //MIND_S_CRAWL_PARTICLE_CUH
