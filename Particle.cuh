#ifndef MIND_S_CRAWL_PARTICLE_CUH
#define MIND_S_CRAWL_PARTICLE_CUH

#include "Polyhedron.cuh"

struct SpacePoint {
    double x, y, z;
};

class Particle
{
public:
    Particle(SpacePoint me, double angle, Polyhedron *polyhedron, int polyhedron_face);

    SpacePoint me;
    SpacePoint left_sensor, middle_sensor, right_sensor;
    SpacePoint polyhedron_face_normal;
};

#endif //MIND_S_CRAWL_PARTICLE_CUH
