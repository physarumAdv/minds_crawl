#ifndef MIND_S_CRAWL_PARTICLE_HPP
#define MIND_S_CRAWL_PARTICLE_HPP

struct SpacePoint {
    double x, y, z;
};

struct Particle
{
    // TODO: Make Particle a class with constructor initializing fields
    SpacePoint me;
    SpacePoint left_sensor, middle_sensor, right_sensor;
};

#endif //MIND_S_CRAWL_PARTICLE_HPP
