#ifndef MIND_S_CRAWL_MAP_POINT_H
#define MIND_S_CRAWL_MAP_POINT_H

#include "Particle.hpp"

struct MapPoint
{
    double trail = 0;
    double temp_trail = 0;
    bool contains_particle = false;
    Particle *particle = nullptr;
};

#endif //MIND_S_CRAWL_MAP_POINT_H
