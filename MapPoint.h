#ifndef MIND_S_CRAWL_MAP_POINT_H
#define MIND_S_CRAWL_MAP_POINT_H

#include "Particle.h"

struct MapPoint
{
    double trail = 0;
    double temp_trail = 0;
    bool contains_particle = false;
    Particle *p;
};

#endif //MIND_S_CRAWL_MAP_POINT_H
