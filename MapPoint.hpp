#ifndef MIND_S_CRAWL_MAP_POINT_HPP
#define MIND_S_CRAWL_MAP_POINT_HPP

#include "Particle.cuh"


class MapPoint
{
public:
    MapPoint()
    {
        trail = temp_trail = food = x = y = z = 0;
        contains_particle = false;
        particle = nullptr;
        top = left = right = bottom = nullptr;
    }

    double trail;
    double temp_trail;
    double food;
    double x, y, z;
    MapPoint *top, *left, *right, *bottom;
    bool contains_particle;
    Particle *particle;
};

#endif //MIND_S_CRAWL_MAP_POINT_HPP
