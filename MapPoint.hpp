#ifndef MIND_S_CRAWL_MAP_POINT_HPP
#define MIND_S_CRAWL_MAP_POINT_HPP

#include "Particle.hpp"


class MapPoint
{
public:
    MapPoint()
    {
        trail = temp_trail = food = x = y = z = 0;
        contains_particle = false;
        particle = nullptr;
        top_left = top = top_right = left = right =
                bottom_left = bottom = bottom_right = nullptr;
    }

    double trail;
    double temp_trail;
    double food;
    double x, y, z;
    MapPoint *top_left, *top, *top_right, *left, *right,
            *bottom_left, *bottom, *bottom_right;
    bool contains_particle;
    Particle *particle;
};

#endif //MIND_S_CRAWL_MAP_POINT_HPP
