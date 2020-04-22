#ifndef MIND_S_CRAWL_MAPNODE_HPP
#define MIND_S_CRAWL_MAPNODE_HPP

#include "Particle.cuh"


class MapNode
{
public:
    MapNode()
    {
        trail = temp_trail = food = x = y = z = 0;
        contains_particle = false;
        particle = nullptr;
        top = left = right = bottom = nullptr;
        polyhedron_face_number = -1;
    }

    double trail;
    double temp_trail;
    double food;
    double x, y, z;
    MapNode *top, *left, *right, *bottom;
    bool contains_particle;
    Particle *particle;

    /** The `polyhedron_face_number` is either a number of the polyhedron's face
     * it is placed on, or -1 if all the node's neighbours are on the same face
     */
    int polyhedron_face_number;
};

#endif //MIND_S_CRAWL_MAPNODE_HPP
