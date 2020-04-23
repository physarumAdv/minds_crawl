#ifndef MIND_S_CRAWL_MAPNODE_CUH
#define MIND_S_CRAWL_MAPNODE_CUH

#include "Particle.cuh"


class MapNode
{
public:
    __device__ MapNode(Polyhedron *polyhedron, int polyhedron_face, SpacePoint coordinates);

    Polyhedron *const polyhedron;

    double trail;
    double temp_trail;
    double food;
    SpacePoint coordinates;
    MapNode *top, *left, *right, *bottom;
    bool contains_particle;
    Particle *particle;

    int polyhedron_face;
    bool is_on_edge;
};

#endif //MIND_S_CRAWL_MAPNODE_CUH
