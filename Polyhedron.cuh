#ifndef MIND_S_CRAWL_POLYHEDRON_CUH
#define MIND_S_CRAWL_POLYHEDRON_CUH

#include "MapPoint.hpp"
#include "fucking_shit.cuh"

typedef long long ll;


class Polyhedron
{
public:
    __device__ Polyhedron(...);

    //__host__ __device__ dim3 get_upper_bound_size() const;
    __host__ __device__ ll get_n_of_points() const;


    /* Both of the following arrays contain pointers to `MapPoint`s, which are on
     * the polyhedron's surface. The `points` array is a linear array just with
     * all the `MapPoint`s on a surface (their order is undefined). The `cube`
     * array is 3-dimensional, and `cube[x][y][z]` points either to MapPoint
     * with (x, y, z) coordinates (if (x, y, z) is on the polyhedron's surface),
     * or to `nullptr` (if (x, y, z) is not on the polyhedron's surface)
     */
    //MapPoint *cube; // 3-dimensional array of MapPoint *
    MapPoint *points; // 1-dimensional of MapPoint *

private:

    //dim3 cube_max;
    ll n_of_points;
};

#endif //MIND_S_CRAWL_POLYHEDRON_CUH
