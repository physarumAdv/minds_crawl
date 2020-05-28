#include "SpacePoint.cuh"


/// Observational error constant
__device__ const double eps = 1. / (1000 * 100 * 100);


__host__ __device__ bool operator!=(SpacePoint a, SpacePoint b)
{
    return (a.x != b.x || a.y != b.y || a.z != b.z);
}

__host__ __device__ SpacePoint operator-(SpacePoint a, SpacePoint b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ SpacePoint operator+(SpacePoint a, SpacePoint b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ SpacePoint operator*(SpacePoint a, double b)
{
    return {a.x * b, a.y * b, a.z * b};
}

__host__ __device__ SpacePoint operator*(double a, SpacePoint b)
{
    return b * a;
}

__host__ __device__ SpacePoint operator/(SpacePoint a, double b)
{
    return {a.x / b, a.y / b, a.z / b};
}

__host__ __device__ double operator*(SpacePoint a, SpacePoint b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ SpacePoint operator%(SpacePoint a, SpacePoint b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}


__device__ double get_distance(SpacePoint a, SpacePoint b)
{
    return sqrt((double) ((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z)));
}

__device__ SpacePoint line_intersection(SpacePoint a, SpacePoint b, SpacePoint c, SpacePoint d)
{
    SpacePoint direction_vectorAB = (b - a) / get_distance(b - a, origin);
    SpacePoint direction_vectorCD = (d - c) / get_distance(d - c, origin);

    SpacePoint h = direction_vectorCD % (c - a);
    SpacePoint k = direction_vectorCD % direction_vectorAB;

    double h_origin_dist = get_distance(h, origin);
    double k_origin_dist = get_distance(k, origin);

    if (h_origin_dist < eps || k_origin_dist < eps)
        return origin;
    else
    {
        SpacePoint l = direction_vectorAB * h_origin_dist / k_origin_dist;
        if ((h * k) / (h_origin_dist * k_origin_dist) > 0)
            return a + l;
        else
            return a - l;
    }
}

__device__ bool is_in_segment(SpacePoint a, SpacePoint b, SpacePoint c)
{
    return (get_distance(c, a) + get_distance(c, b) - get_distance(a, b) < eps);
}
