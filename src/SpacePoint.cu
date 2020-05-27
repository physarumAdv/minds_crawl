#include "SpacePoint.cuh"


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

__device__ SpacePoint line_intersection(SpacePoint pointA, SpacePoint pointB, SpacePoint pointC, SpacePoint pointD)
{
    SpacePoint direction_vectorAB = (pointB - pointA) / get_distance(pointB - pointA, origin);
    SpacePoint direction_vectorCD = (pointD - pointC) / get_distance(pointD - pointC, origin);

    SpacePoint h = direction_vectorCD % (pointC - pointA);
    SpacePoint k = direction_vectorCD % direction_vectorAB;

    double h_origin_dist = get_distance(h, origin);
    double k_origin_dist = get_distance(k, origin);

    if (h_origin_dist < eps || k_origin_dist < eps)
        return origin;
    else
    {
        SpacePoint l = direction_vectorAB * h_origin_dist / k_origin_dist;
        if ((h * k) / (h_origin_dist * k_origin_dist) > 0)
            return pointA + l;
        else
            return pointA - l;
    }
}

__device__ bool is_in_segment(SpacePoint pointA, SpacePoint pointB, SpacePoint pointC)
{
    return (get_distance(pointC, pointA) + get_distance(pointC, pointB) - get_distance(pointA, pointB) < eps);
}
