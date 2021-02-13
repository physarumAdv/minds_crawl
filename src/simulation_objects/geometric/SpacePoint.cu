#ifdef COMPILE_FOR_CPU

#include <cmath>

#endif //COMPILE_FOR_CPU

#include "SpacePoint.cuh"


__host__ __device__ bool operator==(SpacePoint a, SpacePoint b)
{
    return (a.x == b.x && a.y == b.y && a.z == b.z);
}

__host__ __device__ bool operator!=(SpacePoint a, SpacePoint b)
{
    return !(a == b);
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

__host__ __device__ double operator^(SpacePoint a, SpacePoint b)
{
    double l_a = get_distance(a, origin), l_b = get_distance(b, origin), l_c = get_distance(b - a, origin);
    double cos_a = (l_a * l_a + l_b * l_b - l_c * l_c) / (2 * l_a * l_b);

    return l_a * l_b * sqrt(1 - cos_a * cos_a);
}


__host__ __device__ SpacePoint relative_point_rotation(SpacePoint a, SpacePoint b, SpacePoint normal, double angle)
{
    double angle_cos = cos(angle);
    SpacePoint radius = b - a;
    return (1 - angle_cos) * (normal * radius) * normal + angle_cos * radius +
           sin(angle) * (normal % radius) + a;
}


__host__ __device__ double get_distance(SpacePoint a, SpacePoint b)
{
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z));
}

__host__ __device__ bool are_lines_parallel(SpacePoint a, SpacePoint b, SpacePoint c, SpacePoint d,
                                            SpacePoint *intersection)
{
    SpacePoint direction_vectorAB = (b - a) / get_distance(b - a, origin);
    SpacePoint direction_vectorCD = (d - c) / get_distance(d - c, origin);

    SpacePoint h = direction_vectorCD % (c - a);
    SpacePoint k = direction_vectorCD % direction_vectorAB;

    double h_origin_dist = get_distance(h, origin);
    double k_origin_dist = get_distance(k, origin);

    if(h_origin_dist < eps || k_origin_dist < eps)
        return true;
    else
    {
        SpacePoint l = direction_vectorAB * h_origin_dist / k_origin_dist;
        if((h * k) / (h_origin_dist * k_origin_dist) > 0)
            *intersection = a + l;
        else
            *intersection = a - l;
        return false;
    }
}

__host__ __device__ bool is_in_segment(SpacePoint a, SpacePoint b, SpacePoint c)
{
    return (get_distance(c, a) + get_distance(c, b) - get_distance(a, b) < eps);
}
