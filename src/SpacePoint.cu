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

__host__ __device__ double operator*(SpacePoint a, SpacePoint b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ SpacePoint operator%(SpacePoint a, SpacePoint b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}
}