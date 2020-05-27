#ifndef MIND_S_CRAWL_SPACEPOINT_CUH
#define MIND_S_CRAWL_SPACEPOINT_CUH


/// Object describing a point in 3d space
struct SpacePoint
{
    double x, y, z;
};

__device__ SpacePoint origin = {0, 0, 0};
__device__ double eps = 1 / (1000 * 100 * 100);

/**
 * Returns the sum of two vectors in space
 *
 * @param a Point in space, vector
 * @param b Point in space, vector
 *
 * @returns Sum of two vectors in 3D
 */
__host__ __device__ SpacePoint operator-(SpacePoint a, SpacePoint b);

/**
 * Returns the difference of two vectors in space
 *
 * @param a Point in space, mathematical vector
 * @param b Point in space, mathematical vector
 *
 * @returns Difference of two vectors in 3D
 */
__host__ __device__ SpacePoint operator+(SpacePoint a, SpacePoint b);

/**
 * Returns the product of 3D vector by a number
 *
 * @param a Point in space, vector
 * @param b Number to multiply
 *
 * @returns Product of 3D vectors and a number
 */
__host__ __device__ SpacePoint operator*(SpacePoint a, double b);

/**
 * Returns the product of 3D vector by a number
 *
 * @overload
 */
__host__ __device__ SpacePoint operator*(double a, SpacePoint b);

/**
 * Returns the division of 3D vector by a number
 *
 * @param a Point in space, vector
 * @param b Number to multiply
 *
 * @returns Product of 3D vectors and a number
 */
__host__ __device__ SpacePoint operator/(SpacePoint a, double b);

/**
 * Returns the scalar product of two vectors
 *
 * @param a Point in space, vector
 * @param b Point in space, vector
 *
 * @returns Scalar product of two vectors in 3D
 */
__host__ __device__ double operator*(SpacePoint a, SpacePoint b);

/**
 * Returns the cross product of two vectors
 *
 * @param a Point in space, vector
 * @param b Point in space, vector
 *
 * @returns Cross product of two vectors in 3D
 */
__host__ __device__ SpacePoint operator%(SpacePoint a, SpacePoint b);


/**
 * Returns the distance between two points
 *
 * @param a Point in 3D
 * @param b Point in 3D
 *
 * @returns Distance between two points
 */
__device__ double get_distance(SpacePoint a, SpacePoint b);


#endif //MIND_S_CRAWL_SPACEPOINT_CUH
