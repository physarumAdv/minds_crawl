#ifndef MIND_S_CRAWL_SPACEPOINT_CUH
#define MIND_S_CRAWL_SPACEPOINT_CUH


/// Object describing a point in 3d space
struct SpacePoint
{
    double x, y, z;
};


/// The origin of space
__device__ const SpacePoint origin = {0, 0, 0};

/// Observational error constant
__device__ const double eps = 1. / (100 * 100 * 100);

/**
 * Returns the result of comparison of two `SpacePoint`s
 *
 * @param a Point A in space
 * @param b Point B in space
 *
 * @returns `true` if point A coincides with point B, `false` otherwise
 */
__host__ __device__ bool operator==(SpacePoint a, SpacePoint b);

/**
 * Returns the result of comparison of two `SpacePoint`s
 *
 * @param a Point in space
 * @param b Point in space
 *
 * @returns `true` if point A does not coincide with point B, `false` otherwise
 */
__host__ __device__ bool operator!=(SpacePoint a, SpacePoint b);

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
 * Rotates point B relative to point A along the face with given normal
 *
 * @param a Point A in space
 * @param b Point B in space
 * @param normal Normal to the face
 * @param angle Rotation angle
 *
 * @returns Coordinates of point B after rotation
 */
__device__ SpacePoint relative_point_rotation(SpacePoint a, SpacePoint b, SpacePoint normal, double angle);


/**
 * Returns the distance between two points
 *
 * @param a Point in 3D
 * @param b Point in 3D
 *
 * @returns Distance between two points
 */
__device__ double get_distance(SpacePoint a, SpacePoint b);

/**
 * Returns the point of lines AB and CD intersection or origin if they are parallel
 *
 * @param a Point A belongs to the line AB
 * @param b Point B belongs to the line AB
 * @param c Point C belongs to the line CD
 * @param d Point D belongs to the line CD
 *
 * @returns The point of two lines intersection or origin if they are parallel
 */
__device__ SpacePoint line_intersection(SpacePoint a, SpacePoint b, SpacePoint c, SpacePoint d);

/**
 * Checks if the point C is in segment AB
 * @param a Point A of segment AB
 * @param b Point B of segment AB
 * @param c Point C to check
 * @returns `true` if point C belongs to segment AB, `false` otherwise
 */
__device__ bool is_in_segment(SpacePoint a, SpacePoint b, SpacePoint c);


#endif //MIND_S_CRAWL_SPACEPOINT_CUH
