#ifndef MIND_S_CRAWL_SPACEPOINT_CUH
#define MIND_S_CRAWL_SPACEPOINT_CUH


/// Object describing a point in 3d space
struct SpacePoint
{
    double x, y, z;
};

/// The origin of space
__device__ SpacePoint origin = {0, 0, 0};


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

/**
 * Returns the point of lines AB and CD intersection or origin if they are parallel
 *
 * @param pointA Point belongs to the line AB
 * @param pointB Point belongs to the line AB
 * @param pointC Point belongs to the line CD
 * @param pointD Point belongs to the line CD
 *
 * @returns The point of two lines intersection or origin if they are parallel
 */
__device__ SpacePoint line_intersection(SpacePoint pointA, SpacePoint pointB, SpacePoint pointC, SpacePoint pointD);

/**
 * Checks if the point C is in segment AB
 * @param pointA Point A of segment AB
 * @param pointB Point B of segment AB
 * @param pointC Point C to check
 * @returns `true` if point C belongs to segment AB, `false` otherwise
 */
__device__ bool is_in_segment(SpacePoint pointA, SpacePoint pointB, SpacePoint pointC);


#endif //MIND_S_CRAWL_SPACEPOINT_CUH
