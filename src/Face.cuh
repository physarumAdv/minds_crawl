#ifndef MIND_S_CRAWL_FACE_CUH
#define MIND_S_CRAWL_FACE_CUH


#include "SpacePoint.cuh"


/**
 * Returns the normal to faces defined by vertices
 *
 * @param vertices Array of vertices' numbers that belong to the same face
 *
 * @returns Normal to face
 */
__host__ __device__ SpacePoint calculate_normal(const SpacePoint *vertices);


class MapNode;

class Polyhedron;

/**
 * Object describing a polyhedron face
 *
 * @warning <b>Before using the class, please, carefully read documentation for both copy and move constructors and
 *      assignment operators!</b>
 */
class Face
{
public:
    /**
     * Creates a `Face` object
     *
     * @param vertices Array of polyhedron vertices that belong to the face.
     *      Must be ordered in a special way (see note below)
     * @param n_of_vertices Number of vertices on the face
     *
     * @note The vertices order matters: looking on a face <b>from outside</b> the polyhedron, some vertex (let's call
     * it A) must be saved to `vertices[0]`. It's neighbour clockwise - vertex B (B is next to A clockwise) must be
     * saved to `vertices[1]` and so on. Assuming there are N vertices in total, A's neighbors are B clockwise and
     * X counterclockwise, X must be saved to `vertices[N - 1]`
     */
    __host__ __device__ Face(const SpacePoint *vertices, int n_of_vertices);

    /**
     * `Face` object copy assignment operator
     *
     * @warning When copying `Face` object, its `node` field is <b>not</b> copied, but is set to `nullptr`, which means
     *      the new `Face` won't have a node attached. Please, be careful
     */
    __host__ __device__ Face &operator=(const Face &other);

    /**
     * `Face` object copy constructor
     *
     * @warning When copying `Face` object, its `node` field is <b>not</b> copied, but is set to `nullptr`, which means
     *      the new `Face` won't have a node attached. Please, be careful
     */
    __host__ __device__ Face(const Face &other);

    /**
     * `Face` object move assignment operator
     *
     * @warning If the being moved `Face`'s field `node` is not `nullptr`, this might mean there is a node pointing
     *      to the `Face` being moved, so moving it will <b>invalidate</b> the pointers set at `*node`
     */
    __host__ __device__ Face &operator=(Face &&other) noexcept;

    /**
     * `Face` object move constructor
     *
     * @warning If the being moved `Face`'s field `node` is not `nullptr`, this might mean there is a node pointing
     *      to the `Face` being moved, so moving it will <b>invalidate</b> the pointers set at `*node`
     */
    __host__ __device__ Face(Face &&other) noexcept;

    /// Destructs a `Face` object
    __host__ __device__ ~Face();


    /**
     * Attaches given node to the face if the node lays on it, otherwise nothing happens
     *
     * @param node Node laying on the face
     * @param polyhedron Polyhedron in simulation
     */
    __host__ __device__ void set_node(MapNode *node, Polyhedron *polyhedron);


    /**
     * Returns a pointer to some node laying on the face
     *
     * @returns Pointer to some node laying on the face if it exists, otherwise `nullptr`
     */
    __host__ __device__ MapNode *get_node() const;

    __host__ __device__ const SpacePoint *get_vertices() const;

    __host__ __device__ int get_n_of_vertices() const;

    __host__ __device__ SpacePoint get_normal() const;


    /**
     * Checks whether two `Face`s are same (<b>and</b> have same vertices order)
     *
     * @param a `Face` to be compared
     * @param b `Face` to be compared
     *
     * @returns `true` if two faces have same vertices sets and same vertices order, `false` otherwise
     *
     * @note As mentioned in the brief description, this is not a mathematical comparison of two faces. It only returns
     *      `true` if the vertices have the same vertices order, starting from the same one
     */
    __host__ __device__ friend bool operator==(const Face &a, const Face &b);

private:
    /// Pointer to some node laying on the face
    MapNode *node;

    /// Array of vertices that belong to the face (represented as described in the constructor)
    const SpacePoint *vertices;

    /// Number of vertices on the face
    int n_of_vertices;

    /// Normal to the face
    SpacePoint normal;
};


/**
 * Checks whether two `Face`s are same (and have same vertices order)
 *
 * @param a `Face` to be compared
 * @param b `Face` to be compared
 *
 * @returns `false` if two faces have same vertices sets and same vertices order, `true` otherwise
 *
 * @note As mentioned in the brief description, this is not a mathematical comparison of two faces. It only returns
 *      `false` if the vertices have the same vertices order, starting from the same one
 */
__host__ __device__ bool operator!=(const Face &a, const Face &b);


#endif //MIND_S_CRAWL_FACE_CUH
