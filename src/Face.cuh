#ifndef MIND_S_CRAWL_FACE_CUH
#define MIND_S_CRAWL_FACE_CUH


#include "SpacePoint.cuh"
#include "common.cuh"

/**
 * Returns the normal to faces defined by vertices
 *
 * @param vertices Array of vertices' numbers that belong to the same face
 *
 * @returns Normal to face
 */
__device__ SpacePoint get_normal(const SpacePoint *vertices);


class MapNode;

/// Object describing a polyhedron face
class Face
{
public:
    /**
     * Creates a `Face` object
     *
     * @param id Identifier of the polyhedron face
     * @param vertices Array of polyhedron vertices that belong to the face.
     *      Must be ordered in a special way (see note below)
     * @param n_of_vertices Number of vertices on the face
     * @param node Some node laying on the face
     *
     * @note The vertices order. Looking on a face <b>from outside</b> the polyhedron, some vertex (let's call it A)
     * must be saved to vertices[0]. It's neighbour clockwise - vertex B (B is next to A clockwise) must be saved to
     * vertices[1] and so on. Assuming there are N vertices in total, A's neighbors are B clockwise and
     * X counterclock-wise, X must be saved to vertices[N - 2], and A must be saved <b>again</b> to vertices[N - 1]
     */
    __device__ Face(int id, const SpacePoint *vertices, ll n_of_vertices, MapNode *node);

    /// Forbids copying `Face` objects
    __host__ __device__ Face(const Face &) = delete;

    /// Destructs a `Face` object
    __device__ ~Face();


    /// An identifier of the face of a `Polyhedron`
    const int id;

    /**
     * Array of vertices that belong to the face (represented as described in the constructor)
     */
    const SpacePoint *const vertices;

    /// Number of vertices on the face
    const ll n_of_vertices;

    /// Normal to the face
    const SpacePoint normal;

    /// Some node laying on the face
    MapNode *const node;
};


/**
 * Checks whether two `Face`s are same (checked using ids)
 *
 * @param a `Face`
 * @param b `Face`
 *
 * @returns `true` if two faces have same ids, `false` otherwise
 */
bool operator==(const Face &a, const Face &b);


#endif //MIND_S_CRAWL_FACE_CUH
