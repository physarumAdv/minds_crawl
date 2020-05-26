#ifndef MIND_S_CRAWL_FACE_CUH
#define MIND_S_CRAWL_FACE_CUH

#include "SpacePoint.cuh"


__device__ SpacePoint get_normal(const int *vertices, int n_of_vertices);


class MapNode;

/// Object describing a polyhedron face
class Face
{
public:
    __device__ Face(int id, const int *vertices, int n_of_vertices, MapNode *node);
    __device__ ~Face();


    /// An identifier of the face of a `Polyhedron`
    const int id;


    /// Array of vertices' numbers that belong to the face
    const int *vertices;

    /// Number of vertices on the face
    const int n_of_vertices;

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
