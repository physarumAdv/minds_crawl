#ifndef MIND_S_CRAWL_POLYHEDRON_CUH
#define MIND_S_CRAWL_POLYHEDRON_CUH

#include "SpacePoint.cuh"
#include "common.cuh"

/**
 * Returns pointer to a copied array
 *
 * @tparam T Type of an array
 *
 * @param source Pointer to an array to copy
 * @param count Number of elements to copy
 *
 * @returns Pointer to a copied array
 */
template<class T>
__device__ T *allocate_and_copy(T *source, int count);

/// Object describing a polyhedron face
struct Face
{
    /// Array of vertices' numbers that belong to a face
    int *vertices;

    /// Normal to a face
    SpacePoint normal;
};


/// Object describing a geometric polyhedron for the simulation
class Polyhedron
{
public:
    /**
     * Creates a `Polyhedron` object
     *
     * @param vertices Array of polyhedron vertices' coordinates
     * @param faces Array of `Face` objects describing each face of a
     * @param n_of_faces Number of polyhedron faces
     * @param n_of_vertices Number of polyhedron vertices
     */
    __device__ Polyhedron(SpacePoint *vertices, Face *faces, ll n_of_faces, ll n_of_vertices);

    /// Array of coordinates of polyhedron vertices
    const SpacePoint *const vertices;

    /// Array of polyhedron faces
    const Face *const faces;

    /// Number of polyhedron faces
    const ll n_of_faces;

    /// Number of polyhedron vertices
    const ll n_of_vertices;

};


#endif //MIND_S_CRAWL_POLYHEDRON_CUH
