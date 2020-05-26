#ifndef MIND_S_CRAWL_POLYHEDRON_CUH
#define MIND_S_CRAWL_POLYHEDRON_CUH

#include "SpacePoint.cuh"
#include "Face.cuh"
#include "common.cuh"


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
    __device__ ~Polyhedron();


    __device__ int find_face_id_by_point(SpacePoint point);


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
