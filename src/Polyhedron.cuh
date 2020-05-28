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
     * @param faces Array of `Face` objects describing each face of a
     * @param n_of_faces Number of polyhedron faces
     */
    __device__ Polyhedron(Face *faces, ll n_of_faces);

    /// Destructs a `Polyhedron` object
    __device__ ~Polyhedron();


    /**
     * Finds a face that the point belongs to
     *
     * @param point Point in space
     * @returns Identifier of the found face
     */
    __device__ int find_face_id_by_point(SpacePoint point) const;


    /// Array of polyhedron faces
    Face *const faces;

    /// Number of polyhedron faces
    const ll n_of_faces;

};


#endif //MIND_S_CRAWL_POLYHEDRON_CUH
