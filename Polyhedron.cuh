#ifndef MIND_S_CRAWL_POLYHEDRON_CUH
#define MIND_S_CRAWL_POLYHEDRON_CUH

#include "SpacePoint.hpp"

/// Object describing a polyhedron face
struct Face
{
    /// Array of vertice's numbers that belong to the face
    int *vectices;

    /// Normal to face
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
     * @param faces Array of `Face` objects describing each face of polyhedron
     */
    __device__ Polyhedron(SpacePoint *vertices, Face *faces);

    /**
     * Returns coordinates of vertice with the number
     *
     * @param number Number of vertice
     *
     * @returns vertice coordinates
     */
    __device__ SpacePoint get_vertice(int number);

    /**
     * Returns face with the number
     *
     * @param number Number of face
     *
     * @returns face of polyhedron
     */
    __device__ Face get_face(int number);

private:
    /// Array of coordinates of polyhedron vertices
    SpacePoint *vertices;

    /// Array of polyhedron faces
    Face *faces;
};


#endif //MIND_S_CRAWL_POLYHEDRON_CUH
