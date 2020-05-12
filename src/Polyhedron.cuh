#ifndef MIND_S_CRAWL_POLYHEDRON_CUH
#define MIND_S_CRAWL_POLYHEDRON_CUH

#include "SpacePoint.cuh"

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
     * @param faces Array of `Face` objects describing each face of a polyhedron
     */
    __device__ Polyhedron(SpacePoint *vertices, Face *faces);

    /**
     * Returns coordinates of a vertex by number
     *
     * @param number Number of a vertex
     *
     * @returns Vertex coordinates
     */
    __device__ SpacePoint get_vertex(int number);

    /**
     * Returns a face by number
     *
     * @param number Number of a face
     *
     * @returns Face of a polyhedron
     */
    __device__ Face get_face(int number);

private:
    /// Array of coordinates of polyhedron vertices
    SpacePoint *vertices;

    /// Array of polyhedron faces
    Face *faces;
};


#endif //MIND_S_CRAWL_POLYHEDRON_CUH
