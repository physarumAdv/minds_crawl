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
    __device__ Polyhedron(Face *faces, int n_of_faces);

    /// Forbids copying `Polyhedron` objects
    __host__ __device__ Polyhedron(const Polyhedron &) = delete;

    /// Destructs a `Polyhedron` object
    __device__ ~Polyhedron();


    /**
     * Finds a face that the point belongs to
     *
     * @param point Point in space
     *
     * @returns Identifier of the found face
     */
    __device__ int find_face_id_by_point(SpacePoint point) const;


    /// Array of polyhedron faces
    Face *const faces;

    /// Number of polyhedron faces
    const int n_of_faces;

};


/**
 * Returns the coordinates of the end of AB vector's overlay on the polyhedron's surface
 *
 * If vector AB completely lies on the current face of polyhedron, coordinates of point B are returned
 * If vector AB crosses the edge of current face, the end of overlay on the next face to edge is returned
 *
 * @param a The beginning of vector AB, point A
 * @param b The end of vector AB, point B
 * @param current_face_id Identifier of the face point A belongs to
 * @param polyhedron The polyhedron in simulation
 *
 * @returns Coordinates of the end of AB vector's overlay
 */
__device__ SpacePoint get_projected_vector_end(SpacePoint a, SpacePoint b, int current_face_id, Polyhedron *polyhedron);

/**
 * Returns the intersection point of segment AB with an edge of given face if it exists, point B otherwise
 *
 * @param a Point A of segment AB
 * @param b Point B of segment AB
 * @param current_face Face the segment AB belongs to
 * @param intersection_edge Pointer to a variable in which to save index of vertex of edge that intersect segment AB
 *                          in `Face::vertices` array
 *
 * @note `intersection_edge` will not be changed if segment AB does not intersect any edge of face
 *
 * @returns Point of intersection with edge if it exists, point B otherwise
 */
__device__ SpacePoint find_intersection_with_edge(SpacePoint a, SpacePoint b, Face *current_face,
                                                  int *intersection_edge);

/**
 * Finds a face adjacent to the given face along the edge represented by vertices
 * with indexes `vertex_id` and `vertex_id + 1 in `Face::vertices` array
 *
 * @param vertex_id Index of edge vertex in `Face::vertices` array
 * @param current_face_id Identifier of given face
 * @param polyhedron The polyhedron in simulation
 *
 * @returns Identifier of the found face
 */
__device__ int find_face_next_to_edge(int vertex_id, int current_face_id, Polyhedron *polyhedron);

/**
 * Returns whether the edge's vertices belong to face
 *
 * @param a Point A of edge AB
 * @param b Point B of edge AB
 * @param face `Face`
 *
 * @returns `true` if edge AB belongs to face, `false` otherwise
 */
__device__ bool does_edge_belong_to_face(SpacePoint a, SpacePoint b, const Face *face);


#endif //MIND_S_CRAWL_POLYHEDRON_CUH
