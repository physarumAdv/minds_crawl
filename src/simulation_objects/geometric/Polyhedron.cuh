#ifndef MINDS_CRAWL_POLYHEDRON_CUH
#define MINDS_CRAWL_POLYHEDRON_CUH


#include "SpacePoint.cuh"
#include "Face.cuh"


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
    __host__ __device__ Polyhedron(Face *faces, int n_of_faces);

    /**
     * `Polyhedron` object copy assignment operator
     *
     * @warning Copying `Polyhedron` object requires copying `Face`s inside it, which can sometimes lead to unexpected
     *      results (different from when moving!). For details check the documentation of `Face` copy assignment
     *      operator
     */
    __host__ __device__ Polyhedron &operator=(const Polyhedron &other);

    /**
     * `Polyhedron` object copy constructor
     *
     * @warning Copying `Polyhedron` object requires copying `Face`s inside it, which can sometimes lead to unexpected
     *      results (different from when moving!). For details check the documentation of `Face` copy constructor
     */
    __host__ __device__ Polyhedron(const Polyhedron &other);

    /**
     * `Polyhedron` object move assignment operator
     *
     * @warning Moving `Polyhedron` object requires moving `Face`s inside it, which can sometimes lead to unexpected
     *      results (different from when copying!). For details check the documentation of `Face` move assignment
     *      operator
     */
    __host__ __device__ Polyhedron &operator=(Polyhedron &&other) noexcept;

    /**
     * `Polyhedron` object move constructor
     *
     * @warning Moving `Polyhedron` object requires moving `Face`s inside it, which can sometimes lead to unexpected
     *      results (different from when copying!). For details check the documentation of `Face` move constructor
     */
    __host__ __device__ Polyhedron(Polyhedron &&other) noexcept;

    /**
     * Creates a <b>fake</b> `Polyhedron` object
     *
     * Creates a `Polyhedron` object such that it can be destructed safely, but must not be used for any purposes. All
     * the fields, which don't take part in the destruction, stay uninitialized.
     *
     * @warning The behaviour is undefined if you use the object constructed with this constructor in any way but
     *      destructing it (the object can be destructed safely)
     */
    __host__ __device__ Polyhedron();

    /// Destructs a `Polyhedron` object
    __host__ __device__ ~Polyhedron();


    /**
     * Resets `Polyhedron` to a state when it is guaranteed to destruct safely, even if the object wasn't ever
     * constructed (you shouldn't use this method in any other situations)
     *
     * @warning Calling this method for an object, constructed in a usual way, will result in a memory leak. It can only
     *      be used for objects, allocated in a c-like way
     */
    __host__ __device__ void _reset_destructively();


    /**
     * Finds a face the given point belongs to
     *
     * @param point Point in space
     *
     * @returns Pointer to the found face
     */
    __host__ __device__ Face *find_face_by_point(SpacePoint point) const;

    __host__ __device__ Face *get_faces() const;

    __host__ __device__ int get_n_of_faces() const;


    /**
     * Calculates square of polyhedron surface using signed areas
     *
     * @note Supports all types of polyhedron
     *
     * @returns Square of polyhedron surface
     */
    __host__ __device__ double calculate_square_of_surface();


private:
    /// Pointer-represented array of polyhedron faces
    Face *faces;

    /// Number of polyhedron faces
    int n_of_faces;

};


/**
 * Returns whether the edge's vertices belong to face
 *
 * @param a Point A of edge AB
 * @param b Point B of edge AB
 * @param face `Face`
 *
 * @returns `true` if edge AB belongs to face, `false` otherwise
 */
__host__ __device__ bool does_edge_belong_to_face(SpacePoint a, SpacePoint b, const Face *face);

/**
 * Finds a face adjacent to the given face along the edge represented by vertices
 * with indexes `vertex_id` and `(vertex_id + 1) % current_face->get_n_of_vertices()` in `Face::vertices` array
 *
 * @param vertex_id Index of edge vertex in `Face::vertices` array
 * @param current_face Pointer to the face to search next to
 * @param polyhedron The polyhedron in simulation
 *
 * @returns Pointer to the found face
 */
__host__ __device__ Face *find_face_next_to_edge(int vertex_id, Face *current_face, Polyhedron *polyhedron);

/**
 * Returns the intersection point of segment AB with an edge of given face if it exists, point B otherwise
 *
 * @param a Point A of segment AB
 * @param b Point B of segment AB
 * @param current_face Face the segment AB belongs to
 * @param intersection_edge (optional) Pointer to a variable in which to save index of vertex of edge
 * that intersect segment AB in `Face::vertices` array
 *
 * @note `intersection_edge` will not be changed if segment AB does not intersect any edge of face
 *
 * @returns Point of intersection with edge if it exists, point B otherwise
 */
__host__ __device__ SpacePoint find_intersection_with_edge(SpacePoint a, SpacePoint b, Face *current_face,
                                                           int *intersection_edge = nullptr);

/**
 * Returns the coordinates of the end of AB vector's overlay on the polyhedron's surface
 *
 * If vector AB completely lies on the current face of polyhedron, coordinates of point B are returned
 * If vector AB crosses the edge of current face, the end of overlay on the next face to edge is returned
 *
 * @param vector_start The beginning of vector AB, point A
 * @param vector_end The end of vector AB, point B
 * @param current_face Pointer to the face point A belongs to
 * @param polyhedron The polyhedron in simulation
 *
 * @returns Coordinates of the end of AB vector's overlay
 */
__host__ __device__ SpacePoint get_projected_vector_end(SpacePoint vector_start, SpacePoint vector_end,
                                                        Face *current_face, Polyhedron *polyhedron);


#endif //MINDS_CRAWL_POLYHEDRON_CUH
