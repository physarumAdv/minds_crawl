#ifndef MIND_S_CRAWL_PARTICLE_CUH
#define MIND_S_CRAWL_PARTICLE_CUH


#include "Face.cuh"
#include "Polyhedron.cuh"
#include "SpacePoint.cuh"

class MapNode;


/// Object describing a particle in the model (also called "agent" - from the original Jones' book)
class Particle
{
public:
    /**
     * Creates a `Particle` object
     *
     * @param map_node The polyhedron's face to create particle on
     * @param angle Initial direction of the particle
     */
    __device__ Particle(MapNode *map_node, double angle);

    /// Forbids copying `Particle` objects
    __host__ __device__ Particle(const Particle &) = delete;


    /**
     * Moves particle in the direction in which the particle is rotated
     *
     * Tries to move the particle in the direction set by `direction_vector` (correctly handles polyhedron edge
     * crossing). If the target `MapNode` contains a particle already, nothing happens. Otherwise the particle is being
     * moved and reattached to that `MapNode`
     */
    __device__ void do_motor_behaviours();

    /**
     * Rotates particle in the current plane based on amount of trail under sensors
     *
     * If the maximum amount of trail is on the left sensor, particle rotates to the left on `jc::ra` radians;
     * if the maximum amount of trail is on the right sensor, particle rotates to the right on `jc::ra` radians;
     * if the maximum amount of trail is on the middle sensor, particle does not rotate
     *
     * @see Particle::rotate
     */
    __device__ void do_sensory_behaviours();

    /**
     * Rotates the particle by angle in the current plane
     *
     * @param angle The angle (in radians) to be rotated by
     */
    __device__ void rotate(double angle);


    /**
     * Tries to mark the `Particle` as "captured" by a thread while running an iteration
     *
     * Before trying to process a `Particle`, a thread should capture it (if it didn't succeed, this means another
     * thread is processing the `Particle` already). After processing, a `Particle` should be released, see the
     *
     * @returns `true` if the particle was successfully captured and the captured thread is allowed to process it,
     *      otherwise `false`
     *
     * @note This operation is thread-safe
     *
     * @see Particle::release
     */
    [[nodiscard]] __device__ bool capture();

    /**
     * Releases a captured particle
     *
     * Releases a particle to allow capturing it again. The method should <b>only</b> be called when all the threads
     * which could try to capture the `Particle` are finished
     *
     * @see Particle::capture
     */
    __device__ void release();


    /// The particle's location in space
    SpacePoint coordinates;

private:
    /**
     * Rotates given point at the angle relative to agent's coordinates and projects the point on polyhedron
     *
     * @param radius Vector from agent to a point to rotate
     * @param angle Rotation angle
     * @param do_projection If `true`, rotated point will be projected on polyhedron, otherwise it will not
     *
     * @returns New coordinates of the point
     */
    __device__ SpacePoint rotate_point_from_agent(SpacePoint radius, double angle, bool do_projection) const;


    /// Direction <b>vector</b> of the particle's agent
    SpacePoint direction_vector;

    /// Pointer to a map node the particle belongs to
    MapNode *map_node;


    /// Normal to the current face
    SpacePoint normal;


    /**
     * Flag showing whether the particle was processed during current iteration (should be changed with `capture` and
     * `release` methods)
     */
    bool is_captured = false;
};

#endif //MIND_S_CRAWL_PARTICLE_CUH
