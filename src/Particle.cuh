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
     * @param polyhedron The polyhedron to create particle on
     * @param map_node The polyhedron's face to create particle on
     * @param coordinates The coordinates to create particle at
     * @param angle Initial direction of the particle
     */
    __device__ Particle(MapNode *map_node, SpacePoint coordinates, double angle);

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


    /// The particle's location
    SpacePoint coordinates;

private:
    /**
     * Rotates given point at the angle relative to agent's coordinates
     *
     * @param radius Vector from agent to point to rotate
     * @param angle Rotation angle
     *
     * @returns New coordinates of the point
     */
    __device__ SpacePoint rotate_point_from_agent(SpacePoint radius, double angle) const;


    /// The direction vector of the particle's agent
    SpacePoint direction_vector;

    /// A map node the particle belongs to
    MapNode *map_node;


    /// A normal to the current face
    SpacePoint normal;


};

#endif //MIND_S_CRAWL_PARTICLE_CUH
