#ifndef MIND_S_CRAWL_PARTICLE_CUH
#define MIND_S_CRAWL_PARTICLE_CUH


#include "Face.cuh"
#include "Polyhedron.cuh"
#include "SpacePoint.cuh"

class MapNode;


__device__ SpacePoint get_projected_vector_end(SpacePoint a, SpacePoint b, Face *current_face, Polyhedron polyhedron);


class Polyhedron;

/// Object describing a particle in the model (also called "agent" - from the original Jones' book)
class Particle
{
public:
    /**
     * Creates a `Particle` object
     *
     * @param polyhedron The polyhedron to create particle on
     * @param polyhedron_face_id The polyhedron's face to create particle on
     * @param coordinates The coordinates to create particle at
     * @param angle Initial direction of the particle
     */
    __device__ Particle(const Polyhedron *polyhedron, int polyhedron_face_id, SpacePoint coordinates, double angle);

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

    /// The corresponding sensor's location
    SpacePoint left_sensor, middle_sensor, right_sensor;

    /// A number of the current face
    int polyhedron_face_id;

    /// A map node the particle belongs to
    MapNode *map_node;


    /// A normal to the current face
    SpacePoint normal;

private:
    /**
     * Rotates given point at the angle relative to agent's coordinates
     *
     * @param radius Vector from agent to point to rotate
     * @param angle Rotation angle
     *
     * @returns New coordinates of the point
     */
    __device__ SpacePoint rotate_point_angle(SpacePoint radius, double angle) const;

    /// Initializes left and right sensors relative to the middle sensor
    __device__ void init_left_right_sensors();


};

#endif //MIND_S_CRAWL_PARTICLE_CUH
