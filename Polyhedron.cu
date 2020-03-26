#include "Polyhedron.cuh"


/*__host__ __device__ dim3 Polyhedron::get_upper_bound_size() const
{
    return this->cube_max;
}*/

/* __device__ Polyhedron::Polyhedron(...);
 * Inside the constructor we are going to allocate global
 * device memory for `points` array. Note, that after that
 * we WON'T BE ABLE to use it in host code WHATEVER we do
 * (because memory, allocated in device code, can't take
 * part in memory copy operations between host and device).
 */

__host__ __device__ ll Polyhedron::get_n_of_points() const
{
    return this->n_of_points;
}
