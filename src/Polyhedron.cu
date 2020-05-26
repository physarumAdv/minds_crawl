#include "Polyhedron.cuh"
#include "Face.cuh"


__device__ Polyhedron::Polyhedron(SpacePoint *vertices, Face *faces, ll n_of_faces, ll n_of_vertices) :
        vertices(malloc_and_copy(vertices, n_of_vertices)), faces(malloc_and_copy(faces, n_of_faces)),
        n_of_faces(n_of_faces), n_of_vertices(n_of_vertices)
{

}

__device__ Polyhedron::~Polyhedron()
{
    free((void *)vertices);
}
