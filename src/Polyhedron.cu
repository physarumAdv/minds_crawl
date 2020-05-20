#include "Polyhedron.cuh"


template<class T>
__device__ T *allocate_and_copy(T *source, int count)
{
    T *p = new T[count];
    memcpy(p, source, count * sizeof(T));
    return p;
}

__device__ Polyhedron::Polyhedron(SpacePoint *vertices, Face *faces, ll n_of_faces, ll n_of_vertices) :
        vertices(allocate_and_copy(vertices, n_of_vertices)), faces(allocate_and_copy(faces, n_of_faces)),
        n_of_faces(n_of_faces), n_of_vertices(n_of_vertices)
{

}
