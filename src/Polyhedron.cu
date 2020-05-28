#include "Polyhedron.cuh"


__device__ Polyhedron::Polyhedron(Face *faces, int n_of_faces) :
        faces(malloc_and_copy(faces, n_of_faces)), n_of_faces(n_of_faces)
{

}

__device__ Polyhedron::~Polyhedron()
{
    free((void *)faces);
}

__device__ int Polyhedron::find_face_id_by_point(SpacePoint point) const
{
    for (int i = 0; i < n_of_faces; ++i)
    {
        Face &face = faces[i];
        SpacePoint normal = (face.vertices[1] - face.vertices[0]) % (point - face.vertices[0]);
        normal = normal / get_distance(normal, origin);
        if (normal * face.normal >= 1 - eps)
            return face.id;
    }
    return faces[0].id;
}
