#include "Polyhedron.cuh"

__device__ Polyhedron::Polyhedron(SpacePoint *vertices, Face *faces)
{
    this->vertices = vertices;
    this->faces = faces;
}

__device__ SpacePoint Polyhedron::get_vertex(int number)
{
    return this->vertices[number];
}

__device__ Face Polyhedron::get_face(int number)
{
    return this->faces[number];
}