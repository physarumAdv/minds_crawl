#include "Polyhedron.cuh"
#include "common.cuh"


__device__ Polyhedron::Polyhedron(Face *faces, int n_of_faces) :
        faces(malloc_and_copy(faces, n_of_faces)), n_of_faces(n_of_faces)
{

}

__device__ Polyhedron &Polyhedron::operator=(const Polyhedron &other)
{
    if(this != &other)
    {
        faces = malloc_and_copy(other.faces, other.n_of_faces);
        n_of_faces = other.n_of_faces;
    }
    return *this;
}

__device__ Polyhedron::Polyhedron(const Polyhedron &other)
{
    *this = other;
}

__device__ Polyhedron &Polyhedron::operator=(Polyhedron &&other) noexcept
{
    if(this != &other)
    {
        faces = nullptr;

        swap(faces, other.faces);
        swap(n_of_faces, other.n_of_faces);
    }

    return *this;
}

__device__ Polyhedron::Polyhedron(Polyhedron &&other) noexcept
{
    *this = std::move(other);
}

__device__ Polyhedron::~Polyhedron()
{
    free((void *)faces);
}


__device__ Face *Polyhedron::find_face_by_point(SpacePoint point) const
{
    for(int i = 0; i < n_of_faces; ++i)
    {
        Face *face = &faces[i];
        SpacePoint normal = (face->get_vertices()[1] - face->get_vertices()[0]) % (point - face->get_vertices()[0]);
        normal = normal / get_distance(normal, origin);
        if(normal * face->get_normal() >= 1 - eps)
            return face;
    }
    return &faces[0];
}

__device__ Face *Polyhedron::get_faces() const
{
    return faces;
}

__device__ int Polyhedron::get_n_of_faces() const
{
    return n_of_faces;
}


__device__ bool does_edge_belong_to_face(SpacePoint a, SpacePoint b, const Face *face)
{
    bool flag1 = false, flag2 = false;
    for(int i = 0; i < face->get_n_of_vertices(); ++i)
    {
        if(face->get_vertices()[i] == a)
            flag1 = true;
        if(face->get_vertices()[i] == b)
            flag2 = true;
    }
    return flag1 && flag2;
}

__device__ Face *find_face_next_to_edge(int vertex_id, Face *current_face, Polyhedron *polyhedron)
{
    for(int i = 0; i < polyhedron->get_n_of_faces(); ++i)
        if(polyhedron->get_faces()[i] != *current_face &&
           does_edge_belong_to_face(current_face->get_vertices()[vertex_id],
                                    current_face->get_vertices()[vertex_id + 1],
                                    &polyhedron->get_faces()[i]))
            return &polyhedron->get_faces()[i];
    return current_face;
}

__device__ SpacePoint find_intersection_with_edge(SpacePoint a, SpacePoint b, Face *current_face,
                                                  int *intersection_edge)
{
    for(int i = 0; i < current_face->get_n_of_vertices() - 1; ++i)
    {
        SpacePoint intersection = line_intersection(current_face->get_vertices()[i],
                                                    current_face->get_vertices()[i + 1], a, b);
        if(intersection != origin && is_in_segment(a, b, intersection) &&
           is_in_segment(current_face->get_vertices()[i], current_face->get_vertices()[i + 1], intersection) &&
           get_distance(intersection, a) > eps)
        {
            if(intersection_edge != nullptr)
            {
                *intersection_edge = i;
            }
            return intersection;
        }
    }
    return b;
}

__device__ SpacePoint get_projected_vector_end(SpacePoint a, SpacePoint b, Face *current_face, Polyhedron *polyhedron)
{
    int intersection_edge_vertex_id = 0;
    SpacePoint intersection = find_intersection_with_edge(a, b, current_face, &intersection_edge_vertex_id);

    SpacePoint normal_before = current_face->get_normal();
    SpacePoint normal_after = find_face_next_to_edge(intersection_edge_vertex_id, current_face,
                                                     polyhedron)->get_normal();
    SpacePoint moving_vector = (b - a) / get_distance(a, b);

    double phi_cos = normal_after * normal_before;
    double phi_sin = sin(acos(phi_cos));
    double alpha_cos = moving_vector * (normal_before % normal_after);

    SpacePoint faced_vector_direction = (normal_before + normal_after * phi_cos) * sin(acos(alpha_cos)) / phi_sin +
                                        (normal_before % normal_after) * alpha_cos / phi_sin;

    // If vector AB does not intersect any edge of face, `intersection` equals `b`,
    // so `faced_vector_direction` does not affect at all
    return intersection + faced_vector_direction * (get_distance(a, b) - get_distance(intersection, a));
}
