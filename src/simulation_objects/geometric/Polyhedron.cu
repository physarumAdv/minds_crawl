#ifdef COMPILE_FOR_CPU
#include <cmath>
#endif //COMPILE_FOR_CPU

#include <cstdlib>
#include <utility>

#include "Polyhedron.cuh"
#include "../../common.cuh"


__host__ __device__ Polyhedron::Polyhedron(Face *faces, int n_of_faces) :
        faces(malloc_and_copy(faces, n_of_faces)), n_of_faces(n_of_faces)
{

}

__host__ __device__ Polyhedron &Polyhedron::operator=(const Polyhedron &other)
{
    if(this != &other)
    {
        faces = malloc_and_copy(other.faces, other.n_of_faces);
        n_of_faces = other.n_of_faces;
    }
    return *this;
}

__host__ __device__ Polyhedron::Polyhedron(const Polyhedron &other)
{
    *this = other;
}

__host__ __device__ Polyhedron &Polyhedron::operator=(Polyhedron &&other) noexcept
{
    if(this != &other)
    {
        swap(faces, other.faces);
        swap(n_of_faces, other.n_of_faces);
    }

    return *this;
}

__host__ __device__ Polyhedron::Polyhedron(Polyhedron &&other) noexcept
{
    faces = nullptr;

    *this = std::move(other);
}

__host__ __device__ Polyhedron::~Polyhedron()
{
    free((void *)faces);
}


__host__ __device__ Face *Polyhedron::find_face_by_point(SpacePoint point) const
{
    for(int i = 0; i < n_of_faces; ++i)
    {
        Face *face = &faces[i];
        SpacePoint normal = (point - face->get_vertices()[0]) % (face->get_vertices()[1] - face->get_vertices()[0]);
        normal = normal / get_distance(normal, origin);
        if(normal * face->get_normal() >= 1 - eps)
            return face;
    }
    return &faces[0];
}

__host__ __device__ Face *Polyhedron::get_faces() const
{
    return faces;
}

__host__ __device__ int Polyhedron::get_n_of_faces() const
{
    return n_of_faces;
}


__host__ __device__ double Polyhedron::calculate_square_of_surface()
{
    double square = 0;
    for(int i = 0; i < n_of_faces; ++i)
    {
        // Cause first vertex of face repeats again in the end the condition is `j < faces[i].get_n_of_vertices() - 2`
        for(int j = 1; j < faces[i].get_n_of_vertices() - 1; ++j)
        {
            SpacePoint a = faces[i].get_vertices()[j + 1] - faces[i].get_vertices()[0];
            SpacePoint b = faces[i].get_vertices()[j] - faces[i].get_vertices()[0];

            double sign_of_square = (a % b) * faces[i].get_normal();
            sign_of_square /= abs(sign_of_square);
            square += sign_of_square * (a ^ b) / 2;
        }
    }
    return square;
}


__host__ __device__ bool does_edge_belong_to_face(SpacePoint a, SpacePoint b, const Face *face)
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

__host__ __device__ Face *find_face_next_to_edge(int vertex_id, Face *current_face, Polyhedron *polyhedron)
{
    const SpacePoint *point_a = current_face->get_vertices() + vertex_id;
    const SpacePoint *point_b = point_a + 1; // By default it's just the next vertex
    if(vertex_id + 1 == current_face->get_n_of_vertices()) // But if `point_a` was the last one
        point_b = current_face->get_vertices(); // Then `point_b` should be the first one

    for(int i = 0; i < polyhedron->get_n_of_faces(); ++i)
        if(polyhedron->get_faces()[i] != *current_face &&
                does_edge_belong_to_face(*point_a, *point_b, &polyhedron->get_faces()[i]))
            return &polyhedron->get_faces()[i];
    return current_face;
}

__host__ __device__ SpacePoint find_intersection_with_edge(SpacePoint a, SpacePoint b, Face *current_face,
                                                           int *intersection_edge)
{
    int n_of_vertices = current_face->get_n_of_vertices();

    for(int i = 0; i < n_of_vertices; ++i)
    {
        SpacePoint intersection = origin;
        bool are_parallel = are_lines_parallel(current_face->get_vertices()[i],
                                               current_face->get_vertices()[(i + 1) % n_of_vertices],
                                               a, b, &intersection);
        if(!are_parallel && is_in_segment(a, b, intersection) &&
           is_in_segment(current_face->get_vertices()[i], current_face->get_vertices()[(i + 1) % n_of_vertices],
                         intersection) && get_distance(intersection, a) > eps)
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

__host__ __device__ SpacePoint get_projected_vector_end(SpacePoint vector_start, SpacePoint vector_end,
                                                        Face *current_face, Polyhedron *polyhedron)
{
    int intersection_edge_vertex_id = 0;
    SpacePoint intersection = find_intersection_with_edge(vector_start, vector_end, current_face,
                                                          &intersection_edge_vertex_id);
    while(intersection != vector_end)
    {
        SpacePoint normal_before = (-1) * current_face->get_normal();
        SpacePoint normal_after = (-1) * find_face_next_to_edge(intersection_edge_vertex_id, current_face,
                                                                polyhedron)->get_normal();
        SpacePoint moving_vector = (vector_end - vector_start) / get_distance(vector_start, vector_end);

        double phi_cos = (-1) * normal_after * normal_before;
        double phi_sin = sin(acos(phi_cos));
        double alpha_cos = moving_vector * (normal_before % normal_after);

        SpacePoint faced_vector_direction = (normal_before + normal_after * phi_cos) * sin(acos(alpha_cos)) / phi_sin +
                                            (normal_before % normal_after) * alpha_cos / phi_sin;

        vector_end = intersection + faced_vector_direction * (get_distance(vector_start, vector_end) -
                                                              get_distance(intersection, vector_start));
        vector_start = intersection;
        current_face = find_face_next_to_edge(intersection_edge_vertex_id, current_face, polyhedron);
        intersection = find_intersection_with_edge(vector_start + (vector_end - vector_start) * eps,
                                                   vector_end, current_face, &intersection_edge_vertex_id);
    }

    // If vector AB does not intersect any edge of face, `vector_end` equals `b`
    return vector_end;
}
