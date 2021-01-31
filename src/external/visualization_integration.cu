#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "visualization_integration.cuh"
#include "../simulation_objects/Particle.cuh"


__host__ std::pair<std::string, std::string> get_visualization_endpoints()
{
    std::ifstream f("config/visualization_endpoint.txt", std::ios::in);
    std::string particles_url, poly_url;
    getline(f, particles_url);
    getline(f, poly_url);
    return {particles_url, poly_url};
}

template<class T>
__host__ std::string to_string_extended(const T &v)
{
    return std::to_string(v);
}

template<class T>
__host__ std::string to_string_extended(const std::vector<T> &v)
{
    std::string body = "[";
    for(int i = 0; i < v.size(); ++i)
    {
        body += to_string_extended(v[i]) + ',';
    }
    body.back() = ']';

    return body;
}

__host__ bool send_poly_to_visualization(const std::pair<std::string, std::string> &urls, const Polyhedron *polyhedron)
{
    std::vector<double> polyhedron_vertices, polyhedron_faces;
    std::vector<std::vector<double>> poly;

    int n_of_faces = polyhedron->get_n_of_faces();
    Face *faces = polyhedron->get_faces();


    for(int i = 0; i < n_of_faces; ++i)
    {
        const SpacePoint *current_vertices_of_face = faces[i].get_vertices();
        for(int j = 0; j < faces[i].get_n_of_vertices(); ++j)
        {
            SpacePoint v = current_vertices_of_face[j];
            poly.push_back({v.x, v.y, v.z});
        }
    }

    std::string body = to_string_extended(poly);

    http::Request poly_request(urls.second);

    try
    {
        const http::Response response = poly_request.send("POST", body, {"Content-Type: application/json"});

        if(response.status != 200)
            throw http::ResponseError("Response status is not OK");
    }
    catch(const std::exception &e)
    {
        std::cerr << "Request failed, error: " << e.what() << std::endl;
        return false;
    }

    return true;
}

__host__ bool send_particles_to_visualization(const std::pair<std::string, std::string> &urls, const MapNode *nodes,
                                              int n_of_nodes)
{
    std::vector<double> x, y, z;
    x.reserve(n_of_nodes);
    y.reserve(n_of_nodes);
    z.reserve(n_of_nodes);

    for(int i = 0; i < n_of_nodes; ++i)
    {
        if(!nodes[i].does_contain_particle())
            continue;

        Particle *p = nodes[i].get_particle();
        x.push_back(p->coordinates.x);
        y.push_back(p->coordinates.y);
        z.push_back(p->coordinates.z);
    }

    std::string body = "{";
    body += "\"x\":" + to_string_extended(x) +
            ",\"y\":" + to_string_extended(y) +
            ",\"z\":" + to_string_extended(z) + "}";


    http::Request particles_request(urls.first);

    try
    {
        const http::Response response = particles_request.send("POST", body, {"Content-Type: application/json"});

        if(response.status != 200)
            throw http::ResponseError("Response status is not OK");
    }
    catch(const std::exception &e)
    {
        std::cerr << "Request failed, error: " << e.what() << std::endl;
        return false;
    }

    return true;
}
