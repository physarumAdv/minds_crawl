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

__host__ std::string vector_double_to_json_array(const std::vector<double> &v)
{
    if(v.empty())
        return "[]";

    std::string ans = "[";
    for(double i : v)
    {
        ans += std::to_string(i) + ',';
    }
    ans.back() = ']';

    return ans;
}

__host__ bool send_model_to_visualization(const std::pair<std::string, std::string> &urls, Polyhedron *polyhedron)
{
    std::vector<double> polyhedron_vertices, polyhedron_faces;
    std::vector<std::vector<double>> poly;

    for(int i = 0; i < polyhedron->get_n_of_faces(); ++i)
    {
        for(int j = 0; j < polyhedron->get_faces()[i].get_n_of_vertices(); ++j)
        {
            SpacePoint v = polyhedron->get_faces()[i].get_vertices()[j];
            poly.push_back({v.x, v.y, v.z});
        }
    }

    std::string body = "[ ";
    for(int i = 0; i < poly.size(); ++i)
    {
        body += vector_double_to_json_array(poly[i]);
        if (i != poly.size() - 1)
            body += ',';
    }
    body += " ]";

    http::Request poly_request(urls.second);

    try
    {
        const http::Response response = poly_request.send("POST", body, {"Content-Type: application/json"});

        if(response.status < 200 || 300 <= response.status)
            throw http::ResponseError("Response status is not OK");
    }
    catch(const std::exception &e)
    {
        std::cerr << "Request failed, error: " << e.what() << std::endl;
        return false;
    }

    return true;
}

__host__ bool send_particles_to_visualization(const std::pair<std::string, std::string> &urls, MapNode *nodes,
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
    body += "\"x\":" + vector_double_to_json_array(x) +
            ",\"y\":" + vector_double_to_json_array(y) +
            ",\"z\":" + vector_double_to_json_array(z) + "}";


    http::Request particles_request(urls.first);

    try
    {
        const http::Response response = particles_request.send("POST", body, {"Content-Type: application/json"});

        if(response.status < 200 || 300 <= response.status)
            throw http::ResponseError("Response status is not OK");
    }
    catch(const std::exception &e)
    {
        std::cerr << "Request failed, error: " << e.what() << std::endl;
        return false;
    }

    return true;
}
