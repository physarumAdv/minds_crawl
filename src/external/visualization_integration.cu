#include <iostream>
#include <fstream>
#include <string>

#include "visualization_integration.cuh"
#include "../simulation_objects/Particle.cuh"


__host__ std::string get_visualization_endpoint()
{
    std::ifstream f("config/visualization_endpoint.txt", std::ios::in);
    std::string url;
    getline(f, url);
    return url;
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

__host__ bool send_particles_to_visualization(const std::string &url, MapNode *nodes, int n_of_nodes)
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

    http::Request request(url);

    try
    {
        const http::Response response = request.send("POST", body, {"Content-Type: application/json"});

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
