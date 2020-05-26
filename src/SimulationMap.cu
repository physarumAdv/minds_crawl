#include <initializer_list>

#include "SimulationMap.cuh"
#include "Polyhedron.cuh"
#include "common.cuh"


// Implement SimulationMap::SimulationMap(...) here

__device__ SimulationMap::~SimulationMap()
{
    delete[] nodes;
}


__device__ ll SimulationMap::get_n_of_nodes() const
{
    return this->n_of_nodes;
}

__global__ void get_n_of_nodes(const SimulationMap *const simulation_map, ll *return_value)
{
    stop_all_threads_except_first;

    *return_value = simulation_map->get_n_of_nodes();
}


__device__ MapNode *SimulationMap::_find_nearest_mapnode_greedy(const SpacePoint dest, MapNode *const start)
{
    MapNode *current = start;
    double current_dist = get_distance(dest, current->coordinates);
    while(true)
    {
        bool found_better = false;
        for(auto next : {current->left, current->top, current->right, current->bottom})
        {
            double next_dist = get_distance(dest, next->coordinates);
            if(next_dist < current_dist)
            {
                current = next;
                current_dist = next_dist;
                found_better = true;
                break;
            }
        }
        if(!found_better)
            break;
    }
    return current;
}

__device__ MapNode *SimulationMap::find_nearest_mapnode(SpacePoint dest, MapNode *start)
{
    int dest_face = polyhedron->find_face_id_by_point(dest);

    if(start != nullptr)
    {
        MapNode *ans = _find_nearest_mapnode_greedy(dest, start);
        if(ans->polyhedron_face_id == dest_face)
            return ans;
    }

    return _find_nearest_mapnode_greedy(dest, polyhedron->faces[polyhedron->find_face_id_by_point(dest)].node);
}
