#include "SimulationMap.cuh"
#include "common.cuh"


__device__ ll SimulationMap::get_n_of_nodes() const
{
    return this->n_of_nodes;
}

__global__ void get_n_of_nodes(const SimulationMap *const simulation_map, ll *return_value)
{
    stop_all_threads_except_first;

    *return_value = simulation_map->get_n_of_nodes();
}
