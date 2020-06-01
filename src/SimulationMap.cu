#include "SimulationMap.cuh"
#include "Polyhedron.cuh"
#include "common.cuh"


// Implement SimulationMap::SimulationMap(...) here

__device__ SimulationMap::~SimulationMap()
{
    delete[] nodes;
}


__device__ int SimulationMap::get_n_of_nodes() const
{
    return this->n_of_nodes;
}

__global__ void get_n_of_nodes(const SimulationMap *const simulation_map, int *return_value)
{
    STOP_ALL_THREADS_EXCEPT_FIRST;

    *return_value = simulation_map->get_n_of_nodes();
}

