#include "SimulationMap.cuh"


__device__ ll SimulationMap::get_n_of_nodes() const
{
    return this->n_of_nodes;
}

__host__ void SimulationMap::get_n_of_nodes(ll *return_value) const
{
    *return_value = this->n_of_nodes;
}
