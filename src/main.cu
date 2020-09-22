#include <ctime>
#include <iostream>

#include "main_logic.cuh"
#include "visualization_integration.cuh"
#include "random_generator.cuh"


const int cuda_block_size = 256;


__global__ void wrapped_run_iteration_project_nutrients(SimulationMap *const simulation_map,
                                                        const int *const iteration_number)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    run_iteration_project_nutrients(simulation_map, iteration_number, i);
}

__global__ void wrapped_run_iteration_diffuse_trail(SimulationMap *const simulation_map,
                                                    const int *const iteration_number)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    run_iteration_diffuse_trail(simulation_map, iteration_number, i);
}

__global__ void wrapped_run_iteration_process_particles(SimulationMap *const simulation_map,
                                                        const int *const iteration_number)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    run_iteration_process_particles(simulation_map, iteration_number, i);
}

__global__ void wrapped_run_iteration_cleanup(SimulationMap *const simulation_map, int *const iteration_number)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    run_iteration_cleanup(simulation_map, iteration_number, i);
}


/**
 * Sets a device-allocated variable to a given value from host code
 *
 * Simply `cudaMemcpy`s the given value to the given pointer
 *
 * @tparam T Type of the value being copied
 *
 * @param destination Device memory pointer to copy value to
 * @param value Value to be copied
 *
 * @see get_device_variable_value
 */
template<class T>
__host__ inline void set_device_variable_value(T *destination, T value)
{
    cudaMemcpy(destination, &value, sizeof(T), cudaMemcpyHostToDevice);
}


/**
 * Returns a value from the given pointer to device memory
 *
 * @tparam T Type of a value being copied
 *
 * @param source Device memory pointer to copy from
 *
 * @returns Value from device memory
 *
 * @see set_device_variable_value
 */
template<class T>
__host__ inline T get_device_variable_value(T *source)
{
    T ans;
    cudaMemcpy(&ans, source, sizeof(T), cudaMemcpyDeviceToHost);
    return ans;
}


__host__ int main()
{
    {
        int count = 0;
        cudaGetDeviceCount(&count);
        if(count <= 0)
        {
            std::cerr << "No CUDA-capable GPU found. Exiting\n";
            return cudaErrorNoDevice;
        }
    }

    // Initializing cuRAND:
    init_rand<<<1, 1>>>(time(nullptr));

    // Initializing simulation objects
    SimulationMap *simulation_map;
    cudaMallocManaged((void **)&simulation_map, sizeof(SimulationMap));
    Polyhedron *polyhedron;
    cudaMallocManaged((void **)&polyhedron, sizeof(Polyhedron));

    *polyhedron = generate_cube();

    init_simulation_objects<<<1, 1>>>(simulation_map, polyhedron);
    init_environment<<<1, 1>>>(simulation_map);

    int *iteration_number; // Incremented inside of `run_iteration_cleanup`
    cudaMalloc((void **)&iteration_number, sizeof(int));
    set_device_variable_value(iteration_number, 0);

    // Obtaining `n_of_nodes`
    int n_of_nodes;
    {
        int *_temporary;
        cudaMalloc((void **)&_temporary, sizeof(int));
        get_n_of_nodes<<<1, 1>>>(simulation_map, _temporary);
        n_of_nodes = get_device_variable_value(_temporary);
        cudaFree(_temporary);
    }

    // Obtaining `nodes`
    MapNode *nodes, *nodes_d = simulation_map->nodes; // Will be used for drawing
    cudaMallocHost((void **)&nodes, sizeof(MapNode) * n_of_nodes);


    RunIterationFunc iteration_runners[] = {(RunIterationFunc)wrapped_run_iteration_project_nutrients,
                                            (RunIterationFunc)wrapped_run_iteration_diffuse_trail,
                                            (RunIterationFunc)wrapped_run_iteration_process_particles,
                                            wrapped_run_iteration_cleanup};


    // Creating cuda stream
    cudaStream_t iterations_stream;
    cudaStreamCreate(&iterations_stream);


    std::string visualization_endpoint = get_visualization_endpoint();


    const int cuda_grid_size = (n_of_nodes + cuda_block_size - 1) / cuda_block_size;

    if(cudaPeekAtLastError() == cudaSuccess)
    {
        while(true)
        {
            // (implicit synchronization)
            // THIS COPIED ARRAY WILL HAVE ALL THE POINTERS INVALIDATED!!!
            cudaMemcpy((void *)nodes, (void *)nodes_d, sizeof(MapNode) * n_of_nodes, cudaMemcpyDeviceToHost);

            if(cudaPeekAtLastError() != cudaSuccess) // After synchronization caused by cudaMemcpy
            {
                break;
            }

            for(RunIterationFunc f : iteration_runners)
            {
                f<<<cuda_grid_size, cuda_block_size, 0, iterations_stream>>>(simulation_map, iteration_number);
            }

            if(!send_particles_to_visualization(visualization_endpoint, nodes, n_of_nodes))
            {
                std::cerr << "Error sending http request to visualization. Stopping the simulation process\n";
                break;
            }
        }
    }

    cudaFree(nodes);
    cudaFree(iteration_number);
    destruct_simulation_objects<<<1, 1>>>(simulation_map);
    cudaFree(polyhedron);
    cudaFree(simulation_map);

    cudaError_t error = cudaPeekAtLastError();
    if(error != cudaSuccess)
    {
        std::cerr << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << std::endl;
    }
    return error;
}
