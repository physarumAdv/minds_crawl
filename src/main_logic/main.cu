#include <ctime>
#include <iostream>

#include "simulation_logic.cuh"
#include "simulation_motor.cuh"
#include "../external/visualization_integration.cuh"
#include "../external/random_generator.cuh"


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

// The result is stored in the `reflections` array
__global__ void get_mapnodes_reflections(const SimulationMap *const simulation_map, MapNodeReflection *reflections)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= simulation_map->get_n_of_nodes())
        return;

    reflections[i] = get_mapnode_reflection(simulation_map, i);
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
    Polyhedron *polyhedron;
    cudaMalloc((void **)&polyhedron, sizeof(Polyhedron));
    SimulationMap *simulation_map;
    cudaMallocManaged((void **)&simulation_map, sizeof(SimulationMap));

    init_simulation_objects<<<1, 1>>>(polyhedron, simulation_map);
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

    const int cuda_grid_size = (n_of_nodes + cuda_block_size - 1) / cuda_block_size;


    RunIterationFunc iteration_runners[] = {(RunIterationFunc)wrapped_run_iteration_project_nutrients,
                                            (RunIterationFunc)wrapped_run_iteration_diffuse_trail,
                                            (RunIterationFunc)wrapped_run_iteration_process_particles,
                                            wrapped_run_iteration_cleanup};


    // Creating cuda stream
    cudaStream_t iterations_stream;
    cudaStreamCreate(&iterations_stream);


    std::pair<std::string, std::string> visualization_endpoints = get_visualization_endpoints();


    MapNodeReflection *mapnodes_reflections;
    cudaMallocManaged(&mapnodes_reflections, n_of_nodes * sizeof(MapNodeReflection));

    bool modelDispatchFailed = false;
    {
        // Please, pay attention! This is a very ugly and temporary solution for
        // https://github.com/physarumAdv/minds_crawl/issues/47. I know some ways to implement it much better, but
        // I don't want to loose time on doing this now, because it's still unclear for me how we're going to import
        // polyhedrons in future, so the way to send polyhedron I might have chosen could be incompatible with the way
        // we import polyhedrons. That's why now I just create another cube and send it instead of the original one
        Polyhedron temp_poly = generate_cube();
        if(!send_poly_to_visualization(visualization_endpoints, &temp_poly))
        {
            std::cerr << "Error sending http request to visualization. Stopping the simulation process\n";
            modelDispatchFailed = true;
        }
    }

    if(!modelDispatchFailed)
    {
        while(cudaPeekAtLastError() == cudaSuccess)
        {
            // Reflect `MapNode`s (in fact, defer to previous stream operations completion)
            get_mapnodes_reflections<<<cuda_grid_size, cuda_block_size, 0, iterations_stream>>>(simulation_map,
                                                                                                mapnodes_reflections);

            // Wait for all the operations in the stream to finish
            if(cudaStreamSynchronize(iterations_stream) != cudaSuccess)
            {
                break;
            }

            // Run an iteration of the simulation (in fact, defer the iteration runners)
            for(RunIterationFunc f : iteration_runners)
            {
                f<<<cuda_grid_size, cuda_block_size, 0, iterations_stream>>>(simulation_map, iteration_number);
            }

            // Send `MapNodeReflection`s retrieved earlier to a visualizer
            if(!send_particles_to_visualization(visualization_endpoints, mapnodes_reflections, n_of_nodes))
            {
                std::cerr << "Error sending http request to visualization. Stopping the simulation process\n";
                break;
            }
        }
    }

    cudaFree(mapnodes_reflections);
    cudaStreamDestroy(iterations_stream);
    cudaFree(iteration_number);
    destruct_simulation_objects<<<1, 1>>>(polyhedron, simulation_map);
    cudaFree(polyhedron);
    cudaFree(simulation_map);

    cudaError_t error = cudaPeekAtLastError();
    if(error != cudaSuccess)
    {
        std::cerr << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << std::endl;
    }
    return error;
}
