#include <iostream>
#include <cstdlib>

#include "simulation_logic.cuh"
#include "iterations_wrapper.cuh"
#include "../external/visualization_integration.cuh"


void wrapped_run_iteration_project_nutrients(SimulationMap *const simulation_map, const int *const iteration_number)
{
    for(unsigned int i = 0; i < simulation_map->get_n_of_nodes(); ++i)
        run_iteration_project_nutrients(simulation_map, iteration_number, i);
}

void wrapped_run_iteration_diffuse_trail(SimulationMap *const simulation_map, const int *const iteration_number)
{
    for(unsigned int i = 0; i < simulation_map->get_n_of_nodes(); ++i)
        run_iteration_diffuse_trail(simulation_map, iteration_number, i);
}

void wrapped_run_iteration_process_particles(SimulationMap *const simulation_map, const int *const iteration_number)
{
    for(unsigned int i = 0; i < simulation_map->get_n_of_nodes(); ++i)
        run_iteration_process_particles(simulation_map, iteration_number, i);
}

void wrapped_run_iteration_cleanup(SimulationMap *const simulation_map, int *const iteration_number)
{
    for(unsigned int i = 0; i < simulation_map->get_n_of_nodes(); ++i)
        run_iteration_cleanup(simulation_map, iteration_number, i);
}


int main()
{
    auto *simulation_map = (SimulationMap *)malloc(sizeof(SimulationMap));
    auto *polyhedron = (Polyhedron *)malloc(sizeof(Polyhedron));

    *polyhedron = generate_cube();

    init_simulation_objects(simulation_map, polyhedron);
    init_environment(simulation_map);

    int iteration_number = 0; // Incremented inside of `run_iteration_cleanup`

    RunIterationFunc iteration_runners[] = {(RunIterationFunc)wrapped_run_iteration_project_nutrients,
                                            (RunIterationFunc)wrapped_run_iteration_diffuse_trail,
                                            (RunIterationFunc)wrapped_run_iteration_process_particles,
                                            wrapped_run_iteration_cleanup};


    std::string visualization_endpoint = get_visualization_endpoint();


    while(true)
    {
        for(RunIterationFunc f : iteration_runners)
        {
            f(simulation_map, &iteration_number);
        }

        if(!send_particles_to_visualization(visualization_endpoint, simulation_map->nodes,
                                            simulation_map->get_n_of_nodes()))
        {
            std::cerr << "Error sending http request to visualization. Stopping the simulation process\n";
            break;
        }
    }

    destruct_simulation_objects(simulation_map);
    free(polyhedron);
    free(simulation_map);
}