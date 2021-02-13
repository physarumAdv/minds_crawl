#include <iostream>
#include <cstdlib>

#include "simulation_logic.cuh"
#include "simulation_motor.cuh"
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

// The result is stored in the `reflections` array
void get_mapnodes_reflections(SimulationMap *const simulation_map, MapNodeReflection *reflections)
{
    for(unsigned int i = 0; i < simulation_map->get_n_of_nodes(); ++i)
        reflections[i] = get_mapnode_reflection(simulation_map, i);
}


int main()
{
    // `malloc`s and `free`s are not replaced with `new`/`delete`s in this function to make it as similar with the
    // main function from main.cu as possible

    auto *polyhedron = (Polyhedron *)malloc(sizeof(Polyhedron));
    auto *simulation_map = (SimulationMap *)malloc(sizeof(SimulationMap));

    init_simulation_objects(polyhedron, simulation_map);
    init_environment(simulation_map);

    int iteration_number = 0; // Incremented inside of `run_iteration_cleanup`

    int n_of_nodes = simulation_map->get_n_of_nodes();
    auto *mapnodes_reflections = (MapNodeReflection *)malloc(n_of_nodes * sizeof(MapNodeReflection));


    RunIterationFunc iteration_runners[] = {(RunIterationFunc)wrapped_run_iteration_project_nutrients,
                                            (RunIterationFunc)wrapped_run_iteration_diffuse_trail,
                                            (RunIterationFunc)wrapped_run_iteration_process_particles,
                                            wrapped_run_iteration_cleanup};

    std::pair<std::string, std::string> visualization_endpoints = get_visualization_endpoints();

    bool polyhedronDispatchFailed = false;
    if(!send_poly_to_visualization(visualization_endpoints, polyhedron))
    {
        std::cerr << "Error sending http request to visualization. Stopping the simulation process\n";
        polyhedronDispatchFailed = true;
    }

    if(!polyhedronDispatchFailed)
    {
        while (true)
	{
            get_mapnodes_reflections(simulation_map, mapnodes_reflections);

            for (RunIterationFunc f : iteration_runners) {
                f(simulation_map, &iteration_number);
            }

            if(!send_particles_to_visualization(visualization_endpoints, mapnodes_reflections, n_of_nodes))
            {
                std::cerr << "Error sending http request to visualization. Stopping the simulation process\n";
                break;
            }
        }
    }

    destruct_simulation_objects(polyhedron, simulation_map);
    free(polyhedron);
    free(simulation_map);
}
