# This part of CMakeLists is included AFTER creating the project target

function(before_target_creation sources_list)
    # Everything except main.cu/main.cpp:
    set(sources_list src/fucking_shit.cu src/random_generator.cu src/SimulationMap.cu src/Polyhedron.cu
            src/Particle.cu src/MapNode.cu src/SpacePoint.cu src/common.cuh src/Face.cu src/main_logic.cuh PARENT_SCOPE)

endfunction(before_target_creation)
