cmake_minimum_required(VERSION 3.10)

project(minds_crawl LANGUAGES CXX CUDA)

set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 14)


set(SOURCES src/main_logic/simulation_logic.cu src/external/random_generator.cu src/simulation_objects/SimulationMap.cu
        src/simulation_objects/geometric/Polyhedron.cu src/simulation_objects/Particle.cu
        src/simulation_objects/MapNode.cu src/simulation_objects/geometric/SpacePoint.cu
        src/simulation_objects/geometric/Face.cu src/external/visualization_integration.cu src/external/HashSet.cuh)

option(COMPILE_FOR_CPU "Make produced executable run on CPU instead of GPU" OFF)

if(COMPILE_FOR_CPU)
    list(APPEND SOURCES src/main_logic/main.cpp)

    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
    add_definitions(-x c++)
    add_definitions(-DCOMPILE_FOR_CPU -D__host__= -D__device__= -D__global__=)
else()
    list(APPEND SOURCES src/main_logic/main.cu)
endif()

add_executable(minds_crawl ${SOURCES})
set_target_properties(
        minds_crawl
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
