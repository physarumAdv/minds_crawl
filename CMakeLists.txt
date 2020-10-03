cmake_minimum_required(VERSION 3.10)

project(minds_crawl LANGUAGES CXX CUDA)

set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 14)


set(SOURCES src/simulation_logic.cu src/random_generator.cu src/SimulationMap.cu src/Polyhedron.cu src/Particle.cu
        src/MapNode.cu src/SpacePoint.cu src/Face.cu)

option(COMPILE_FOR_CPU "Make produced executable run on CPU instead of GPU" OFF)

if(COMPILE_FOR_CPU)
    list(APPEND SOURCES src/main.cpp)

    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
    add_definitions(-x c++)
    add_definitions(-DCOMPILE_FOR_CPU -D__host__= -D__device__= -D__global__=)
else()
    list(APPEND SOURCES src/main.cu)
endif()

add_executable(minds_crawl ${SOURCES})
set_target_properties(
        minds_crawl
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
