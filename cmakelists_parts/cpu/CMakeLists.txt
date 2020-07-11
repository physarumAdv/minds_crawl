cmake_minimum_required(VERSION 3.10)

project(mind_s_crawl CXX)

set(CMAKE_CXX_STANDARD 17)

add_definitions(-DCOMPILE_FOR_CPU -D__host__= -D__device__= -D__global__=)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmakelists_parts/common/1/CMakeLists.txt)
before_target_creation(sources_list)

set_source_files_properties(${sources_list} PROPERTIES LANGUAGE CXX)
add_definitions(-x c++)

add_executable(mind_s_crawl src/main.cpp ${sources_list})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmakelists_parts/common/2/CMakeLists.txt)
after_target_creation()
