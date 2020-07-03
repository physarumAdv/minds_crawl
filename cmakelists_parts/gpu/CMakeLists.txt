cmake_minimum_required(VERSION 3.10)

project(mind_s_crawl CUDA)

set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CUDA_STANDARD 14)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmakelists_parts/common/1/CMakeLists.txt)
before_target_creation(sources_list)

add_executable(mind_s_crawl src/main.cu ${sources_list})

include(${CMAKE_CURRENT_SOURCE_DIR}/cmakelists_parts/common/2/CMakeLists.txt)
after_target_creation()

set_target_properties(
        mind_s_crawl
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
