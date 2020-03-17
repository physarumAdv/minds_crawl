cmake_minimum_required(VERSION 3.16)

project(mind_s_crawl CUDA)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmakelists_parts/common/1/CMakeLists.txt)
before_target_creation()

add_executable(mind_s_crawl main.cu ${executables_list})

include(${CMAKE_CURRENT_SOURCE_DIR}/cmakelists_parts/common/2/CMakeLists.txt)
after_target_creation()
