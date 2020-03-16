cmake_minimum_required(VERSION 3.16)

set(CMAKE_CUDA_COMPILER nvcc) # We are using nvcc 10.1

project(mind_s_crawl CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(mind_s_crawl main.cu)

set_target_properties(
    mind_s_crawl
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)