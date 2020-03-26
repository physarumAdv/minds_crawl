# This part of CMakeLists is included AFTER creating the project target

function(before_target_creation executable_list)

    set(CMAKE_CUDA_COMPILER nvcc) # We are using nvcc 10.1

    set(CMAKE_CUDA_STANDARD 14)

    # Everything except main.cu/main.cpp:
    set(executables_list fucking_shit.cu random_generator.cu Polyhedron.cu PARENT_SCOPE)

endfunction(before_target_creation)
