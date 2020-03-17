# This part of CMakeLists is included AFTER creating the project target

function(after_target_creation)

    set_target_properties(
        mind_s_crawl
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

endfunction(after_target_creation)
