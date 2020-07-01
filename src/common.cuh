#ifndef MIND_S_CRAWL_COMMON_CUH
#define MIND_S_CRAWL_COMMON_CUH


#define stop_all_threads_except_first if(threadIdx.x || threadIdx.y || threadIdx.z || \
        blockIdx.x || blockIdx.y || blockIdx.z) return


/**
 * Returns pointer to a copied array
 *
 * @tparam T Type of an array
 *
 * @param source Pointer to an array to copy
 * @param count Number of elements to copy
 *
 * @returns Pointer to a copied array
 */
template<class T>
__device__ T *malloc_and_copy(T *source, int count)
{
    T *p = (T *)malloc(count * sizeof(T));
    memcpy((void *)p, (void *)source, count * sizeof(T));
    return p;
}

/**
 * Creates new array with `new_size`, moves source to it, frees up source memory
 *
 * @tparam T Type of the array
 *
 * @param source Pointer to an array to reallocate
 * @param old_size Size of the source array
 * @param new_size Size of new array
 *
 * @note old_size must be less than or equal to new_size
 *
 * @returns Pointer to the created array
 */
template<class T>
__device__ T *device_realloc(T *source, int old_size, int new_size)
{
    T *new_array = (T *)malloc(new_size * sizeof(T));
    for(int i = 0; i < old_size; ++i)
    {
        new_array[i] = source[i];
    }
    free(source);
    return new_array;
}


#endif //MIND_S_CRAWL_COMMON_CUH
