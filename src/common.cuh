#ifndef MIND_S_CRAWL_COMMON_CUH
#define MIND_S_CRAWL_COMMON_CUH


#include <cstdlib>
#include <cstring>
#include <utility>


#ifdef COMPILE_FOR_CPU
#define STOP_ALL_THREADS_EXCEPT_FIRST
#else
#define STOP_ALL_THREADS_EXCEPT_FIRST if(threadIdx.x || threadIdx.y || threadIdx.z || \
        blockIdx.x || blockIdx.y || blockIdx.z) return
#endif //COMPILE_FOR_CPU


/**
 * Returns pointer to a copied array
 *
 * @tparam T Type of an array
 *
 * @param source Pointer-represented array to copy
 * @param count Number of elements to copy
 *
 * @returns Pointer to a copied array
 */
template<class T>
__device__ T *malloc_and_copy(const T *const source, int count)
{
    T *new_array = (T *)malloc(count * sizeof(T));
    for(int i = 0; i < count; ++i)
    {
        new_array[i] = source[i];
    }
    return new_array;
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
        new_array[i] = std::move(source[i]);
    }
    free(source);
    return new_array;
}


/**
 * Swaps two given values
 *
 * @tparam T Type of values to be swapped
 *
 * @param a Value to be swapped
 * @param b Value to be swapped
 */
template<class T>
__device__ void swap(T &a, T &b)
{
    T c = std::move(a);
    a = std::move(b);
    b = std::move(c);
}


#endif //MIND_S_CRAWL_COMMON_CUH
