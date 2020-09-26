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
__host__ __device__ static T *malloc_and_copy(const T *source, int count)
{
    T *new_array = (T *)malloc(count * sizeof(T));
    for(int i = 0; i < count; ++i)
    {
        new_array[i] = source[i];
    }
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
__host__ __device__ static void swap(T &a, T &b)
{
    T c = std::move(a);
    a = std::move(b);
    b = std::move(c);
}


#endif //MIND_S_CRAWL_COMMON_CUH
