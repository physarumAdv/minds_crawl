#include <cstdlib>
#include <cstring>
#include <utility>

#include "common.cuh"


template<class T>
__host__ __device__ T *malloc_and_copy(const T *const source, int count)
{
    T *new_array = (T *)malloc(count * sizeof(T));
    for(int i = 0; i < count; ++i)
    {
        new_array[i] = source[i];
    }
    return new_array;
}

template<class T>
__host__ __device__ T *device_realloc(T *source, int old_size, int new_size)
{
    T *new_array = (T *)malloc(new_size * sizeof(T));
    for(int i = 0; i < old_size; ++i)
    {
        new_array[i] = std::move(source[i]);
    }
    free(source);
    return new_array;
}


template<class T>
__host__ __device__ void swap(T &a, T &b)
{
    T c = std::move(a);
    a = std::move(b);
    b = std::move(c);
}
