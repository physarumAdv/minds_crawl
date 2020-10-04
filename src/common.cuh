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


// ----------------------------------------------------
// ------ BEGIN LOCAL IMPLEMENTATIONS OF ATOMICS ------
// ----------------------------------------------------


typedef unsigned long long base_atomic_type;

#ifdef COMPILE_FOR_CPU

static base_atomic_type atomicCAS(base_atomic_type *address, const base_atomic_type compare, const base_atomic_type val)
{
    base_atomic_type ans = *address;

    if(*address == compare)
        *address = val;

    return ans;
}

static base_atomic_type atomicAdd(base_atomic_type *address, const base_atomic_type value)
{
    base_atomic_type ans = *address;
    *address += value;
    return ans;
}

static double atomicAdd(double *address, const double value)
{
    double ans = *address;
    *address += value;
    return ans;
}

#endif //COMPILE_FOR_CPU

// The following code is copied from https://stackoverflow.com/a/62094892/11248508 and modified (@kolayne can explain)
// `address` CANNOT be pointer to const, because we are trying to edit memory by it's source
__device__ static bool atomicCAS(bool *const address, const bool compare, const bool val)
{
    static_assert(sizeof(base_atomic_type) > 1, "The local atomicCAS implementation won't work if the size of "
                                                "`base_atomic_type` is <= 1");
    static_assert((sizeof(base_atomic_type) - 1 & sizeof(base_atomic_type)) == 0,
                  "The local atomicCAS implementation won't work if the size of `base_atomic_type` is not a power of 2");


    auto address_num = (unsigned long long)address;
    unsigned pos = address_num & (sizeof(base_atomic_type) - 1);  // byte position within the `base_atomic_type`

    auto *address_of_extended = (base_atomic_type *)(address - pos);  // `base_atomic_type`-aligned address
    base_atomic_type old_extended = *address_of_extended, compare_extended, current_value_extended;

    bool current_value;

    do
    {
        current_value = (bool)(old_extended & ((0xFFU) << (8 * pos)));

        if(current_value != compare) // If we expected that bool to be different, then
            break; // stop trying to update it and just return it's current value

        compare_extended = old_extended;

        if(val)
            current_value_extended = old_extended | (1U << (8 * pos));
        else
            current_value_extended = old_extended & (~((0xFFU) << (8 * pos)));

        old_extended = atomicCAS(address_of_extended, compare_extended, current_value_extended);
    } while(compare_extended != old_extended);

    return current_value;
}

// I (Nikolay Nechaev, @kolayne) have no idea why the fuck the following only works with #else. If you're reading this
// and now why, PLEASE, contact me and tell me
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
// The following code is from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions and
// slightly changed (@kolayne can explain)
__device__ static double atomicAdd(double* address, double val)
{
    auto* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600


// ----------------------------------------------------
// ------- END LOCAL IMPLEMENTATIONS OF ATOMICS -------
// ----------------------------------------------------


#endif //MIND_S_CRAWL_COMMON_CUH
