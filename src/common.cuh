#ifndef MIND_S_CRAWL_COMMON_CUH
#define MIND_S_CRAWL_COMMON_CUH


#define STOP_ALL_THREADS_EXCEPT_FIRST if(threadIdx.x || threadIdx.y || threadIdx.z || \
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


#endif //MIND_S_CRAWL_COMMON_CUH
