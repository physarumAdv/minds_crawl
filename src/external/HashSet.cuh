#ifndef MINDS_CRAWL_HASHSET_CUH
#define MINDS_CRAWL_HASHSET_CUH


#include <functional>

#include "../simulation_objects/SimulationMap.cuh"


/**
 * Very simple implementation of hash set (which should consume memory as hell because of using same number of
 * elements per bucket)
 *
 * Implementation of hash set which only supports adding elements and checking if an element was added to it. Uses same
 * bucket size for each bucket. Undefined behaviour if trying to add more than `max_bucket_size` elements to one bucket
 *
 * @tparam T Type of the container's element
 */
template<class T>
class HashSet
{
public:
    /**
     * Constructor of `HashSet`
     *
     * @param buckets_count Number of buckets to be created for the container's elements
     * @param max_bucket_size Number of elements can be stored in the same bucket
     * @param hasher_func Function accepting `const T &` and returning its hash as `unsigned long long`.
     *      Should be `__host__ __device__`, undefined behaviour otherwise
     */
    explicit __host__ __device__ HashSet(size_t buckets_count, size_t max_bucket_size,
                                         unsigned long long (*hasher_func)(const T &))
            : sizes_of_buckets((size_t *)malloc(buckets_count * sizeof(size_t))),
              elements((T *)malloc(buckets_count * max_bucket_size * sizeof(T))),
              max_bucket_size(max_bucket_size), buckets_count(buckets_count), hasher(hasher_func)
    {
        for(int i = 0; i < buckets_count; ++i)
            sizes_of_buckets[i] = 0;
    }

    /**
     * Add element to the container
     *
     * @param elem Element to be added
     * @returns `true` if the element was added, `false` otherwise (which means the element was in the container
     *      already)
     */
    __host__ __device__ bool add(const T &elem)
    {
        unsigned long long hash = hasher(elem);
        size_t bucket_index = hash % buckets_count;

        for(size_t i = 0; i < sizes_of_buckets[bucket_index]; ++i)
        {
            if(elements[bucket_index * max_bucket_size + i] == elem)
                return false;
        }

        elements[bucket_index * max_bucket_size + sizes_of_buckets[bucket_index]++] = elem;
        return true;
    }

    /**
     * Checks if an element is present in the container
     *
     * @param elem Element to search for
     * @returns `true` if the element is in the container, `false` otherwise
     */
    __host__ __device__ bool contains(const T &elem)
    {
        size_t hash = hasher(elem);
        size_t bucket_index = hash % buckets_count;

        for(size_t i = 0; i < sizes_of_buckets[bucket_index]; ++i)
        {
            if(elements[bucket_index * max_bucket_size + i] == elem)
                return true;
        }

        return false;
    }

private:
    T *elements;
    size_t *sizes_of_buckets;

    size_t buckets_count;
    size_t max_bucket_size;

    unsigned long long (*hasher)(const T &);
};


#endif //MINDS_CRAWL_HASHSET_CUH
