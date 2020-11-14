#ifndef MINDS_CRAWL_HASHMAP_CUH
#define MINDS_CRAWL_HASHMAP_CUH


#include <functional>

#include "../simulation_objects/SimulationMap.cuh"


/**
 * Very simple (and pretty ugly :( ) implementation of hash map (which should, among other things, consume memory as
 * hell because of using same number of keys per bucket)
 *
 * Implementation of hash map which only supports adding key-value pairs and retrieving a value by key. Uses same bucket
 * size for each bucket. Undefined behaviour if trying to add more than `max_bucket_size` keys to one bucket
 *
 * @tparam KeyT Type of the container's element
 */
template<class KeyT, class ValueT>
class HashMap
{
public:
    /**
     * Constructor of `HashMap`
     *
     * @param buckets_count Number of buckets to be created for the container's elements
     * @param max_bucket_size Number of elements can be stored in the same bucket
     * @param hasher_func Function accepting `const KeyT &` and returning its hash as `unsigned long long`.
     *      Should be `__host__ __device__`, undefined behaviour otherwise
     */
    explicit __host__ __device__ HashMap(size_t buckets_count, size_t max_bucket_size,
                                         unsigned long long (*hasher_func)(const KeyT &))
            : sizes_of_buckets((size_t *)malloc(buckets_count * sizeof(size_t))),
              keys((KeyT *)malloc(buckets_count * max_bucket_size * sizeof(KeyT))),
              values((ValueT *)malloc(buckets_count * max_bucket_size * sizeof(ValueT))),
              max_bucket_size(max_bucket_size), buckets_count(buckets_count), hasher(hasher_func)
    {
        for(int i = 0; i < buckets_count; ++i)
            sizes_of_buckets[i] = 0;
    }

    __host__ __device__ ~HashMap()
    {
        free(sizes_of_buckets);
        free(keys);
        free(values);
    }

    /**
     * Set value in container by its key. If element with the given key is not in the container, it is created
     *
     * @param key Key to be added
     * @param value Value to be set by `key`
     */
    __host__ __device__ void set(const KeyT &key, ValueT value)
    {
        unsigned long long hash = hasher(key);
        size_t bucket_index = hash % buckets_count;

        for(size_t i = 0; i < sizes_of_buckets[bucket_index]; ++i)
        {
            size_t idx = bucket_index * max_bucket_size + i;

            if(keys[idx] == key)
            {
                values[idx] = value;
                return;
            }
        }

        size_t idx = bucket_index * max_bucket_size + sizes_of_buckets[bucket_index]++;
        keys[idx] = key;
        values[idx] = value;
    }

    /**
     * Get value by key
     *
     * @param key Key to search for
     * @param default_ Value to be returned if `key` is not in the container
     * @returns Value stored in the container by `key`, if `key` is in the container
     */
    __host__ __device__ ValueT get(const KeyT &key, const ValueT &default_)
    {
        size_t hash = hasher(key);
        size_t bucket_index = hash % buckets_count;

        for(size_t i = 0; i < sizes_of_buckets[bucket_index]; ++i)
        {
            size_t idx = bucket_index * max_bucket_size + i;
            if(keys[bucket_index * max_bucket_size + i] == key)
                return values[idx];
        }

        return default_;
    }

private:
    KeyT *keys;
    ValueT *values;
    size_t *sizes_of_buckets;

    size_t buckets_count;
    size_t max_bucket_size;

    unsigned long long (*hasher)(const KeyT &);
};


#endif //MINDS_CRAWL_HASHMAP_CUH
