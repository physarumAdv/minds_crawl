#ifndef MIND_S_CRAWL_POLYHEDRON_CUH
#define MIND_S_CRAWL_POLYHEDRON_CUH

class Polyhedron
{
public:
    Polyhedron(...);

    __device__ bool contains_point(long long x, long long y, long long z) const;
    
    __host__ __device__ long long get_max_x() const;
    __host__ __device__ long long get_max_y() const;
    __host__ __device__ long long get_max_z() const;

private:
    long long max_x, max_y, max_z;
};

#endif //MIND_S_CRAWL_POLYHEDRON_CUH
