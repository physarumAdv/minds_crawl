#ifdef COMPILE_FOR_CPU
#include <cmath>
#endif //COMPILE_FOR_CPU

#include <initializer_list>
#include <cstdio>

#include "fucking_shit.cuh"
#include "random_generator.cuh"
#include "jones_constants.hpp"
#include "Particle.cuh"
#include "jones_constants.hpp"

namespace jc = jones_constants;


[[nodiscard]] __device__ bool create_particle(MapNode *node)
{
    auto p = new Particle(node, rand0to1() * 2 * M_PI);

    if(node->attach_particle(p))
        return true;

    delete p;
    return false;
}

[[nodiscard]] __device__ bool delete_particle(MapNode *node)
{
    Particle *p = node->get_particle();

    if(!node->detach_particle(p))
        return false;

    delete p;
    return true;
}


__host__ __device__ void diffuse_trail(MapNode *node)
{
    auto left = node->get_left(), top = node->get_top(), right = node->get_right(), bottom = node->get_bottom();

    double sum = top->get_left()->trail + top->trail + top->get_right()->trail +
                 left->trail + node->trail + right->trail +
                 bottom->get_left()->trail + bottom->trail + bottom->get_right()->trail;

    node->temp_trail = (1 - jc::diffdamp) * (sum / 9.0);
}


__host__ __device__ int count_particles_in_node_window(MapNode *node, int window_size)
{
    for(int i = 0; i < window_size / 2; ++i)
        node = node->get_top()->get_left();

    MapNode *row = node;
    int ans = 0;
    for(int i = 0; i < window_size; ++i)
    {
        MapNode *cur = row;
        for(int j = 0; j < window_size; ++j)
        {
            if(cur->does_contain_particle())
                ++ans;
            cur = cur->get_right();
        }
        row = row->get_bottom();
    }

    return ans;
}


__device__ bool random_death_test(MapNode *node)
{
    if(rand0to1() < jc::random_death_probability)
    {
        if(!delete_particle(node))
        {
            // This is what called "undefined behaviour" in the docs :)
            printf("%s:%d - this line should never be reached", __FILE__, __LINE__);
            return false; // Particle was not removed
        }
        return true; // Particle was removed
    }
    return false; // Particle was not removed
}

__device__ bool death_test(MapNode *node)
{
    int particles_in_window = count_particles_in_node_window(node, jc::sw);
    if(jc::smin <= particles_in_window && particles_in_window <= jc::smax)
    {/* if in survival range, then stay alive */}
    else
    {
        if(!delete_particle(node))
        {
            // This is what called "undefined behaviour" in the docs :)
            printf("%s:%d - this line should never be reached", __FILE__, __LINE__);
            return false; // Particle was not removed
        }
        return true; // Particle was removed
    }
    return false; // Particle was not removed
}

__device__ void division_test(MapNode *node)
{
    int particle_window = count_particles_in_node_window(node, jc::gw);
    if(jc::gmin <= particle_window && particle_window <= jc::gmax)
    {
        if(rand0to1() <= jc::division_probability)
        {
            MapNode *row = node->get_top()->get_left();
            for(int i = 0; i < 3; ++i)
            {
                MapNode *cur = row;
                for(int j = 0; j < 3; ++j)
                {
                    if(create_particle(cur)) // If new particle was successfully created
                        return;
                    cur = cur->get_right();
                }
                row = row->get_bottom();
            }
        }
    }
}


__host__ __device__ MapNode *find_nearest_mapnode_greedy(const SpacePoint &dest, MapNode *const start)
{
    MapNode *current = start;
    double current_dist = get_distance(dest, current->get_coordinates());
    while(true)
    {
        bool found_better = false;
        for(auto next : {current->get_left(), current->get_top(), current->get_right(), current->get_bottom()})
        {
            double next_dist = get_distance(dest, next->get_coordinates());
            if(next_dist < current_dist)
            {
                current = next;
                current_dist = next_dist;
                found_better = true;
                break;
            }
        }
        if(!found_better)
            break;
    }
    return current;
}

__host__ __device__ MapNode *find_nearest_mapnode(const Polyhedron *const polyhedron, const SpacePoint &dest,
                                                  MapNode *const start)
{
    Face *dest_face = polyhedron->find_face_by_point(dest);

    if(start != nullptr)
    {
        MapNode *ans = find_nearest_mapnode_greedy(dest, start);
        if(*ans->get_face() == *dest_face)
            return ans;
    }

    return find_nearest_mapnode_greedy(dest, dest_face->get_node());
}

#ifdef COMPILE_FOR_CPU

base_atomic_type atomicCAS(base_atomic_type *address, const base_atomic_type compare, const base_atomic_type val)
{
    base_atomic_type ans = *address;

    if(*address == compare)
        *address = val;

    return ans;
}

base_atomic_type atomicAdd(base_atomic_type *address, const base_atomic_type value)
{
    base_atomic_type ans = *address;
    *address += value;
    return ans;
}

double atomicAdd(double *address, const double value)
{
    double ans = *address;
    *address += value;
    return ans;
}

#endif //COMPILE_FOR_CPU

// The following code is compied from https://stackoverflow.com/a/62094892/11248508 and modified (@kolayne can explain)
// `address` CANNOT be pointer to const, because we are trying to edit memory by it's source
__device__ bool atomicCAS(bool *const address, const bool compare, const bool val)
{
    static_assert(sizeof(base_atomic_type) > 1, "The local atomicCAS implementation won't work if `base_atomic_type"
                                                   "type size <= 1");
    static_assert((sizeof(base_atomic_type) - 1 & sizeof(base_atomic_type)) == 0,
            "The local atomicCAS implementation won't work if `base_atomic_type` type size is not a power of 2");


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
__device__ double atomicAdd(double* address, double val)
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
