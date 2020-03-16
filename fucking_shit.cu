#include "fucking_shit.h"
typedef long long ll;

__device__ ll get_index(ll x, ll y, ll z, ll mx, ll my, ll mz)
{
    return x * my * mz + y * mz + z;
}
