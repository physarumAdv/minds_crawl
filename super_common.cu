typedef ll long long;

__device__ ll get_index(ll x, ll y, ll z, ll mx, ll my, ll mz)
{
    return x * my * mz + y * mz + z;
}
