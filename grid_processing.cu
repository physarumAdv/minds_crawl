#include "MapPoint.h"
typedef long long ll;

__device__ void diffuse_trail(MapPoint *grid, ll x, ll y, ll z, ll mx, ll my, ll mz)
{
    ll sum, cnt;
    sum = cnt = 0;
    for(int dx = -1; dx <= 1; ++dx)
        for(int dy = -1; dy <= 1; ++dy)
            for(int dz = -1; dz <= 1; ++dz)
            {
                if(x + dx > 0 && x + dx < mx)
                    if(y + dy > 0 && y + dy < my)
                        if(z + dz > 0 && z + dz < mz)
                        {
                            sum += grid[get_index(x+dx, y+dy, z+dz, mx, my, mz)].trail;
                            ++cnt;
                        }
            }
    grid[get_index(x, y, z, mx, my, mz)].temp_trail += sum;
}
