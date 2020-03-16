#ifndef MIND_S_CRAWL_GRID_PROCESSING_H
#define MIND_S_CRAWL_GRID_PROCESSING_H

#include "MapPoint.h"
#include "fucking_shit.h"
typedef long long ll;

__device__ void diffuse_trail(MapPoint *grid, ll x, ll y, ll z, ll mx, ll my, ll mz);

#endif //MIND_S_CRAWL_GRID_PROCESSING_H
