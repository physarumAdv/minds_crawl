#ifndef MIND_S_CRAWL_COMMON_CUH
#define MIND_S_CRAWL_COMMON_CUH


typedef long long ll;


#define stop_all_threads_except_first if(threadIdx.x || threadIdx.y || threadIdx.z || \
        blockIdx.x || blockIdx.y || blockIdx.z) return


#endif //MIND_S_CRAWL_COMMON_CUH
