#include <cuda_pipeline.h>

// Pipeline Primitives wrappers
inline __device__ void okl_memcpy_async(void* __restrict__ dst_shared,
                     const void* __restrict__ src_global,
                     size_t size_and_align,
                     size_t zfill = 0)
{
    __pipeline_memcpy_async(dst_shared, src_global, size_and_align, zfill);
}

inline __device__ void okl_pipeline_commit() {
    __pipeline_commit();
}

inline __device__ void okl_pipeline_wait_prior(size_t N) {
    __pipeline_wait_prior(N);
}
