#include <stddef.h>

// Pipeline Primitives Interface
[[maybe_unused]]
inline void okl_memcpy_async(void* dst_shared,
                 const void*  src_global,
                 size_t size_and_align,
                 size_t zfill = 0);

[[maybe_unused]]
inline void okl_pipeline_commit();

[[maybe_unused]]
void okl_pipeline_wait_prior(size_t);
