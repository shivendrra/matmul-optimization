#ifndef __MATMUL__H__
#define __MATMUL__H__

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static inline float* aligned_malloc_32(size_t n) {
#if defined(_MSC_VER) || defined(__MINGW32__)
  return (float*)_aligned_malloc(n, 32);
#else
  void* ptr = NULL;
  if (posix_memalign(&ptr, 32, n) != 0) return NULL;
  return (float*)ptr;
#endif
}

static inline void aligned_free(void* p) {
#if defined(_MSC_VER) || defined(__MINGW32__)
  _aligned_free(p);
#else
  free(p);
#endif
}

// parallelization methods on naive matmul
void naive_matmul(float* a, float* b, float* out, int* shape1, int* shape2, size_t ndim, size_t size1, size_t size2);
void blocked_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b);
void openmp_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b);
void simd_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b);
void hybrid_parallel_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b);

// transpose matmul & parallel kernels
void transposed_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b);
void simd_transpose_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b);
void simd_transpose_matmul_blocked(float* a, float* b, float* out, int* shape_a, int* shape_b);
void openmp_transpose_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b);
void blocked_transpose_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b);
void hybrid_transposed_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b);

#endif  //!__MATMUL__H__