#ifndef __MATMUL__H__
#define __MATMUL__H__

#include <stddef.h>

// parallelization methods on naive matmul
void naive_matmul(float* a, float* b, float* out, int* shape1, int* shape2, size_t ndim, size_t size1, size_t size2);
void blocked_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b);
void openmp_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b);
void simd_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b);
void hybrid_parallel_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b);

// transpose matmul & parallel kernels
void optimized_ops(float* a, float* b, float* out, int* shape_a, int* shape_b);
void simd_transpose_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b);
void simd_transpose_matmul_blocked(float* a, float* b, float* out, int* shape_a, int* shape_b);
void openmp_transpose_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b);
void blocked_transpose_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b);


#endif  //!__MATMUL__H__