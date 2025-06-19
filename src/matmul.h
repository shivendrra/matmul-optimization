#ifndef __MATMUL__H__
#define __MATMUL__H__

#include <stddef.h>

void transpose_2d_array_ops(float* a, float* out, int* shape);
void optimized_ops(float* a, float* b, float* out, int* shape_a, int* shape_b);

void naive_matmul(float* a, float* b, float* out, int* shape1, int* shape2, size_t ndim, size_t size1, size_t size2);

#endif  //!__MATMUL__H__