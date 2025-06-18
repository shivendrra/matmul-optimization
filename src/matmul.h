#ifndef __MATMUL__H__
#define __MATMUL__H__

#include <stddef.h>

void naive_matmul(float* a, float* b, float* out, int* shape1, int* shape2, size_t ndim, size_t size1, size_t size2);
void optimized_matmul(float* a, float* b, float* out, int* shape1, int* shape2, size_t ndim, size_t size1, size_t size2);
void transpose_array(float* a, float* out, int* shape, size_t size, size_t ndim);

#endif  //!__MATMUL__H__