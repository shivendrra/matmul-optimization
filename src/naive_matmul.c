#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include "matmul.h"

void naive_matmul(float* a, float* b, float* out, int *shape1, int *shape2, size_t ndim, size_t size1, size_t size2) {
  for (int i = 0; i < shape1[0]; i++) {
    for (int j = 0; j < shape2[1]; j++) {
      float sum = 0.0;
      for (int k = 0; k < shape1[1]; k++) {
        sum += a[i * shape1[1] + k] * b[k * shape2[1] + j];
      }
      out[i * shape2[1] + j] = sum;
    }
  }
}