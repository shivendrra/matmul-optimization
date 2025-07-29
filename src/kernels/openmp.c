#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../matmul.h"

void openmp_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int rows_a = shape_a[0], cols_a = shape_a[1], cols_b = shape_b[1];
  #pragma omp parallel for collapse(2) schedule(dynamic, 8)
  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      float sum = 0.0f;
      for (int k = 0; k < cols_a; k++) {
        sum += a[i * cols_a + k] * b[k * cols_b + j];
      }
      out[i * cols_b + j] = sum;
    }
  }
}