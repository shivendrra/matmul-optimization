#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../matmul.h"

void openmp_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int M = shape_a[0], K = shape_a[1], N = shape_b[1];

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) sum += a[i * K + k] * b[k * N + j];
      out[i * N + j] = sum;
    }
  }
}

void openmp_transpose_2d_array_ops(float* a, float* out, int* shape) {
  int rows = shape[0], cols = shape[1];
  #pragma omp parallel for schedule(static)
  for (int idx = 0; idx < rows * cols; ++idx) {
    int i = idx / cols, j = idx % cols;
    out[j * rows + i] = a[idx];
  }
}

void openmp_transpose_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int rows_a = shape_a[0], cols_a = shape_a[1], rows_b = shape_b[0], cols_b = shape_b[1];
  float* b_transposed = (float*)malloc(rows_b * cols_b * sizeof(float));
  if (b_transposed == NULL) {
    fprintf(stderr, "Memory allocation failed for transpose buffer\n");
    exit(EXIT_FAILURE);
  }
  openmp_transpose_2d_array_ops(b, b_transposed, shape_b);
  #pragma omp parallel for collapse(2) schedule(dynamic, 8)
  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      float sum = 0.0f;
      for (int k = 0; k < cols_a; k++) {
        sum += a[i * cols_a + k] * b_transposed[j * cols_a + k];
      }
      out[i * cols_b + j] = sum;
    }
  }
  free(b_transposed);
}