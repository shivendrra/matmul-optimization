#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../matmul.h"

#define BLOCK_SIZE 64

void blocked_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int M = shape_a[0], K = shape_a[1], N = shape_b[1];
  memset(out, 0, M * N * sizeof(float));

  for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
    for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
      for (int jj = 0; jj < N; jj += BLOCK_SIZE) {

        int ie = ii + BLOCK_SIZE < M ? ii + BLOCK_SIZE : M;
        int ke = kk + BLOCK_SIZE < K ? kk + BLOCK_SIZE : K;
        int je = jj + BLOCK_SIZE < N ? jj + BLOCK_SIZE : N;

        for (int i = ii; i < ie; i++)
          for (int k = kk; k < ke; k++) {
            float aik = a[i * K + k];
            for (int j = jj; j < je; j++)
              out[i * N + j] += aik * b[k * N + j];
          }
      }
    }
  }
}

void blocked_transpose_2d_array_ops(float* a, float* out, int* shape) {
  int rows = shape[0], cols = shape[1];
  for (int ii = 0; ii < rows; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < cols; jj += BLOCK_SIZE) {
      int i_end = (ii + BLOCK_SIZE < rows) ? ii + BLOCK_SIZE : rows;
      int j_end = (jj + BLOCK_SIZE < cols) ? jj + BLOCK_SIZE : cols;
      for (int i = ii; i < i_end; i++) {
        for (int j = jj; j < j_end; j++) {
          int idx = i * cols + j;
          out[j * rows + i] = a[idx];
        }
      }
    }
  }
}

void blocked_transpose_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int rows_a = shape_a[0], cols_a = shape_a[1], rows_b = shape_b[0], cols_b = shape_b[1];
  float* b_transposed = (float*)malloc(rows_b * cols_b * sizeof(float));
  if (b_transposed == NULL) {
    fprintf(stderr, "Memory allocation failed for transpose buffer\n");
    exit(EXIT_FAILURE);
  }
  blocked_transpose_2d_array_ops(b, b_transposed, shape_b);
  memset(out, 0, rows_a * cols_b * sizeof(float));
  for (int ii = 0; ii < rows_a; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < cols_b; jj += BLOCK_SIZE) {
      for (int kk = 0; kk < cols_a; kk += BLOCK_SIZE) {
        int i_end = (ii + BLOCK_SIZE < rows_a) ? ii + BLOCK_SIZE : rows_a;
        int j_end = (jj + BLOCK_SIZE < cols_b) ? jj + BLOCK_SIZE : cols_b;
        int k_end = (kk + BLOCK_SIZE < cols_a) ? kk + BLOCK_SIZE : cols_a;
        for (int i = ii; i < i_end; i++) {
          for (int j = jj; j < j_end; j++) {
            float sum = out[i * cols_b + j];
            for (int k = kk; k < k_end; k++) {
              sum += a[i * cols_a + k] * b_transposed[j * cols_a + k];
            }
            out[i * cols_b + j] = sum;
          }
        }
      }
    }
  }
  free(b_transposed);
}