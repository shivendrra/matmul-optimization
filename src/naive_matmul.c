#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <stddef.h>
#include <immintrin.h>
#include "matmul.h"

// #define BLOCK_SIZE 64  uncomment this line for runtime test cases, and replace all `block_size`-> `BLOCK_SIZE` & vice versa for block_testing

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

// void hybrid_parallel_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) { uncomment this as well for normal test cases
void hybrid_parallel_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b, int block_size) {
  int rows_a = shape_a[0], cols_a = shape_a[1], cols_b = shape_b[1];

  memset(out, 0, rows_a * cols_b * sizeof(float));
  #pragma omp parallel for collapse(2) schedule(dynamic, 4)
  for (int ii = 0; ii < rows_a; ii += block_size) {
    for (int jj = 0; jj < cols_b; jj += block_size) {
      for (int kk = 0; kk < cols_a; kk += block_size) {

        int i_end = (ii + block_size < rows_a) ? ii + block_size : rows_a;
        int j_end = (jj + block_size < cols_b) ? jj + block_size : cols_b;
        int k_end = (kk + block_size < cols_a) ? kk + block_size : cols_a;

        for (int i = ii; i < i_end; i++) {
          for (int j = jj; j < j_end; j += 8) {
            __m256 sum = _mm256_loadu_ps(&out[i * cols_b + j]);

            for (int k = kk; k < k_end; k++) {
              __m256 a_vec = _mm256_broadcast_ss(&a[i * cols_a + k]);

              int remaining = j_end - j;
              if (remaining >= 8 && j + 8 <= cols_b) {
                __m256 b_vec = _mm256_loadu_ps(&b[k * cols_b + j]);
                sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
              } else {
                for (int jj_inner = j; jj_inner < j_end && jj_inner < j + 8; jj_inner++) {
                  out[i * cols_b + jj_inner] += a[i * cols_a + k] * b[k * cols_b + jj_inner];
                }
                continue;
              }
            }
            if (j + 8 <= cols_b && j + 8 <= j_end) { _mm256_storeu_ps(&out[i * cols_b + j], sum); }
          }
        }
      }
    }
  }
}