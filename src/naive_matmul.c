#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <stddef.h>
#include <immintrin.h>
#include "matmul.h"

#define HYBRID_BLOCK_SIZE 64

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

void hybrid_parallel_matmul_impl(float* a, float* b, float* out, int* shape_a, int* shape_b, int block_size) {
  int M = shape_a[0], K = shape_a[1], N = shape_b[1];
  int N8 = N & ~7;

  memset(out, 0, M * N * sizeof(float));

  #pragma omp parallel for collapse(2) schedule(static)
  for (int ii = 0; ii < M; ii += block_size) {
    for (int jj = 0; jj < N8; jj += block_size) {
      for (int kk = 0; kk < K; kk += block_size) {

        int ie = ii + block_size < M ? ii + block_size : M;
        int je = jj + block_size < N8 ? jj + block_size : N8;
        int ke = kk + block_size < K ? kk + block_size : K;

        for (int i = ii; i < ie; i++) {
          for (int j = jj; j < je; j += 8) {
            __m256 acc = _mm256_loadu_ps(&out[i * N + j]);

            for (int k = kk; k < ke; k++) {
              __m256 a_b = _mm256_broadcast_ss(&a[i * K + k]);
              __m256 b_v = _mm256_loadu_ps(&b[k * N + j]);
              acc = _mm256_fmadd_ps(a_b, b_v, acc);
            }

            _mm256_storeu_ps(&out[i * N + j], acc);
          }
        }
      }
    }
  }

  for (int i = 0; i < M; i++) {
    for (int j = N8; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) sum += a[i * K + k] * b[k * N + j];
      out[i * N + j] = sum;
    }
  }
}

void hybrid_parallel_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  hybrid_parallel_matmul_impl(a, b, out, shape_a, shape_b, HYBRID_BLOCK_SIZE);
}