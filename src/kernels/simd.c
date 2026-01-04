#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include "../matmul.h"

#define  BLOCK_SIZE  64
#define  NUM_ELEMNS  8

void simd_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int M = shape_a[0], K = shape_a[1], N = shape_b[1];
  int N8 = N & ~7;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N8; j += 8) {
      __m256 acc = _mm256_setzero_ps();
      for (int k = 0; k < K; k++) {
        __m256 a_b = _mm256_broadcast_ss(&a[i * K + k]);
        __m256 b_v = _mm256_loadu_ps(&b[k * N + j]);
        acc = _mm256_fmadd_ps(a_b, b_v, acc);
      }
      _mm256_storeu_ps(&out[i * N + j], acc);
    }
    for (int j = N8; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) sum += a[i * K + k] * b[k * N + j];
      out[i * N + j] = sum;
    }
  }
}

void simd_transpose_2d_array_ops(float* a, float* out, int* shape) {
  int rows = shape[0], cols = shape[1];
  
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j += NUM_ELEMNS) {
      int remaining = cols - j;
      if (remaining >= NUM_ELEMNS) {
        __m256 data = _mm256_loadu_ps(&a[i * cols + j]);
        float temp[NUM_ELEMNS];
        _mm256_storeu_ps(temp, data);
        for (int offset = 0; offset < NUM_ELEMNS; offset++) {
          out[(j + offset) * rows + i] = temp[offset];
        }
      } else {
        for (int offset = 0; offset < remaining; offset++) {
          out[(j + offset) * rows + i] = a[i * cols + j + offset];
        }
      }
    }
  }
}

void simd_transpose_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int M = shape_a[0], K = shape_a[1], N = shape_b[1];
  float* bt = aligned_malloc_32(K * N * sizeof(float));

  for (int i = 0; i < shape_b[0]; i++)
    for (int j = 0; j < shape_b[1]; j++)
      bt[j * K + i] = b[i * N + j];
  int K8 = K & ~7;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      __m256 acc = _mm256_setzero_ps();
      int k = 0;
      for (; k < K8; k += 8) {
        __m256 av = _mm256_loadu_ps(&a[i * K + k]);
        __m256 bv = _mm256_loadu_ps(&bt[j * K + k]);
        acc = _mm256_fmadd_ps(av, bv, acc);
      }
      __m128 hi = _mm256_extractf128_ps(acc, 1);
      __m128 lo = _mm256_castps256_ps128(acc);
      __m128 sum = _mm_add_ps(hi, lo);
      sum = _mm_hadd_ps(sum, sum);
      sum = _mm_hadd_ps(sum, sum);
      float s = _mm_cvtss_f32(sum);
      for (; k < K; k++) s += a[i * K + k] * bt[j * K + k];
      out[i * N + j] = s;
    }
  }
  aligned_free(bt);
}

void simd_transpose_matmul_blocked(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int M = shape_a[0], K = shape_a[1], N = shape_b[1];
  float* bt = aligned_malloc_32(K * N * sizeof(float));

  for (int i = 0; i < shape_b[0]; i++)
    for (int j = 0; j < shape_b[1]; j++)
      bt[j * K + i] = b[i * N + j];

  memset(out, 0, M * N * sizeof(float));

  for (int ii = 0; ii < M; ii += BLOCK_SIZE)
    for (int jj = 0; jj < N; jj += BLOCK_SIZE)
      for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
        int ie = ii + BLOCK_SIZE < M ? ii + BLOCK_SIZE : M;
        int je = jj + BLOCK_SIZE < N ? jj + BLOCK_SIZE : N;
        int ke = kk + BLOCK_SIZE < K ? kk + BLOCK_SIZE : K;
        int ke8 = ke & ~7;
        for (int i = ii; i < ie; i++)
          for (int j = jj; j < je; j++) {
            __m256 acc = _mm256_setzero_ps();
            int k = kk;
            for (; k < ke8; k += 8) {
              __m256 av = _mm256_loadu_ps(&a[i * K + k]);
              __m256 bv = _mm256_loadu_ps(&bt[j * K + k]);
              acc = _mm256_fmadd_ps(av, bv, acc);
            }
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 sum = _mm_add_ps(hi, lo);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            float s = _mm_cvtss_f32(sum);
            for (; k < ke; k++) s += a[i * K + k] * bt[j * K + k];
            out[i * N + j] += s;
          }
      }
  aligned_free(bt);
}