#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include "../matmul.h"

void simd_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int rows_a = shape_a[0], cols_a = shape_a[1], cols_b = shape_b[1];

  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j += 8) {
      __m256 sum = _mm256_setzero_ps();

      for (int k = 0; k < cols_a; k++) {
        __m256 a_vec = _mm256_broadcast_ss(&a[i * cols_a + k]);

        int remaining = cols_b - j;
        if (remaining >= 8) {
          __m256 b_vec = _mm256_loadu_ps(&b[k * cols_b + j]);
          sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
        } else {
          float b_vals[8] = {0};
          for (int idx = 0; idx < remaining; idx++) {
            b_vals[idx] = b[k * cols_b + j + idx];
          }
          __m256 b_vec = _mm256_loadu_ps(b_vals);
          sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
        }
      }

      float result[8];
      _mm256_storeu_ps(result, sum);
      int remaining = cols_b - j;
      for (int idx = 0; idx < remaining && idx < 8; idx++) {
        out[i * cols_b + j + idx] = result[idx];
      }
    }
  }
}