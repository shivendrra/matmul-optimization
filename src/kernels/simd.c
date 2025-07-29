#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include "../matmul.h"

#define  BLOCK_SIZE  64
#define  NUM_ELEMNS  8

void simd_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int rows_a = shape_a[0], cols_a = shape_a[1], cols_b = shape_b[1];

  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j += NUM_ELEMNS) {
      __m256 sum = _mm256_setzero_ps();

      for (int k = 0; k < cols_a; k++) {
        __m256 a_vec = _mm256_broadcast_ss(&a[i * cols_a + k]);

        int remaining = cols_b - j;
        if (remaining >= NUM_ELEMNS) {
          __m256 b_vec = _mm256_loadu_ps(&b[k * cols_b + j]);
          sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
        } else {
          float b_vals[NUM_ELEMNS] = {0};
          for (int idx = 0; idx < remaining; idx++) {
            b_vals[idx] = b[k * cols_b + j + idx];
          }
          __m256 b_vec = _mm256_loadu_ps(b_vals);
          sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
        }
      }

      float result[NUM_ELEMNS];
      _mm256_storeu_ps(result, sum);
      int remaining = cols_b - j;
      for (int idx = 0; idx < remaining && idx < NUM_ELEMNS; idx++) {
        out[i * cols_b + j + idx] = result[idx];
      }
    }
  }
}

// SIMD version of transpose operation
void simd_transpose_2d_array_ops(float* a, float* out, int* shape) {
  int rows = shape[0], cols = shape[1];
  
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j += NUM_ELEMNS) {
      int remaining = cols - j;
      if (remaining >= NUM_ELEMNS) {
        __m256 data = _mm256_loadu_ps(&a[i * cols + j]);
        float temp[NUM_ELEMNS];
        _mm256_storeu_ps(temp, data);
        for (int offset = 0; offset < NUM_ELEMNS; offset++) { out[(j + offset) * rows + i] = temp[offset]; }
      } else {
        for (int offset = 0; offset < remaining; offset++) { out[(j + offset) * rows + i] = a[i * cols + j + offset]; }
      }
    }
  }
}

// SIMD optimized matrix multiplication using transposed second matrix
void simd_transpose_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int rows_a = shape_a[0], cols_a = shape_a[1], rows_b = shape_b[0], cols_b = shape_b[1];
  float* b_transposed = (float*)malloc(rows_b * cols_b * sizeof(float));
  if (b_transposed == NULL) {
    fprintf(stderr, "Memory allocation failed for transpose buffer\n");
    exit(EXIT_FAILURE);
  }
  simd_transpose_2d_array_ops(b, b_transposed, shape_b);
  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      __m256 sum_vec = _mm256_setzero_ps();
      int simd_end = (cols_a / NUM_ELEMNS) * NUM_ELEMNS;

      for (int k = 0; k < simd_end; k += NUM_ELEMNS) {
        __m256 a_vec = _mm256_loadu_ps(&a[i * cols_a + k]);
        __m256 b_vec = _mm256_loadu_ps(&b_transposed[j * cols_a + k]);
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
      }

      float sum_array[NUM_ELEMNS];
      _mm256_storeu_ps(sum_array, sum_vec);
      float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] + sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
      for (int k = simd_end; k < cols_a; k++) { sum += a[i * cols_a + k] * b_transposed[j * cols_a + k]; }
      out[i * cols_b + j] = sum;
    }
  }
  free(b_transposed);
}

void simd_transpose_matmul_blocked(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int rows_a = shape_a[0], cols_a = shape_a[1], rows_b = shape_b[0], cols_b = shape_b[1];
  float* b_transposed = (float*)malloc(rows_b * cols_b * sizeof(float));
  if (b_transposed == NULL) {
    fprintf(stderr, "Memory allocation failed for transpose buffer\n");
    exit(EXIT_FAILURE);
  }
  simd_transpose_2d_array_ops(b, b_transposed, shape_b);
  for (int i = 0; i < rows_a * cols_b; i++) { out[i] = 0.0f; }
  for (int ii = 0; ii < rows_a; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < cols_b; jj += BLOCK_SIZE) {
      for (int kk = 0; kk < cols_a; kk += BLOCK_SIZE) {
        int i_end = (ii + BLOCK_SIZE < rows_a) ? ii + BLOCK_SIZE : rows_a;
        int j_end = (jj + BLOCK_SIZE < cols_b) ? jj + BLOCK_SIZE : cols_b;
        int k_end = (kk + BLOCK_SIZE < cols_a) ? kk + BLOCK_SIZE : cols_a;

        for (int i = ii; i < i_end; i++) {
          for (int j = jj; j < j_end; j++) {
            __m256 sum_vec = _mm256_setzero_ps();
            int k_simd_end = kk + ((k_end - kk) / NUM_ELEMNS) * NUM_ELEMNS;
            for (int k = kk; k < k_simd_end; k += NUM_ELEMNS) {
              __m256 a_vec = _mm256_loadu_ps(&a[i * cols_a + k]);
              __m256 b_vec = _mm256_loadu_ps(&b_transposed[j * cols_a + k]);
              sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
            }

            float sum_array[NUM_ELEMNS];
            _mm256_storeu_ps(sum_array, sum_vec);
            float partial_sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] + sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

            for (int k = k_simd_end; k < k_end; k++) {
              partial_sum += a[i * cols_a + k] * b_transposed[j * cols_a + k];
            }
            out[i * cols_b + j] += partial_sum;
          }
        }
      }
    }
  }
  free(b_transposed);
}