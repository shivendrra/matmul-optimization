#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <stddef.h>
#include <immintrin.h>
#include "matmul.h"

#define BLOCK_SIZE 64

void transpose_2d_array_ops(float* a, float* out, int* shape) {
  int rows = shape[0], cols = shape[1];
  // Transpose: out[j][i] = a[i][j]
  for (int idx = 0; idx < rows * cols ; ++idx) {
    // out is cols x rows, so out[j][i] = out[j * rows + i]
    // a is rows x cols, so a[i][j] = a[idx]
    int i = idx / cols, j = idx % cols;
    out[j * rows + i] = a[idx];
    // this brings down time complexity from O(n2) -> O(n)
  }
}

// Optimized matrix multiplication using transposed second matrix
// A: shape_a[0] x shape_a[1], B^T: shape_b[1] x shape_b[0], C: shape_a[0] x shape_b[0]
// This computes C = A @ B where B is provided in transposed form
void transposed_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int rows_a = shape_a[0];    // rows in 'a'
  int cols_a = shape_a[1];    // cols in 'a' 
  int rows_b = shape_b[0];    // rows in 'b' (original 'b' before transpose)
  int cols_b = shape_b[1];    // cols in 'b' (original 'b' before transpose)
  float* b_transposed = (float*)malloc(rows_b * cols_b * sizeof(float));
  if (b_transposed == NULL) {
    fprintf(stderr, "Memory allocation failed for transpose buffer\n");
    exit(EXIT_FAILURE);
  }
  transpose_2d_array_ops(b, b_transposed, shape_b);
  // for a @ b^T: a(rows_a × cols_a) @ b^T(cols_b × rows_b) = out(rows_a × rows_b)
  // we need cols_a == cols_b for this to work
  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      float sum = 0.0f;
      // dot product between row i of A and row j of B^T (which is column j of original B)
      for (int k = 0; k < cols_a; k++) {
        sum += a[i * cols_a + k] * b_transposed[j * cols_a + k];
      }
      out[i * cols_b + j] = sum;
    }
  }
}

void hybrid_transpose_2d_array_ops(float* a, float* out, int* shape) {
  int rows = shape[0], cols = shape[1];
  #pragma omp parallel for schedule(static)
  for (int ii = 0; ii < rows; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < cols; jj += BLOCK_SIZE) {
      int i_end = (ii + BLOCK_SIZE < rows) ? ii + BLOCK_SIZE : rows;
      int j_end = (jj + BLOCK_SIZE < cols) ? jj + BLOCK_SIZE : cols;
      for (int i = ii; i < i_end; i++) {
        for (int j = jj; j < j_end; j++) { out[j * rows + i] = a[i * cols + j]; }
      }
    }
  }
}

void hybrid_transposed_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int rows_a = shape_a[0], cols_a = shape_a[1], cols_b = shape_b[1];
  int rows_b = shape_b[0];
  float* b_transposed = (float*)malloc(rows_b * cols_b * sizeof(float));
  if (b_transposed == NULL) {
    fprintf(stderr, "Memory allocation failed for transpose buffer\n");
    exit(EXIT_FAILURE);
  }
  hybrid_transpose_2d_array_ops(b, b_transposed, shape_b);
  memset(out, 0, rows_a * cols_b * sizeof(float));  
  #pragma omp parallel for collapse(2) schedule(dynamic, 8)
  for (int ii = 0; ii < rows_a; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < cols_b; jj += BLOCK_SIZE) {
      int i_end = (ii + BLOCK_SIZE < rows_a) ? ii + BLOCK_SIZE : rows_a;
      int j_end = (jj + BLOCK_SIZE < cols_b) ? jj + BLOCK_SIZE : cols_b;

      // restructured to leverage transpose benefits with proper memory access
      for (int i = ii; i < i_end; i++) {
        for (int j = jj; j < j_end; j++) {
          float sum = 0.0f;

          // vectorize the inner dot product using consecutive access on b_transposed
          int k = 0;
          for (; k <= cols_a - 8; k += 8) {
            __m256 a_vec = _mm256_loadu_ps(&a[i * cols_a + k]);
            __m256 b_vec = _mm256_loadu_ps(&b_transposed[j * cols_a + k]);
            __m256 prod = _mm256_mul_ps(a_vec, b_vec);

            // horizontal sum of the 8 products
            __m128 sum_high = _mm256_extractf128_ps(prod, 1);
            __m128 sum_low = _mm256_castps256_ps128(prod);
            __m128 sum128 = _mm_add_ps(sum_low, sum_high);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum += _mm_cvtss_f32(sum128);
          }

          // handle remaining elements
          for (; k < cols_a; k++) { sum += a[i * cols_a + k] * b_transposed[j * cols_a + k]; }
          out[i * cols_b + j] = sum;
        }
      }
    }
  }
  
  free(b_transposed);
}