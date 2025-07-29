#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include "../matmul.h"

#define  BLOCK_SIZE  64
#define  NUM_ELEMNS  16

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
  // for small matrices or when SIMD doesn't help much, fall back to scalar
  if (rows * cols < BLOCK_SIZE) {
    for (int idx = 0; idx < rows * cols; ++idx) {
      int i = idx / cols, j = idx % cols;
      out[j * rows + i] = a[idx];
    }
    return;
  }

  int total_elements = rows * cols;
  int simd_end = (total_elements / NUM_ELEMNS) * NUM_ELEMNS;
  for (int idx = 0; idx < simd_end; idx += NUM_ELEMNS) {
    __m256 data = _mm256_loadu_ps(&a[idx]);
    float temp[NUM_ELEMNS];
    _mm256_storeu_ps(temp, data);
    // Transpose each element individually
    for (int offset = 0; offset < NUM_ELEMNS; offset++) {
        int current_idx = idx + offset;
        int i = current_idx / cols, j = current_idx % cols;
        out[j * rows + i] = temp[offset];
    }
  }

  for (int idx = simd_end; idx < total_elements; ++idx) {
    int i = idx / cols, j = idx % cols;
    out[j * rows + i] = a[idx];
  }
}

// SIMD optimized matrix multiplication using transposed second matrix
void simd_transpose_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int rows_a = shape_a[0];    // rows in 'a'
  int cols_a = shape_a[1];    // cols in 'a' 
  int rows_b = shape_b[0];    // rows in 'b' (original 'b' before transpose)
  int cols_b = shape_b[1];    // cols in 'b' (original 'b' before transpose)

  float* b_transposed = (float*)malloc(rows_b * cols_b * sizeof(float));
  if (b_transposed == NULL) {
    fprintf(stderr, "Memory allocation failed for transpose buffer\n");
    exit(EXIT_FAILURE);
  }
  simd_transpose_2d_array_ops(b, b_transposed, shape_b);

  // Perform matrix multiplication: A @ B^T using SIMD
  // A(rows_a × cols_a) @ B^T(cols_b × rows_b) = out(rows_a × cols_b)
  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      __m256 sum_vec = _mm256_setzero_ps();
      float sum = 0.0f;
      int simd_end = (cols_a / NUM_ELEMNS) * NUM_ELEMNS;
      for (int k = 0; k < simd_end; k += NUM_ELEMNS) {
        __m256 a_vec = _mm256_loadu_ps(&a[i * cols_a + k]);          // Load NUM_ELEMNS elements from row i of matrix A
        __m256 b_vec = _mm256_loadu_ps(&b_transposed[j * cols_a + k]);            // Load NUM_ELEMNS elements from row j of transposed matrix B
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);           // Multiply and accumulate using FMA (Fused Multiply-Add)
      }

      // horizontal sum of the vector to get final dot product
      // extracting high and low 128-bit lanes
      __m128 high = _mm256_extractf128_ps(sum_vec, 1);
      __m128 low = _mm256_castps256_ps128(sum_vec);
      __m128 sum128 = _mm_add_ps(high, low); // adding high and low lanes

      // horizontal add within 128-bit vector
      sum128 = _mm_hadd_ps(sum128, sum128);
      sum128 = _mm_hadd_ps(sum128, sum128);
      sum = _mm_cvtss_f32(sum128);        // extracting the final sum

      // handling remaining elements (scalar processing)
      for (int k = simd_end; k < cols_a; k++) { sum += a[i * cols_a + k] * b_transposed[j * cols_a + k]; }
      out[i * cols_b + j] = sum;
    }
  }
  free(b_transposed);
}

// alternative version with better cache utilization for larger matrices
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
            float sum = 0.0f;
            int k_simd_end = kk + ((k_end - kk) / NUM_ELEMNS) * NUM_ELEMNS; // SIMD processing within block
            for (int k = kk; k < k_simd_end; k += NUM_ELEMNS) {
              __m256 a_vec = _mm256_loadu_ps(&a[i * cols_a + k]);
              __m256 b_vec = _mm256_loadu_ps(&b_transposed[j * cols_a + k]);
              sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
            }
            // Horizontal sum
            __m128 high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 low = _mm256_castps256_ps128(sum_vec);
            __m128 sum128 = _mm_add_ps(high, low);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum += _mm_cvtss_f32(sum128);
            for (int k = k_simd_end; k < k_end; k++) { sum += a[i * cols_a + k] * b_transposed[j * cols_a + k]; }
            out[i * cols_b + j] += sum;
          }
        }
      }
    }
  }
  free(b_transposed);
}