#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include "matmul.h"

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
void optimized_ops(float* a, float* b, float* out, int* shape_a, int* shape_b) {
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
