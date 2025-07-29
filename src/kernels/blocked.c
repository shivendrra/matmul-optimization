#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../matmul.h"

#define BLOCK_SIZE 64

void blocked_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int rows_a = shape_a[0], cols_a = shape_a[1], cols_b = shape_b[1];
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
              sum += a[i * cols_a + k] * b[k * cols_b + j];
            }
            out[i * cols_b + j] = sum;
          }
        }
      }
    }
  }
}