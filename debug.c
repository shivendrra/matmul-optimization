// gcc -O3 -fopenmp -mavx2 -mfma -std=c11 -o debug debug.c src/naive_matmul.c src/trans_matmul.c src/kernels/blocked.c src/kernels/openmp.c src/kernels/simd.c -lm
// ./debug

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "src/inc/helpers.h"
#include "src/matmul.h"

typedef struct {
  const char* name;
  void (*func)(float*, float*, float*, int*, int*);
  void (*func_naive)(float*, float*, float*, int*, int*, size_t, size_t, size_t);
  int is_naive;
} TestFunction;

void print_matrix_sample(float* matrix, int rows, int cols, const char* name) {
  printf("%s (showing first 8x8):\n", name);
  int max_rows = rows < 8 ? rows : 8, max_cols = cols < 8 ? cols : 8;
  for (int i = 0; i < max_rows; i++) {
    for (int j = 0; j < max_cols; j++) { printf("%8.3f ", matrix[i * cols + j]); }
    printf("\n");
  }
  printf("\n");
}

void find_differences(float* ref, float* test, int rows, int cols, const char* func_name, float tolerance) {
  int diff_count = 0, max_diffs = 10;  
  printf("Checking %s vs reference:\n", func_name);
  for (int i = 0; i < rows && diff_count < max_diffs; i++) {
    for (int j = 0; j < cols && diff_count < max_diffs; j++) {
      int idx = i * cols + j;
      float diff = fabs(ref[idx] - test[idx]);
      if (diff > tolerance) {
        printf("  Diff at [%d,%d]: ref=%.6f, test=%.6f, diff=%.6f\n", i, j, ref[idx], test[idx], diff);
        diff_count++;
      }
    }
  }
  
  if (diff_count == 0) { printf("  All values match within tolerance %.6f\n", tolerance); }
  else if (diff_count >= max_diffs) { printf("  ... and more differences (showing first %d)\n", max_diffs); }
  printf("\n");
}

int verify_and_debug(float* ref, float* test, int rows, int cols, const char* func_name, float tolerance) {
  int total_errors = 0;
  float max_diff = 0.0f;
  int max_diff_i = 0, max_diff_j = 0;

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int idx = i * cols + j;
      float diff = fabs(ref[idx] - test[idx]);
      if (diff > tolerance) { total_errors++; }
      if (diff > max_diff) {
        max_diff = diff;
        max_diff_i = i;
        max_diff_j = j;
      }
    }
  }
  printf("%-35s: ", func_name);
  if (total_errors == 0) { printf("PASS (max_diff: %.2e)\n", max_diff); }
  else { printf("FAIL (%d errors, max_diff: %.2e at [%d,%d])\n", total_errors, max_diff, max_diff_i, max_diff_j); }
  return total_errors == 0;
}

void debug_matrix_size(int size) {
  printf("========================================\n");
  printf("DEBUGGING MATRIX SIZE: %dx%d\n", size, size);
  printf("========================================\n");
  
  int shape_a[2] = {size, size};
  int shape_b[2] = {size, size};
  size_t matrix_size = size * size;
  
  float* a = randn_array(shape_a, matrix_size, 2);
  float* b = randn_array(shape_b, matrix_size, 2);
  float* ref_out = (float*)calloc(matrix_size, sizeof(float));
  float* test_out = (float*)calloc(matrix_size, sizeof(float));
  
  if (!a || !b || !ref_out || !test_out) {
    printf("Memory allocation failed for size %d\n", size);
    return;
  }
  
  printf("Computing reference (naive) result...\n");
  naive_matmul(a, b, ref_out, shape_a, shape_b, 2, matrix_size, matrix_size);
  
  TestFunction functions[] = {
    {"blocked_matmul", blocked_matmul, NULL, 0},
    {"openmp_matmul", openmp_matmul, NULL, 0},
    {"simd_matmul", simd_matmul, NULL, 0},
    {"hybrid_parallel_matmul", hybrid_parallel_matmul, NULL, 0},
    {"transposed_matmul", transposed_matmul, NULL, 0},
    {"simd_transpose_matmul", simd_transpose_matmul, NULL, 0},
    {"simd_transpose_matmul_blocked", simd_transpose_matmul_blocked, NULL, 0},
    {"openmp_transpose_matmul", openmp_transpose_matmul, NULL, 0},
    {"blocked_transpose_matmul", blocked_transpose_matmul, NULL, 0},
    {"hybrid_transposed_matmul", hybrid_transposed_matmul, NULL, 0}
  };

  int num_functions = sizeof(functions) / sizeof(functions[0]);
  float tolerance = 1e-4f;

  printf("\nTesting all implementations:\n");
  printf("%-35s  %s\n", "Function", "Result");
  printf("%-35s  %s\n", "========", "======");

  for (int i = 0; i < num_functions; i++) {
    memset(test_out, 0, matrix_size * sizeof(float));

    if (functions[i].is_naive) { functions[i].func_naive(a, b, test_out, shape_a, shape_b, 2, matrix_size, matrix_size); }
    else { functions[i].func(a, b, test_out, shape_a, shape_b); }
    int passed = verify_and_debug(ref_out, test_out, size, size, functions[i].name, tolerance);
    if (!passed && size <= 256) { find_differences(ref_out, test_out, size, size, functions[i].name, tolerance); }
  }

  if (size <= 8) {
    print_matrix_sample(a, size, size, "Matrix A");
    print_matrix_sample(b, size, size, "Matrix B");
    print_matrix_sample(ref_out, size, size, "Reference Output");
  }

  free(a);
  free(b);
  free(ref_out);
  free(test_out);
  printf("\n");
}

void debug_rectangular_matrices() {
  printf("========================================\n");
  printf("DEBUGGING RECTANGULAR MATRICES\n");
  printf("========================================\n");

  int test_cases[][4] = {
    {128, 256, 256, 128},
    {256, 512, 512, 256},
    {512, 256, 256, 1024}
  };

  int num_cases = sizeof(test_cases) / sizeof(test_cases[0]);
  for (int case_idx = 0; case_idx < num_cases; case_idx++) {
    int rows_a = test_cases[case_idx][0], cols_a = test_cases[case_idx][1], rows_b = test_cases[case_idx][2], cols_b = test_cases[case_idx][3];
    printf("Testing A(%dx%d) * B(%dx%d) = C(%dx%d)\n", rows_a, cols_a, rows_b, cols_b, rows_a, cols_b);
    if (cols_a != rows_b) {
      printf("Skipping: incompatible dimensions\n\n");
      continue;
    }

    int shape_a[2] = {rows_a, cols_a}, shape_b[2] = {rows_b, cols_b};
    size_t size_a = rows_a * cols_a, size_b = rows_b * cols_b, size_out = rows_a * cols_b;
    float *a = randn_array(shape_a, size_a, 2), *b = randn_array(shape_b, size_b, 2);
    float *ref_out = (float*)calloc(size_out, sizeof(float)), *test_out = (float*)calloc(size_out, sizeof(float));

    if (!a || !b || !ref_out || !test_out) {
      printf("Memory allocation failed\n");
      continue;
    }

    naive_matmul(a, b, ref_out, shape_a, shape_b, 2, size_a, size_b);
    TestFunction functions[] = {
      {"blocked_matmul", blocked_matmul, NULL, 0},
      {"openmp_matmul", openmp_matmul, NULL, 0},
      {"simd_matmul", simd_matmul, NULL, 0},
      {"hybrid_parallel_matmul", hybrid_parallel_matmul, NULL, 0},
      {"transposed_matmul", transposed_matmul, NULL, 0},
      {"simd_transpose_matmul", simd_transpose_matmul, NULL, 0},
      {"simd_transpose_matmul_blocked", simd_transpose_matmul_blocked, NULL, 0},
      {"openmp_transpose_matmul", openmp_transpose_matmul, NULL, 0},
      {"hybrid_transposed_matmul", hybrid_transposed_matmul, NULL, 0}
    };

    int num_functions = sizeof(functions) / sizeof(functions[0]);
    float tolerance = 1e-4f;

    for (int i = 0; i < num_functions; i++) {
      memset(test_out, 0, size_out * sizeof(float));
      functions[i].func(a, b, test_out, shape_a, shape_b);
      verify_and_debug(ref_out, test_out, rows_a, cols_b, functions[i].name, tolerance);
    }

    free(a);
    free(b);
    free(ref_out);
    free(test_out);
    printf("\n");
  }
}

void debug_edge_cases() {
  printf("========================================\n");
  printf("DEBUGGING EDGE CASES\n");
  printf("========================================\n");

  printf("Testing small matrices (2x2, 4x4, 8x8):\n");
  int small_sizes[] = {2, 4, 8};
  int num_small = sizeof(small_sizes) / sizeof(small_sizes[0]);

  for (int i = 0; i < num_small; i++) {
    printf("\nSize %dx%d:\n", small_sizes[i], small_sizes[i]);
    debug_matrix_size(small_sizes[i]);
  }

  printf("Testing matrices with zeros:\n");
  int size = 64;
  int shape[2] = {size, size};
  size_t matrix_size = size * size;

  float* a = (float*)calloc(matrix_size, sizeof(float));
  float* b = randn_array(shape, matrix_size, 2);
  float *ref_out = (float*)calloc(matrix_size, sizeof(float)), *test_out = (float*)calloc(matrix_size, sizeof(float));

  if (a && b && ref_out && test_out) {
    naive_matmul(a, b, ref_out, shape, shape, 2, matrix_size, matrix_size);

    blocked_matmul(a, b, test_out, shape, shape);
    printf("Zero matrix A: %s\n", verify_and_debug(ref_out, test_out, size, size, "blocked_matmul", 1e-6f) ? "PASS" : "FAIL");
    memset(test_out, 0, matrix_size * sizeof(float));
    simd_matmul(a, b, test_out, shape, shape);
    printf("Zero matrix A: %s\n", verify_and_debug(ref_out, test_out, size, size, "simd_matmul", 1e-6f) ? "PASS" : "FAIL");
  }
  free(a);
  free(b);
  free(ref_out);
  free(test_out);
}

int main() {
  printf("Matrix Multiplication Manual Debug Suite\n");
  printf("=========================================\n");
  printf("OpenMP threads: %d\n", omp_get_max_threads());
  printf("Available processors: %d\n\n", omp_get_num_procs());

  debug_edge_cases();  
  debug_matrix_size(256);
  debug_matrix_size(1024);
  debug_rectangular_matrices();

  printf("========================================\n");
  printf("DEBUG SUMMARY\n");
  printf("========================================\n");
  printf("All functions tested against naive_matmul reference\n");
  printf("Tolerance used: 1e-4 for most tests, 1e-6 for edge cases\n");
  printf("PASS: Function produces correct results\n");
  printf("FAIL: Function has numerical differences beyond tolerance\n");
  printf("Check individual function outputs above for details\n");
  return 0;
}