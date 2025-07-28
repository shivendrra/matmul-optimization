// gcc -o test test.c src/naive_matmul.c src/trans_matmul.c -lm
// ./test

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "src/helpers.h"
#include "src/matmul.h"

typedef struct {
  int rows;
  int cols;
} MatrixSize;

double get_time_diff(clock_t start, clock_t end) {
  return ((double)(end - start)) / CLOCKS_PER_SEC;
}

void run_benchmark(MatrixSize size_a, MatrixSize size_b, const char* test_name) {
  printf("\n=== %s ===\n", test_name);
  printf("Matrix A: %dx%d, Matrix B: %dx%d\n", size_a.rows, size_a.cols, size_b.rows, size_b.cols);
  
  if (size_a.cols != size_b.rows) {
    printf("Error: Invalid matrix dimensions for multiplication\n");
    return;
  }
  
  int shape_a[2] = {size_a.rows, size_a.cols};
  int shape_b[2] = {size_b.rows, size_b.cols};
  int shape_out[2] = {size_a.rows, size_b.cols};
  
  size_t size_a_total = size_a.rows * size_a.cols;
  size_t size_b_total = size_b.rows * size_b.cols;
  size_t size_out_total = size_a.rows * size_b.cols;
  
  float* a = randn_array(shape_a, size_a_total, 2);
  float* b = randn_array(shape_b, size_b_total, 2);
  float* out_naive = (float*)malloc(size_out_total * sizeof(float));
  float* out_optimized = (float*)malloc(size_out_total * sizeof(float));
  
  if (!out_naive || !out_optimized) {
    printf("Memory allocation failed\n");
    return;
  }
  
  memset(out_naive, 0, size_out_total * sizeof(float));
  memset(out_optimized, 0, size_out_total * sizeof(float));
  
  clock_t start, end;
  
  start = clock();
  naive_matmul(a, b, out_naive, shape_a, shape_b, 2, size_a_total, size_b_total);
  end = clock();
  double naive_time = get_time_diff(start, end);
  
  start = clock();
  optimized_ops(a, b, out_optimized, shape_a, shape_b);
  end = clock();
  double optimized_time = get_time_diff(start, end);
  
  printf("Naive implementation:     %.6f seconds\n", naive_time);
  printf("Optimized implementation: %.6f seconds\n", optimized_time);
  printf("Speedup: %.2fx\n", naive_time / optimized_time);
  
  int results_match = 1;
  for (size_t i = 0; i < size_out_total && results_match; i++) {
    if (fabs(out_naive[i] - out_optimized[i]) > 1e-5) {
      results_match = 0;
    }
  }
  printf("Results match: %s\n", results_match ? "YES" : "NO");
  
  free(a);
  free(b);
  free(out_naive);
  free(out_optimized);
}

void print_small_matrix_test() {
  printf("\n=== Small Matrix Verification ===\n");
  
  int shape_a[2] = {3, 4};
  int shape_b[2] = {4, 3};
  
  float* a = randn_array(shape_a, 12, 2);
  float* b = randn_array(shape_b, 12, 2);
  float* out_naive = (float*)calloc(9, sizeof(float));
  float* out_optimized = (float*)calloc(9, sizeof(float));
  
  printf("Matrix A (3x4):\n");
  print_array(a, shape_a);
  printf("\nMatrix B (4x3):\n");
  print_array(b, shape_b);
  
  naive_matmul(a, b, out_naive, shape_a, shape_b, 2, 12, 12);
  optimized_ops(a, b, out_optimized, shape_a, shape_b);
  
  int shape_out[2] = {3, 3};
  printf("\nNaive Result (3x3):\n");
  print_array(out_naive, shape_out);
  printf("\nOptimized Result (3x3):\n");
  print_array(out_optimized, shape_out);
  
  free(a);
  free(b);
  free(out_naive);
  free(out_optimized);
}

int main() {
  printf("Matrix Multiplication Performance Benchmark\n");
  printf("============================================\n");
  
  print_small_matrix_test();
  
  MatrixSize test_cases[][2] = {
    {{64, 64}, {64, 64}},
    {{128, 64}, {64, 32}},
    {{100, 200}, {200, 150}},
    {{256, 256}, {256, 256}},
    {{512, 128}, {128, 64}},
    {{300, 400}, {400, 500}},
    {{1024, 512}, {512, 256}},
    {{800, 600}, {600, 400}}
  };
  
  const char* test_names[] = {
    "Small Square (64x64)",
    "Small Rectangle (128x64 @ 64x32)",
    "Medium Rectangle (100x200 @ 200x150)",
    "Medium Square (256x256)",
    "Medium Rectangle (512x128 @ 128x64)",
    "Large Rectangle (300x400 @ 400x500)",
    "Large Rectangle (1024x512 @ 512x256)",
    "Large Rectangle (800x600 @ 600x400)"
  };
  
  int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
  
  for (int i = 0; i < num_tests; i++) {
    run_benchmark(test_cases[i][0], test_cases[i][1], test_names[i]);
  }
  
  printf("\n=== Summary ===\n");
  printf("Tested %d different matrix size combinations\n", num_tests);
  printf("All tests include both square and rectangular matrices\n");
  printf("The optimized implementation uses matrix transposition\n");
  printf("to improve cache locality and memory access patterns.\n");
  
  return 0;
}