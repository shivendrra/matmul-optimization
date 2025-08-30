// gcc -O3 -march=native -fopenmp test_block.c src/naive_matmul.c src/trans_matmul.c src/kernels/blocked.c src/kernels/openmp.c src/kernels/simd.c -o test_block -lm
// test_block

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include "src/matmul.h"
#include "src/inc/helpers.h"

void create_logs_directory() {
  struct stat st = {0};
  if (stat("logs", &st) == -1) {
    #ifdef _WIN32
      mkdir("logs");
    #else
      mkdir("logs", 0755);
    #endif
  }
}

float* create_matrix(int rows, int cols) {
  float* matrix = (float*)malloc(rows * cols * sizeof(float));
  if (!matrix) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  fill_randn(matrix, rows * cols);
  return matrix;
}

double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

void run_hybrid_test(FILE* log_file, const char* method_name, void (*matmul_func)(float*, float*, float*, int*, int*), float* a, float* b, float* out, int* shape_a, int* shape_b, int block_size, int warmup_runs, int test_runs) {

  for (int w = 0; w < warmup_runs; w++) { matmul_func(a, b, out, shape_a, shape_b); }
  double total_time = 0.0;
  for (int t = 0; t < test_runs; t++) {
    double start_time = get_time();
    matmul_func(a, b, out, shape_a, shape_b);
    double end_time = get_time();
    total_time += (end_time - start_time);
  }

  double avg_time = total_time / test_runs;
  double gflops = (2.0 * shape_a[0] * shape_a[1] * shape_b[1]) / (avg_time * 1e9);

  fprintf(log_file, "%s,block_%d,%dx%d,%dx%d,%.6f,%.3f\n", method_name, block_size, shape_a[0], shape_a[1], shape_b[0], shape_b[1], avg_time, gflops);
  printf("%s (block=%d): %dx%d @ %dx%d -> %.3f ms, %.3f GFLOPS\n", method_name, block_size, shape_a[0], shape_a[1], shape_b[0], shape_b[1], avg_time * 1000, gflops);
}

int main() {
  create_logs_directory();
  
  FILE* log_file = fopen("logs/batch_testing.log", "w");
  if (!log_file) {
    fprintf(stderr, "Failed to create log file\n");
    return 1;
  }

  fprintf(log_file, "method,block_size,matrix_a_shape,matrix_b_shape,time_seconds,gflops\n");

  int matrix_sizes[] = {256, 512, 1024, 3072, 4096};
  int block_sizes[] = {8, 16, 32, 64, 128, 256, 512};
  int num_matrix_sizes = sizeof(matrix_sizes) / sizeof(matrix_sizes[0]);
  int num_block_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);

  int warmup_runs = 2;
  int test_runs = 5;

  printf("Matrix Multiplication Performance Tests - Hybrid Methods Only\n");
  printf("Block sizes: ");
  for (int i = 0; i < num_block_sizes; i++) printf("%d ", block_sizes[i]);
  printf("\nMatrix sizes: ");
  for (int i = 0; i < num_matrix_sizes; i++) printf("%d ", matrix_sizes[i]);
  printf("\n\n");

  for (int size_idx = 0; size_idx < num_matrix_sizes; size_idx++) {
    int size = matrix_sizes[size_idx];
    printf("=== Testing %dx%d matrices ===\n", size, size);

    int shape_square_a[2] = {size, size}, shape_square_b[2] = {size, size}, shape_rect_a[2] = {size, size/2}, shape_rect_b[2] = {size/2, size};

    float* a_square = create_matrix(size, size), *b_square = create_matrix(size, size), *out_square = (float*)malloc(size * size * sizeof(float));
    float *a_rect = create_matrix(size, size/2), *b_rect = create_matrix(size/2, size), *out_rect = (float*)malloc(size * size * sizeof(float));
    if (!out_square || !out_rect) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(EXIT_FAILURE);
    }
    for (int block_idx = 0; block_idx < num_block_sizes; block_idx++) {
      int block_size = block_sizes[block_idx];
      if (block_size > size) continue;
      printf("\nBlock Size: %d\n", block_size);
      printf("Square (%dx%d):\n", size, size);
      run_hybrid_test(log_file, "hybrid_parallel", hybrid_parallel_matmul, a_square, b_square, out_square, shape_square_a, shape_square_b, block_size, warmup_runs, test_runs);
      run_hybrid_test(log_file, "hybrid_transposed", hybrid_transposed_matmul, a_square, b_square, out_square, shape_square_a, shape_square_b, block_size, warmup_runs, test_runs);
      printf("Rectangular (%dx%d @ %dx%d):\n", size, size/2, size/2, size);
      run_hybrid_test(log_file, "hybrid_parallel", hybrid_parallel_matmul, a_rect, b_rect, out_rect, shape_rect_a, shape_rect_b, block_size, warmup_runs, test_runs);
      run_hybrid_test(log_file, "hybrid_transposed", hybrid_transposed_matmul, a_rect, b_rect, out_rect, shape_rect_a, shape_rect_b, block_size, warmup_runs, test_runs);
    }
    
    free(a_square);
    free(b_square);
    free(out_square);
    free(a_rect);
    free(b_rect);
    free(out_rect);
  }
  fclose(log_file);
  printf("Testing completed. Results in logs/batch_testing.log\n");
  return 0;
}