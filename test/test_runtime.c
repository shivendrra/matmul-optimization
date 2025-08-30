// gcc -O3 -fopenmp -mavx2 -mfma -std=c11 -o test test.c src/naive_matmul.c src/trans_matmul.c src/kernels/blocked.c src/kernels/openmp.c src/kernels/simd.c -lm
// ./test.exe

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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
  int enabled;
} MatmulImplementation;

typedef struct {
  const char* category;
  MatmulImplementation* implementations;
  int count;
} MatmulCategory;

double get_time_diff_precise(struct timespec start, struct timespec end) {
  return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int verify_results(float* ref, float* test, size_t size, float tolerance) {
  for (size_t i = 0; i < size; i++) {
    if (fabs(ref[i] - test[i]) > tolerance) {
      return 0;
    }
  }
  return 1;
}

// Function to determine if naive should be enabled based on matrix size
int should_enable_naive(int rows_a, int cols_a, int rows_b, int cols_b) {
  // Disable naive for matrices larger than 512 in any dimension
  return (rows_a <= 512 && cols_a <= 512 && rows_b <= 512 && cols_b <= 512);
}

void run_category_benchmark(MatmulCategory* category, int rows_a, int cols_a, int rows_b, int cols_b) {
  printf("\n=== %s Implementations ===\n", category->category);
  printf("A: %dx%d, B: %dx%d, Output: %dx%d\n", rows_a, cols_a, rows_b, cols_b, rows_a, cols_b);
  
  if (cols_a != rows_b) {
    printf("Error: Matrix dimensions incompatible\n");
    return;
  }
  
  int shape_a[2] = {rows_a, cols_a};
  int shape_b[2] = {rows_b, cols_b};
  size_t size_a = rows_a * cols_a;
  size_t size_b = rows_b * cols_b;
  size_t size_out = rows_a * cols_b;
  
  float* a = randn_array(shape_a, size_a, 2);
  float* b = randn_array(shape_b, size_b, 2);
  float* out_ref = (float*)calloc(size_out, sizeof(float));
  float* out_test = (float*)calloc(size_out, sizeof(float));
  
  if (!a || !b || !out_ref || !out_test) {
    printf("Memory allocation failed\n");
    return;
  }
  
  clock_t times[category->count];
  int correct[category->count];
  int reference_idx = -1;
  
  // Disable naive for large matrices
  int enable_naive = should_enable_naive(rows_a, cols_a, rows_b, cols_b);
  if (!enable_naive) {
    printf("Note: Naive implementation disabled for large matrices (>512 in any dimension)\n");
  }
  
  printf("\n%-35s %12s %12s %10s\n", "Implementation", "Time (s)", "Speedup", "Correct");
  printf("%-35s %12s %12s %10s\n", "==============", "========", "=======", "=======");
  
  for (int i = 0; i < category->count; i++) {
    if (!category->implementations[i].enabled) continue;
    
    // Skip naive implementation for large matrices
    if (category->implementations[i].is_naive && !enable_naive) {
      printf("%-35s %12s %12s %10s\n", category->implementations[i].name, "SKIPPED", "-", "-");
      continue;
    }
    
    memset(out_test, 0, size_out * sizeof(float));
    
    clock_t start = clock();
    
    if (category->implementations[i].is_naive) {
      category->implementations[i].func_naive(a, b, out_test, shape_a, shape_b, 2, size_a, size_b);
    } else {
      category->implementations[i].func(a, b, out_test, shape_a, shape_b);
    }
    
    clock_t end = clock();
    
    times[i] = end - start;
    double time_seconds = ((double)times[i]) / CLOCKS_PER_SEC;
    
    // Set reference for correctness checking
    if (reference_idx == -1) {
      memcpy(out_ref, out_test, size_out * sizeof(float));
      correct[i] = 1;
      reference_idx = i;
    } else { 
      correct[i] = verify_results(out_ref, out_test, size_out, 1e-4); 
    }

    double speedup = (reference_idx == i) ? 1.0 : ((double)times[reference_idx]) / times[i];
    printf("%-35s %12.6f %12.2fx %10s\n", category->implementations[i].name, time_seconds, speedup, correct[i] ? "YES" : "NO");
  }
  
  // Find best performing implementation
  clock_t best_time = (clock_t)-1; // Maximum clock_t value
  int best_idx = -1;
  
  for (int i = 0; i < category->count; i++) {
    if (!category->implementations[i].enabled) continue;
    if (category->implementations[i].is_naive && !enable_naive) continue;
    if (correct[i] && times[i] < best_time) {
      best_time = times[i];
      best_idx = i;
    }
  }

  if (best_idx != -1 && reference_idx != -1) {
    printf("Best: %s (%.2fx speedup)\n", category->implementations[best_idx].name, 
           ((double)times[reference_idx]) / best_time);
    
    double total_ops = 2.0 * rows_a * cols_a * cols_b;
    double best_time_seconds = ((double)best_time) / CLOCKS_PER_SEC;
    printf("GFLOPS: %.2f\n", total_ops / (best_time_seconds * 1e9));
  }
  
  free(a);
  free(b);
  free(out_ref);
  free(out_test);
}

void run_comparison_benchmark(int rows_a, int cols_a, int rows_b, int cols_b) {
  printf("\n=== STANDARD vs TRANSPOSED COMPARISON ===\n");
  printf("A: %dx%d, B: %dx%d, Output: %dx%d\n", rows_a, cols_a, rows_b, cols_b, rows_a, cols_b);
  
  if (cols_a != rows_b) {
    printf("Error: Matrix dimensions incompatible\n");
    return;
  }
  
  int shape_a[2] = {rows_a, cols_a};
  int shape_b[2] = {rows_b, cols_b};
  size_t size_a = rows_a * cols_a;
  size_t size_b = rows_b * cols_b;
  size_t size_out = rows_a * cols_b;
  
  float* a = randn_array(shape_a, size_a, 2);
  float* b = randn_array(shape_b, size_b, 2);
  float* out_std = (float*)calloc(size_out, sizeof(float));
  float* out_trans = (float*)calloc(size_out, sizeof(float));

  if (!a || !b || !out_std || !out_trans) {
    printf("Memory allocation failed\n");
    return;
  }

  typedef struct {
    const char* name;
    void (*std_func)(float*, float*, float*, int*, int*);
    void (*trans_func)(float*, float*, float*, int*, int*);
    void (*std_naive)(float*, float*, float*, int*, int*, size_t, size_t, size_t);
    int is_naive;
  } ComparisonPair;
  
  ComparisonPair pairs[] = {
    {"Naive", NULL, transposed_matmul, naive_matmul, 1},
    {"OpenMP", openmp_matmul, openmp_transpose_matmul, NULL, 0},
    {"SIMD", simd_matmul, simd_transpose_matmul, NULL, 0},
    {"Blocked", blocked_matmul, blocked_transpose_matmul, NULL, 0},
    {"Hybrid", hybrid_parallel_matmul, hybrid_transposed_matmul, NULL, 0}
  };
  int num_pairs = sizeof(pairs) / sizeof(pairs[0]);
  
  int enable_naive = should_enable_naive(rows_a, cols_a, rows_b, cols_b);
  if (!enable_naive) {
    printf("Note: Naive comparison disabled for large matrices (>512 in any dimension)\n");
  }
  
  printf("\n%-15s %12s %12s %12s %12s %10s\n", "Method", "Std Time(s)", "Trans Time(s)", "Std GFLOPS", "Trans GFLOPS", "Winner");
  printf("%-15s %12s %12s %12s %12s %10s\n", "======", "===========", "============", "==========", "===========", "======");
  
  double total_ops = 2.0 * rows_a * cols_a * cols_b;
  
  for (int i = 0; i < num_pairs; i++) {
    // Skip naive comparison for large matrices
    if (pairs[i].is_naive && !enable_naive) {
      printf("%-15s %12s %12s %12s %12s %10s\n", pairs[i].name, "SKIPPED", "SKIPPED", "-", "-", "-");
      continue;
    }
    
    memset(out_std, 0, size_out * sizeof(float));
    memset(out_trans, 0, size_out * sizeof(float));
    
    clock_t std_start = clock();
    if (pairs[i].is_naive) {
      pairs[i].std_naive(a, b, out_std, shape_a, shape_b, 2, size_a, size_b);
    } else {
      pairs[i].std_func(a, b, out_std, shape_a, shape_b);
    }
    clock_t std_end = clock();

    clock_t trans_start = clock();
    pairs[i].trans_func(a, b, out_trans, shape_a, shape_b);
    clock_t trans_end = clock();

    double std_time = ((double)(std_end - std_start)) / CLOCKS_PER_SEC;
    double trans_time = ((double)(trans_end - trans_start)) / CLOCKS_PER_SEC;

    double std_gflops = total_ops / (std_time * 1e9);
    double trans_gflops = total_ops / (trans_time * 1e9);

    int results_match = verify_results(out_std, out_trans, size_out, 1e-4);
    const char* winner = (std_time < trans_time) ? "Standard" : "Transposed";
    printf("%-15s %12.6f %12.6f %12.2f %12.2f %10s%s\n", 
           pairs[i].name, std_time, trans_time, std_gflops, trans_gflops, 
           winner, results_match ? "" : " (MISMATCH!)");
  }
  
  free(a);
  free(b);
  free(out_std);
  free(out_trans);
}

void run_comprehensive_benchmark(int rows_a, int cols_a, int rows_b, int cols_b) {
  int enable_naive = should_enable_naive(rows_a, cols_a, rows_b, cols_b);
  
  MatmulImplementation standard_impls[] = {
    {"Naive (Reference)", NULL, naive_matmul, 1, enable_naive},
    {"OpenMP", openmp_matmul, NULL, 0, 1},
    {"SIMD (AVX2)", simd_matmul, NULL, 0, 1},
    {"Blocked", blocked_matmul, NULL, 0, 1},
    {"Hybrid (OpenMP+SIMD+Blocked)", hybrid_parallel_matmul, NULL, 0, 1},
  };
  
  MatmulImplementation transposed_impls[] = {
    {"Transposed (Reference)", transposed_matmul, NULL, 0, 1},
    {"OpenMP Transposed", openmp_transpose_matmul, NULL, 0, 1},
    {"SIMD Transposed", simd_transpose_matmul, NULL, 0, 1},
    {"SIMD Blocked Transposed", simd_transpose_matmul_blocked, NULL, 0, 1},
    {"Blocked Transposed", blocked_transpose_matmul, NULL, 0, 1},
    {"Hybrid Transposed", hybrid_transposed_matmul, NULL, 0, 1},
  };
  
  MatmulCategory categories[] = {
    {"Standard Matrix Multiplication", standard_impls, 5},
    {"Transposed Matrix Multiplication", transposed_impls, 6}
  };
  
  for (int i = 0; i < 2; i++) {
    run_category_benchmark(&categories[i], rows_a, cols_a, rows_b, cols_b);
  }
  
  run_comparison_benchmark(rows_a, cols_a, rows_b, cols_b);
}

int main() {
  printf("Matrix Multiplication Comprehensive Benchmark Suite\n");
  printf("===================================================\n");
  
  printf("System Info:\n");
  printf("OpenMP threads: %d\n", omp_get_max_threads());
  printf("Available processors: %d\n", omp_get_num_procs());
  
  int test_sizes[][4] = {
    {128, 128, 128, 128},    // Small - includes naive
    {256, 256, 256, 256},    // Medium - includes naive
    {512, 256, 256, 128},    // Medium - includes naive
    {512, 512, 512, 512},    // Large - includes naive (borderline)
    {1024, 512, 512, 256},   // Large - skips naive
    {1024, 1024, 1024, 1024}, // Very large - skips naive
    {4096, 1024, 1024, 3072}, // Huge - skips naive
    {4096, 4096, 4096, 4096}  // Largest - skips naive
  };
  
  int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
  
  for (int i = 0; i < num_tests; i++) {
    printf("\n" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("TEST %d/%d\n", i + 1, num_tests);
    run_comprehensive_benchmark(test_sizes[i][0], test_sizes[i][1], test_sizes[i][2], test_sizes[i][3]);
  }

  printf("\n" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
  printf("SUMMARY\n");
  printf("Standard implementations: naive (small matrices only), openmp, simd, blocked, hybrid\n");
  printf("Transposed implementations: transposed, openmp_transpose, simd_transpose, simd_transpose_blocked, blocked_transpose, hybrid_transposed\n");
  printf("Direct comparisons show performance differences between standard and transposed approaches\n");
  printf("SIMD operations process 8 floats simultaneously using AVX2\n");
  printf("Blocked algorithms use 64x64 tiles for cache efficiency\n");
  printf("Hybrid combines OpenMP + SIMD + blocking for maximum performance\n");
  printf("Naive implementation automatically disabled for matrices >512 in any dimension\n");
  return 0;
}