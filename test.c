#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "src/helpers.h"
#include "src/matmul.h"

typedef struct {
  const char* name;
  void (*func)(float*, float*, float*, int*, int*);
  int enabled;
} MatmulImplementation;

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

void run_comprehensive_benchmark(int rows_a, int cols_a, int rows_b, int cols_b) {
  printf("\n=== Matrix Multiplication Benchmark ===\n");
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
  
  MatmulImplementation implementations[] = {
    {"Naive (Reference)", (void*)naive_matmul, 1},
    {"OpenMP", openmp_matmul, 1},
    {"SIMD (AVX2)", simd_matmul, 1},
    {"Blocked", blocked_matmul, 1},
    {"Hybrid (OpenMP + SIMD + Blocked)", hybrid_parallel_matmul, 1},
  };
  
  int num_impls = sizeof(implementations) / sizeof(implementations[0]);
  clock_t times[num_impls];
  int correct[num_impls];
  
  printf("\nRunning benchmarks...\n");
  printf("%-35s %12s %12s %10s\n", "Implementation", "Time (s)", "Speedup", "Correct");
  printf("%-35s %12s %12s %10s\n", "==============", "========", "=======", "=======");
  
  for (int i = 0; i < num_impls; i++) {
    if (!implementations[i].enabled) continue;
    memset(out_test, 0, size_out * sizeof(float));
    
    clock_t start = clock();
    if (i == 0) { 
      naive_matmul(a, b, out_test, shape_a, shape_b, 2, size_a, size_b); 
    } else { 
      implementations[i].func(a, b, out_test, shape_a, shape_b); 
    }
    clock_t end = clock();
    
    times[i] = end - start;
    double time_seconds = ((double)times[i]) / CLOCKS_PER_SEC;

    if (i == 0) {
      memcpy(out_ref, out_test, size_out * sizeof(float));
      correct[i] = 1;
    } else { 
      correct[i] = verify_results(out_ref, out_test, size_out, 1e-4); 
    }
    
    double speedup = (i == 0) ? 1.0 : ((double)times[0]) / times[i];

    printf("%-35s %12.6f %12.2fx %10s\n", implementations[i].name, time_seconds, speedup, correct[i] ? "YES" : "NO");
  }
  
  printf("\nPerformance Analysis:\n");
  clock_t best_time = times[0];
  int best_idx = 0;
  
  for (int i = 1; i < num_impls; i++) {
    if (implementations[i].enabled && correct[i] && times[i] < best_time) {
      best_time = times[i];
      best_idx = i;
    }
  }
  
  printf("Best performing: %s (%.2fx speedup)\n", implementations[best_idx].name, ((double)times[0]) / best_time);
  
  double total_ops = 2.0 * rows_a * cols_a * cols_b;
  double best_time_seconds = ((double)best_time) / CLOCKS_PER_SEC;
  printf("GFLOPS (best): %.2f\n", total_ops / (best_time_seconds * 1e9));
  
  free(a);
  free(b);
  free(out_ref);
  free(out_test);
}

int main() {
  printf("Parallel Matrix Multiplication Benchmark Suite\n");
  printf("===============================================\n");
  
  printf("System Info:\n");
  printf("OpenMP threads: %d\n", omp_get_max_threads());
  printf("Available processors: %d\n", omp_get_num_procs());
  
  int test_sizes[][4] = {
    {128, 128, 128, 128},
    {256, 256, 256, 256},
    {512, 256, 256, 128},
    {512, 512, 512, 512},
    {1024, 512, 512, 256},
    {4096, 1024, 1024, 3072},
  };
  
  int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
  
  for (int i = 0; i < num_tests; i++) {
    run_comprehensive_benchmark(test_sizes[i][0], test_sizes[i][1], test_sizes[i][2], test_sizes[i][3]);
  }

  printf("\n=== Implementation Details ===\n");
  printf("1. Pthread: Manual thread management with row-wise parallelization\n");
  printf("2. OpenMP: Compiler-based parallelization with dynamic scheduling\n");
  printf("3. SIMD: AVX2 vectorization processing 8 floats simultaneously\n");
  printf("4. Blocked: Cache-friendly tiled computation (64x64 blocks)\n");
  printf("5. Hybrid: Combines OpenMP + SIMD + blocking for maximum performance\n");
  
  return 0;
}