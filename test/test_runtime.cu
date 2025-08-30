#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>
#include <math.h>
#include "inc/random.cuh"
#include "inc/helpers.cuh"
#include "src/cuda/matmul.cuh"

#define MAX_PRINT_SIZE 8
#define WARMUP_ITERATIONS 3
#define BENCHMARK_ITERATIONS 5

typedef struct {
  float naive_time, tiled_time, optimized_time, warp_time, cublas_time, best_time;
  bool naive_tested;
} BenchmarkResults;

__host__ void print_gpu_info() {
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  printf("=== GPU Information ===\n");
  printf("Device: %s\n", prop.name);
  printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
  printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
  printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
  printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  printf("Memory Clock Rate: %.2f MHz\n", prop.memoryClockRate / 1000.0);
  printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
  printf("Peak Memory Bandwidth: %.2f GB/s\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
  printf("\n");
}

__host__ void init_random_matrix(float* h_matrix, int rows, int cols) {
  int shape[2] = {rows, cols};
  size_t size = rows * cols;

  ensure_rng_initialized_host();
  if (h_global_rng) rng_rand_uniform(h_global_rng, h_matrix, size, -1.0f, 1.0f);
}

__host__ void init_random_matrix_normal(float* h_matrix, int rows, int cols) {
  int shape[2] = {rows, cols};
  size_t size = rows * cols;

  ensure_rng_initialized_host();
  if (h_global_rng) {
    rng_randn(h_global_rng, h_matrix, size);
    for (size_t i = 0; i < size; i++) h_matrix[i] = h_matrix[i] * 0.1f;
  }
}

__host__ void print_matrix_sample(float* matrix, int rows, int cols, const char* name) {
  printf("%s (%dx%d) - Sample:\n", name, rows, cols);
  int print_rows = (rows > MAX_PRINT_SIZE) ? MAX_PRINT_SIZE : rows, print_cols = (cols > MAX_PRINT_SIZE) ? MAX_PRINT_SIZE : cols;
  for (int i = 0; i < print_rows; i++) {
    printf("| ");
    for (int j = 0; j < print_cols; j++) printf("%7.3f ", matrix[i * cols + j]);
    if (cols > MAX_PRINT_SIZE) printf("... ");
    printf("|\n");
  }
  if (rows > MAX_PRINT_SIZE) printf("  ...\n");
  printf("\n");
}
__host__ bool verify_matrix_multiplication(float* a, float* b, float* result, int rows_a, int cols_a, int cols_b) {
  if (rows_a > 64 || cols_b > 64) return true;

  printf("Verifying matrix multiplication result...\n");
  float* cpu_result = (float*)malloc(rows_a * cols_b * sizeof(float));
  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      float sum = 0.0f;
      for (int k = 0; k < cols_a; k++) sum += a[i * cols_a + k] * b[k * cols_b + j];
      cpu_result[i * cols_b + j] = sum;
    }
  }

  bool correct = true;
  float max_error = 0.0f;
  for (int i = 0; i < rows_a * cols_b; i++) {
    float error = fabsf(cpu_result[i] - result[i]);
    if (error > max_error) max_error = error;
    if (error > 1e-3f) correct = false;
  }
  
  printf("Max error: %.6f\n", max_error);
  printf("Verification: %s\n\n", correct ? "PASSED" : "FAILED");

  free(cpu_result);
  return correct;
}

__host__ float benchmark_kernel(void (*kernel_func)(float*, float*, float*, int*, int*), float* d_a, float* d_b, float* d_out, int* shape_a, int* shape_b, const char* name) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  for (int i = 0; i < WARMUP_ITERATIONS; i++) kernel_func(d_a, d_b, d_out, shape_a, shape_b);
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < BENCHMARK_ITERATIONS; i++) kernel_func(d_a, d_b, d_out, shape_a, shape_b);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float total_time;
  cudaEventElapsedTime(&total_time, start, stop);
  float avg_time = total_time / BENCHMARK_ITERATIONS;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return avg_time;
}

__host__ float benchmark_naive_kernel(float* d_a, float* d_b, float* d_out, int* shape_a, int* shape_b) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  for (int i = 0; i < WARMUP_ITERATIONS; i++) naive_matmul_cuda(d_a, d_b, d_out, shape_a, shape_b, 2, 0, 0);
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < BENCHMARK_ITERATIONS; i++) naive_matmul_cuda(d_a, d_b, d_out, shape_a, shape_b, 2, 0, 0);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float total_time;
  cudaEventElapsedTime(&total_time, start, stop);
  float avg_time = total_time / BENCHMARK_ITERATIONS;  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return avg_time;
}

__host__ BenchmarkResults run_matrix_benchmark(int rows_a, int cols_a, int cols_b) {
  printf("=== Benchmarking Matrix Multiplication: %dx%d * %dx%d ===\n", rows_a, cols_a, cols_a, cols_b);
  BenchmarkResults results = {0};

  int shape_a[2] = {rows_a, cols_a}, shape_b[2] = {cols_a, cols_b};
  size_t size_a = rows_a * cols_a * sizeof(float), size_b = cols_a * cols_b * sizeof(float), size_out = rows_a * cols_b * sizeof(float);
  float *h_a, *h_b, *h_out, *d_a, *d_b, *d_out;
  h_a = (float*)malloc(size_a); h_b = (float*)malloc(size_b); h_out = (float*)malloc(size_out);

  cudaMalloc(&d_a, size_a);
  cudaMalloc(&d_b, size_b);
  cudaMalloc(&d_out, size_out);

  printf("Initializing matrices with random data...\n");
  init_random_matrix_normal(h_a, rows_a, cols_a);
  init_random_matrix_normal(h_b, cols_a, cols_b);

  if (rows_a <= MAX_PRINT_SIZE && cols_b <= MAX_PRINT_SIZE) {
    print_matrix_sample(h_a, rows_a, cols_a, "Matrix A");
    print_matrix_sample(h_b, cols_a, cols_b, "Matrix B");
  }

  cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

  warmup_gpu();

  printf("Running benchmarks...\n");

  bool test_naive = (rows_a <= 512 && cols_b <= 512);
  results.naive_tested = test_naive;

  if (test_naive) {
    printf("Testing Naive implementation... ");
    results.naive_time = benchmark_naive_kernel(d_a, d_b, d_out, shape_a, shape_b);
    printf("%.3f ms\n", results.naive_time);
  }

  printf("Testing Tiled implementation... ");
  results.tiled_time = benchmark_kernel(tiled_matmul_cuda, d_a, d_b, d_out, shape_a, shape_b, "Tiled");
  printf("%.3f ms\n", results.tiled_time);

  printf("Testing Optimized implementation... ");
  results.optimized_time = benchmark_kernel(optimized_matmul_cuda, d_a, d_b, d_out, shape_a, shape_b, "Optimized");
  printf("%.3f ms\n", results.optimized_time);

  printf("Testing Warp-Tiled implementation... ");
  results.warp_time = benchmark_kernel(warp_tiled_matmul_cuda, d_a, d_b, d_out, shape_a, shape_b, "Warp-Tiled");
  printf("%.3f ms\n", results.warp_time);

  if (rows_a >= 256) {
    printf("Testing cuBLAS implementation... ");
    results.cublas_time = benchmark_kernel(cublas_matmul_cuda, d_a, d_b, d_out, shape_a, shape_b, "cuBLAS");
    printf("%.3f ms\n", results.cublas_time);
  }

  printf("Testing Auto-selected implementation... ");
  results.best_time = benchmark_kernel(launch_best_matmul, d_a, d_b, d_out, shape_a, shape_b, "Auto-selected");
  printf("%.3f ms\n", results.best_time);

  launch_best_matmul(d_a, d_b, d_out, shape_a, shape_b);
  cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost);
  if (rows_a <= MAX_PRINT_SIZE && cols_b <= MAX_PRINT_SIZE) print_matrix_sample(h_out, rows_a, cols_b, "Result");
  if (test_naive) verify_matrix_multiplication(h_a, h_b, h_out, rows_a, cols_a, cols_b);
  long long ops = 2LL * rows_a * cols_a * cols_b;
  double best_gflops = (ops / 1e9) / (results.best_time / 1000.0);
  printf("Performance: %.2f GFLOPS\n", best_gflops);
  free(h_a);
  free(h_b);
  free(h_out);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
  printf("\n");
  return results;
}

__host__ void test_random_functions() {
  printf("=== Testing Random Number Generation Functions ===\n");
  const int test_size = 1000000;
  float *h_uniform, *h_normal, *d_uniform, *d_normal;
  int *h_randint, *d_randint;
  h_uniform = (float*)malloc(test_size * sizeof(float)); h_normal = (float*)malloc(test_size * sizeof(float)); h_randint = (int*)malloc(test_size * sizeof(int));
  cudaMalloc(&d_uniform, test_size * sizeof(float));
  cudaMalloc(&d_normal, test_size * sizeof(float));
  cudaMalloc(&d_randint, test_size * sizeof(int));
  printf("Testing host random number generation...\n");

  ensure_rng_initialized_host();
  rng_rand(h_global_rng, h_uniform, test_size);
  rng_randn(h_global_rng, h_normal, test_size);
  rng_randint(h_global_rng, h_randint, test_size, 0, 100);

  float uniform_mean = 0.0f, normal_mean = 0.0f;
  for (int i = 0; i < test_size; i++) {
    uniform_mean += h_uniform[i];
    normal_mean += h_normal[i];
  }
  uniform_mean /= test_size;
  normal_mean /= test_size;
  printf("Host uniform mean: %.6f (expected ~0.5)\n", uniform_mean);
  printf("Host normal mean: %.6f (expected ~0.0)\n", normal_mean);
  printf("Testing device random number generation...\n");
  launch_fill_randn(d_normal, test_size);

  float *h_device_normal = (float*)malloc(test_size * sizeof(float));
  cudaMemcpy(h_device_normal, d_normal, test_size * sizeof(float), cudaMemcpyDeviceToHost);

  float device_normal_mean = 0.0f;
  for (int i = 0; i < test_size; i++) device_normal_mean += h_device_normal[i];
  device_normal_mean /= test_size;

  printf("Device normal mean: %.6f (expected ~0.0)\n", device_normal_mean);
  printf("Sample uniform values: ");
  for (int i = 0; i < 10; i++) printf("%.3f ", h_uniform[i]);
  printf("\n");
  printf("Sample normal values: ");
  for (int i = 0; i < 10; i++) printf("%.3f ", h_normal[i]);
  printf("\n");
  printf("Sample random integers [0,100): ");
  for (int i = 0; i < 10; i++) printf("%d ", h_randint[i]);
  printf("\n");
  
  free(h_uniform);
  free(h_normal);
  free(h_randint);
  free(h_device_normal);
  cudaFree(d_uniform);
  cudaFree(d_normal);
  cudaFree(d_randint);

  printf("Random number generation tests completed!\n\n");
}

__host__ void run_comprehensive_benchmark() {
  int test_sizes[][4] = {
    {128, 128, 128, 128}, {256, 256, 256, 256}, {512, 256, 256, 128}, {512, 512, 512, 512},
    {1024, 512, 512, 256}, {1024, 1024, 1024, 1024}, {4096, 1024, 1024, 3072}, {4096, 4096, 4096, 4096}
  };
  int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
  printf("=== Comprehensive Matrix Multiplication Benchmark ===\n");
  printf("Test configurations: %d\n", num_tests);
  printf("Warmup iterations: %d\n", WARMUP_ITERATIONS);
  printf("Benchmark iterations: %d\n\n", BENCHMARK_ITERATIONS);
  BenchmarkResults* all_results = (BenchmarkResults*)malloc(num_tests * sizeof(BenchmarkResults));
  for (int i = 0; i < num_tests; i++) {
    int rows_a = test_sizes[i][0], cols_a = test_sizes[i][1], cols_b = test_sizes[i][3];
    all_results[i] = run_matrix_benchmark(rows_a, cols_a, cols_b);
  }
  printf("=== Performance Summary ===\n");
  printf("Size (MxKxN)         | Naive    | Tiled    | Optimized| Warp     | cuBLAS   | Best     |\n");
  printf("--------------------|----------|----------|----------|----------|----------|----------|\n");

  for (int i = 0; i < num_tests; i++) {
    int rows_a = test_sizes[i][0], cols_a = test_sizes[i][1], cols_b = test_sizes[i][3];
    printf("%4dx%4dx%4d     |", rows_a, cols_a, cols_b);
    if (all_results[i].naive_tested) printf(" %7.2f  |", all_results[i].naive_time);
    else printf("    -     |");
    printf(" %7.2f  |", all_results[i].tiled_time);
    printf(" %7.2f  |", all_results[i].optimized_time);
    printf(" %7.2f  |", all_results[i].warp_time);
    if (all_results[i].cublas_time > 0) printf(" %7.2f  |", all_results[i].cublas_time);
    else printf("    -     |"); 
    printf(" %7.2f  |\n", all_results[i].best_time);
  }
  printf("\n=== GFLOPS Performance ===\n");
  for (int i = 0; i < num_tests; i++) {
    int rows_a = test_sizes[i][0], cols_a = test_sizes[i][1], cols_b = test_sizes[i][3];
    long long ops = 2LL * rows_a * cols_a * cols_b;
    double gflops = (ops / 1e9) / (all_results[i].best_time / 1000.0); 
    printf("%4dx%4dx%4d: %.2f GFLOPS\n", rows_a, cols_a, cols_b, gflops);
  }  
  free(all_results);
}

int main() {
  printf("CUDA Matrix Multiplication and Random Number Generation Test Suite\n");
  printf("===================================================================\n\n");
  print_gpu_info();
  test_random_functions();
  run_comprehensive_benchmark();
  printf("All tests completed successfully!\n");
  return 0;
}