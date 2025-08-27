#ifndef __MATMUL__CUH__
#define __MATMUL__CUH__

#include <cuda_runtime.h>
#include <cublas_v2.h>
#define TILE_SIZE 32
#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define THREAD_TILE_M 8
#define THREAD_TILE_N 8
#define THREAD_TILE_K 8
#define CUDA_CHECK(call) check_cuda_error(__FILE__, __LINE__)

extern "C" {
  __global__ void naive_matmul_kernel(float* a, float* b, float* out, int rows_a, int cols_a, int cols_b);
  __global__ void tiled_matmul_kernel(float* a, float* b, float* out, int rows_a, int cols_a, int cols_b);
  __global__ void optimized_matmul_kernel(float* a, float* b, float* out, int rows_a, int cols_a, int cols_b);
  __global__ void warp_tiled_matmul_kernel(float* a, float* b, float* out, int rows_a, int cols_a, int cols_b);

  __host__ void naive_matmul_cuda(float* a, float* b, float* out, int* shape1, int* shape2, size_t ndim, size_t size1, size_t size2);
  __host__ void tiled_matmul_cuda(float* a, float* b, float* out, int* shape1, int* shape2);
  __host__ void optimized_matmul_cuda(float* a, float* b, float* out, int* shape1, int* shape2);
  __host__ void warp_tiled_matmul_cuda(float* a, float* b, float* out, int* shape1, int* shape2);
  __host__ void cublas_matmul_cuda(float* a, float* b, float* out, int* shape1, int* shape2);
  __host__ void launch_best_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b);
  __host__ float* matmul_host_wrapper(float* h_a, float* h_b, int* shape_a, int* shape_b);
  __host__ cudaError_t check_cuda_error(const char* file, int line);
  __host__ void print_device_properties();
  __host__ void warmup_gpu();
}

#endif