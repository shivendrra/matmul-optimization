#ifndef __MATMUL__CUH__
#define __MATMUL__CUH__

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include "matmul.cuh"

__global__ void naive_matmul_kernel(float* a, float* b, float* out, int rows_a, int cols_a, int cols_b) {
  int row = blockIdx.y * blockDim.y + threadIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows_a && col < cols_b) {
    float sum = 0.0f;
    for (int k = 0; k < cols_a; k++) sum += a[row * cols_a + k] * b[k * cols_b + col];
    out[row * cols_b + col] = sum;
  }
}

__global__ void tiled_matmul_kernel(float* a, float* b, float* out, int rows_a, int cols_a, int cols_b) {
  __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
  __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * TILE_SIZE + ty, col = blockIdx.x * TILE_SIZE + tx;
  float sum = 0.0f;
  for (int tile = 0; tile < (cols_a + TILE_SIZE - 1) / TILE_SIZE; tile++) {
    if (row < rows_a && tile * TILE_SIZE + tx < cols_a) { tile_a[ty][tx] = a[row * cols_a + tile * TILE_SIZE + tx]; }
    else { tile_a[ty][tx] = 0.0f; }  
    if (col < cols_b && tile * TILE_SIZE + ty < cols_a) { tile_b[ty][tx] = b[(tile * TILE_SIZE + ty) * cols_b + col]; }
    else { tile_b[ty][tx] = 0.0f; }

    __syncthreads();
    for (int k = 0; k < TILE_SIZE; k++) sum += tile_a[ty][k] * tile_b[k][tx];
    __syncthreads();
  }
  if (row < rows_a && col < cols_b) out[row * cols_b + col] = sum;
}

__global__ void optimized_matmul_kernel(float* a, float* b, float* out, int rows_a, int cols_a, int cols_b) {
  __shared__ float tile_a[TILE_SIZE][TILE_SIZE + 1];
  __shared__ float tile_b[TILE_SIZE][TILE_SIZE + 1];

  int tx = threadIdx.x, ty = threadIdx.y;
  int block_row = blockIdx.y, block_col = blockIdx.x;
  int row = block_row * TILE_SIZE + ty, col = block_col * TILE_SIZE + tx;

  float reg_a[THREAD_TILE_M], reg_b[THREAD_TILE_N], reg_c[THREAD_TILE_M][THREAD_TILE_N] = {0};
  for (int tile = 0; tile < (cols_a + TILE_SIZE - 1) / TILE_SIZE; tile++) {
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i += blockDim.y) {
      if (row + i < rows_a && tile * TILE_SIZE + tx < cols_a) { tile_a[ty + i][tx] = a[(row + i) * cols_a + tile * TILE_SIZE + tx]; }
      else { tile_a[ty + i][tx] = 0.0f; }
    }

    #pragma unroll
    for (int j = 0; j < THREAD_TILE_N; j += blockDim.x) {
      if (col + j < cols_b && tile * TILE_SIZE + ty < cols_a) { tile_b[ty][tx + j] = b[(tile * TILE_SIZE + ty) * cols_b + col + j]; }
      else { tile_b[ty][tx + j] = 0.0f; }
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
      #pragma unroll
      for (int i = 0; i < THREAD_TILE_M; i++) reg_a[i] = tile_a[ty + i][k];
      #pragma unroll
      for (int j = 0; j < THREAD_TILE_N; j++) reg_b[j] = tile_b[k][tx + j];
      #pragma unroll
      for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) reg_c[i][j] += reg_a[i] * reg_b[j];
      }
    }

    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < THREAD_TILE_M; i++) {
    #pragma unroll
    for (int j = 0; j < THREAD_TILE_N; j++) { if (row + i < rows_a && col + j < cols_b) out[(row + i) * cols_b + col + j] = reg_c[i][j]; }
  }
}

__global__ void warp_tiled_matmul_kernel(float* a, float* b, float* out, int rows_a, int cols_a, int cols_b) {
  __shared__ float tile_a[TILE_SIZE][TILE_SIZE + 1];
  __shared__ float tile_b[TILE_SIZE][TILE_SIZE + 1];

  int warp_id = threadIdx.x / WARP_SIZE, lane_id = threadIdx.x % WARP_SIZE;
  int block_row = blockIdx.y * (blockDim.y * THREAD_TILE_M / WARP_SIZE), block_col = blockIdx.x * (blockDim.x * THREAD_TILE_N / WARP_SIZE);
  int warp_row = block_row + (warp_id / (TILE_SIZE / THREAD_TILE_N)) * THREAD_TILE_M, warp_col = block_col + (warp_id % (TILE_SIZE / THREAD_TILE_N)) * THREAD_TILE_N;
  int thread_row = warp_row + (lane_id / THREAD_TILE_N), thread_col = warp_col + (lane_id % THREAD_TILE_N);
  float reg_c[THREAD_TILE_M][THREAD_TILE_N] = {0};

  for (int tile = 0; tile < (cols_a + TILE_SIZE - 1) / TILE_SIZE; tile++) {
    if (threadIdx.y * TILE_SIZE + threadIdx.x < TILE_SIZE * TILE_SIZE) {
      int load_row = threadIdx.y * TILE_SIZE + threadIdx.x;
      int a_row = blockIdx.y * TILE_SIZE + load_row / TILE_SIZE, a_col = tile * TILE_SIZE + load_row % TILE_SIZE;
      int b_row = tile * TILE_SIZE + load_row / TILE_SIZE, b_col = blockIdx.x * TILE_SIZE + load_row % TILE_SIZE;

      tile_a[load_row / TILE_SIZE][load_row % TILE_SIZE] = (a_row < rows_a && a_col < cols_a) ? a[a_row * cols_a + a_col] : 0.0f;
      tile_b[load_row / TILE_SIZE][load_row % TILE_SIZE] = (b_row < cols_a && b_col < cols_b) ? b[b_row * cols_b + b_col] : 0.0f;
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
      float reg_a[THREAD_TILE_M], reg_b[THREAD_TILE_N];
      #pragma unroll
      for (int i = 0; i < THREAD_TILE_M; i++) reg_a[i] = tile_a[thread_row + i - warp_row][k];
      #pragma unroll
      for (int j = 0; j < THREAD_TILE_N; j++) reg_b[j] = tile_b[k][thread_col + j - warp_col];
      #pragma unroll
      for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) reg_c[i][j] += reg_a[i] * reg_b[j];
      }
    }

    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < THREAD_TILE_M; i++) {
    #pragma unroll
    for (int j = 0; j < THREAD_TILE_N; j++) {
      int out_row = thread_row + i, out_col = thread_col + j;
      if (out_row < rows_a && out_col < cols_b) out[out_row * cols_b + out_col] = reg_c[i][j];
    }
  }
}

__host__ void naive_matmul_cuda(float* a, float* b, float* out, int* shape1, int* shape2, size_t ndim, size_t size1, size_t size2) {
  int rows_a = shape1[0], cols_a = shape1[1], cols_b = shape2[1];
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((cols_b + block.x - 1) / block.x, (rows_a + block.y - 1) / block.y);
  naive_matmul_kernel<<<grid, block>>>(a, b, out, rows_a, cols_a, cols_b);
  cudaDeviceSynchronize();
}

__host__ void tiled_matmul_cuda(float* a, float* b, float* out, int* shape1, int* shape2) {
  int rows_a = shape1[0], cols_a = shape1[1], cols_b = shape2[1];
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((cols_b + TILE_SIZE - 1) / TILE_SIZE, (rows_a + TILE_SIZE - 1) / TILE_SIZE);
  tiled_matmul_kernel<<<grid, block>>>(a, b, out, rows_a, cols_a, cols_b);
  cudaDeviceSynchronize();
}

__host__ void optimized_matmul_cuda(float* a, float* b, float* out, int* shape1, int* shape2) {
  int rows_a = shape1[0], cols_a = shape1[1], cols_b = shape2[1];
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((cols_b + TILE_SIZE - 1) / TILE_SIZE, (rows_a + TILE_SIZE - 1) / TILE_SIZE);
  optimized_matmul_kernel<<<grid, block>>>(a, b, out, rows_a, cols_a, cols_b);
  cudaDeviceSynchronize();
}

__host__ void warp_tiled_matmul_cuda(float* a, float* b, float* out, int* shape1, int* shape2) {
  int rows_a = shape1[0], cols_a = shape1[1], cols_b = shape2[1];
  dim3 block(128, 4);
  dim3 grid((cols_b + TILE_SIZE - 1) / TILE_SIZE, (rows_a + TILE_SIZE - 1) / TILE_SIZE);
  warp_tiled_matmul_kernel<<<grid, block>>>(a, b, out, rows_a, cols_a, cols_b);
  cudaDeviceSynchronize();
}

__host__ void cublas_matmul_cuda(float* a, float* b, float* out, int* shape1, int* shape2) {
  int rows_a = shape1[0], cols_a = shape1[1], cols_b = shape2[1];
  cublasHandle_t handle;
  cublasCreate(&handle);
  const float alpha = 1.0f, beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols_b, rows_a, cols_a, &alpha, b, cols_b, a, cols_a, &beta, out, cols_b);
  cublasDestroy(handle);
  cudaDeviceSynchronize();
}

__host__ static inline void launch_best_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int rows_a = shape_a[0], cols_a = shape_a[1], cols_b = shape_b[1];
  
  if (rows_a <= 64 && cols_b <= 64) { naive_matmul_cuda(a, b, out, shape_a, shape_b, 2, 0, 0); }
  else if (rows_a <= 512 && cols_b <= 512) { tiled_matmul_cuda(a, b, out, shape_a, shape_b); }
  else if (rows_a <= 2048 && cols_b <= 2048) { optimized_matmul_cuda(a, b, out, shape_a, shape_b); }
  else { cublas_matmul_cuda(a, b, out, shape_a, shape_b); }
}

__host__ static inline float* matmul_host_wrapper(float* h_a, float* h_b, int* shape_a, int* shape_b) {
  int rows_a = shape_a[0], cols_a = shape_a[1], cols_b = shape_b[1];
  size_t size_a = rows_a * cols_a * sizeof(float), size_b = cols_a * cols_b * sizeof(float), size_out = rows_a * cols_b * sizeof(float);
  float *d_a, *d_b, *d_out, *h_out;
  cudaMalloc(&d_a, size_a);
  cudaMalloc(&d_b, size_b);
  cudaMalloc(&d_out, size_out);
  h_out = (float*)malloc(size_out);

  cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
  launch_best_matmul(d_a, d_b, d_out, shape_a, shape_b);
  cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
  return h_out;
}

#endif