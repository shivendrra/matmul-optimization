#ifndef __HELPERS__CUH__
#define __HELPERS__CUH__

#include <stddef.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "random.cuh"

__device__ static RNG* d_global_rng;
__device__ static int d_rng_initialized = 0;

static RNG* h_global_rng;
static int h_rng_initialized = 0;

__device__ static inline void ensure_rng_initialized_device() {
  if (!d_rng_initialized) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      if (d_global_rng) {
        rng_state(d_global_rng, cuda_thread_seed());
        d_rng_initialized = 1;
      }
    }
    __syncthreads();
  }
}

__host__ static inline void ensure_rng_initialized_host() {
  if (!h_rng_initialized) {
    if (!h_global_rng) {
      h_global_rng = (RNG*)malloc(sizeof(RNG));
    }
    rng_state(h_global_rng, current_time_seed());
    h_rng_initialized = 1;
  }
}

__global__ void init_global_rng_kernel(RNG* rng) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    d_global_rng = rng;
    rng_state(d_global_rng, cuda_thread_seed());
    d_rng_initialized = 1;
  }
}

__host__ static inline void init_device_rng() {
  static RNG* d_rng = nullptr;
  if (!d_rng) {
    cudaMalloc(&d_rng, sizeof(RNG));
    init_global_rng_kernel<<<1, 1>>>(d_rng);
    cudaDeviceSynchronize();
  }
}

__device__ __host__ static inline void fill_randn(float* out, size_t size) {
  if (!out) return;
  
#ifdef __CUDA_ARCH__
  ensure_rng_initialized_device();
  if (d_global_rng) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    RNG local_rng;
    rng_state(&local_rng, cuda_thread_seed() + tid);
    
    for (size_t i = tid; i < size; i += stride) {
      float temp;
      rng_randn(&local_rng, &temp, 1);
      out[i] = temp;
    }
  }
#else
  ensure_rng_initialized_host();
  if (h_global_rng) {
    rng_randn(h_global_rng, out, size);
  }
#endif
}

__host__ static inline float* randn_array_host(int* shape, size_t size, size_t ndim) {
  float* out = (float*)malloc(size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  fill_randn(out, size);
  return out;
}

__host__ static inline float* randn_array_device(int* shape, size_t size, size_t ndim) {
  float* d_out;
  cudaMalloc(&d_out, size * sizeof(float));
  
  init_device_rng();
  
  dim3 block(256);
  dim3 grid((size + block.x - 1) / block.x);
  
  fill_randn<<<grid, block>>>(d_out, size);
  cudaDeviceSynchronize();
  
  return d_out;
}

__device__ __host__ static inline float* randn_array(int* shape, size_t size, size_t ndim) {
#ifdef __CUDA_ARCH__
  return nullptr;
#else
  return randn_array_host(shape, size, ndim);
#endif
}

__device__ static inline void print_array_device(float* a, int* shape) {
  if (a == NULL) {
    printf("array(NULL)\n");
    return;
  }
  int rows = shape[0], cols = shape[1];
  for (int i = 0; i < rows; i++) {
    printf("| ");
    for (int j = 0; j < cols; j++) {
      int index = i * cols + j;
      printf("%.3f ", a[index]);
    }
    printf("|\n");
  }
}

__host__ static inline void print_array_host(float* a, int* shape) {
  if (a == NULL) {
    printf("array(NULL)\n");
    return;
  }
  int rows = shape[0], cols = shape[1];
  for (int i = 0; i < rows; i++) {
    printf("| ");
    for (int j = 0; j < cols; j++) {
      int index = i * cols + j;
      printf("%.3f ", a[index]);
    }
    printf("|\n");
  }
}

__device__ __host__ static inline void print_array(float* a, int* shape) {
#ifdef __CUDA_ARCH__
  print_array_device(a, shape);
#else
  print_array_host(a, shape);
#endif
}

__global__ void fill_randn_kernel(float* out, size_t size) {
  fill_randn(out, size);
}

__host__ static inline void launch_fill_randn(float* d_out, size_t size) {
  init_device_rng();
  
  dim3 block(256);
  dim3 grid((size + block.x - 1) / block.x);
  
  fill_randn_kernel<<<grid, block>>>(d_out, size);
  cudaDeviceSynchronize();
}

#endif