#ifndef __RANDOM__CUH__
#define __RANDOM__CUH__

#include <time.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

#define  PI_VAL  3.14159265358979323846f
#define  UINT64_CAST  0xFFFFFFFFFFFFFFFFULL
#define  UINT32_CAST  0xFFFFFFFFUL

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct {
  uint64_t state;
  uint64_t inc;
  float spare;
  int has_spare;
} RNG;

__device__ __host__ static inline uint32_t random_u32(RNG* rng) {
  uint64_t oldstate = rng->state;
  rng->state = oldstate * 6364136223846793005ULL + rng->inc;
  uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  uint32_t rot = oldstate >> 59u;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

__device__ __host__ static inline void rng_state(RNG* rng, uint64_t state) {
  rng->state = state;
  rng->inc = (state << 1) | 1;
  rng->has_spare = 0;
  rng->spare = 0.0f;
  random_u32(rng);
}

__device__ __host__ static inline float rng_random(RNG* rng) {
  return (random_u32(rng) >> 8) * 5.9604644775390625e-8f;
}

__device__ __host__ static inline float rng_uniform(RNG* rng, float a, float b) {
  return a + (b - a) * rng_random(rng);
}

__device__ __host__ static inline void rng_rand(RNG* rng, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    out[i] = rng_random(rng);
  }
}

__device__ __host__ static inline void rng_rand_uniform(RNG* rng, float* out, size_t size, float low, float high) {
  for (size_t i = 0; i < size; i++) {
    out[i] = rng_uniform(rng, low, high);
  }
}

__device__ __host__ static inline void rng_randint(RNG* rng, int* out, size_t size, int low, int high) {
  if (high <= low) return;
  uint32_t range = (uint32_t)(high - low);
  
  for (size_t i = 0; i < size; i++) {
    uint32_t threshold = (0x100000000ULL % range);
    uint32_t r;
    do {
      r = random_u32(rng);
    } while (r < threshold);
    out[i] = low + (int)(r % range);
  }
}

__device__ __host__ static inline void rng_randn(RNG* rng, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (rng->has_spare) {
      out[i] = rng->spare;
      rng->has_spare = 0;
    } else {
      float u1, u2;
      do {
        u1 = rng_random(rng);
      } while (u1 <= 1e-7f);
      u2 = rng_random(rng);
      
      float mag = sqrtf(-2.0f * logf(u1));
      float z0 = mag * cosf(2.0f * PI_VAL * u2);
      float z1 = mag * sinf(2.0f * PI_VAL * u2);
      
      out[i] = z0;
      rng->spare = z1;
      rng->has_spare = 1;
    }
  }
}

__device__ static inline void rng_choice_device(RNG* rng, int* a, int* out, size_t a_len, size_t size, int replace) {
  if (!replace && size > a_len) return;

  if (replace) {
    for (size_t i = 0; i < size; i++) {
      size_t idx = (size_t)(rng_random(rng) * a_len);
      if (idx >= a_len) idx = a_len - 1;
      out[i] = a[idx];
    }
  } else {
    for (size_t i = 0; i < size; i++) {
      size_t j = i + (size_t)(rng_random(rng) * (a_len - i));
      if (j >= a_len) j = a_len - 1;
      
      int tmp = a[i];
      a[i] = a[j];
      a[j] = tmp;
    }
    
    for (size_t i = 0; i < size; i++) out[i] = a[i];
  }
}

__host__ static inline void rng_choice_host(RNG* rng, int* a, int* out, size_t a_len, size_t size, int replace) {
  if (!replace && size > a_len) return;

  if (replace) {
    for (size_t i = 0; i < size; i++) {
      size_t idx = (size_t)(rng_random(rng) * a_len);
      if (idx >= a_len) idx = a_len - 1;
      out[i] = a[idx];
    }
  } else {
    int* temp = (int*)malloc(sizeof(int) * a_len);
    if (!temp) return;
    
    for (size_t i = 0; i < a_len; i++) temp[i] = a[i];
    
    for (size_t i = 0; i < size; i++) {
      size_t j = i + (size_t)(rng_random(rng) * (a_len - i));
      if (j >= a_len) j = a_len - 1;
      
      int tmp = temp[i];
      temp[i] = temp[j];
      temp[j] = tmp;
    }
    
    for (size_t i = 0; i < size; i++) out[i] = temp[i];
    free(temp);
  }
}

__device__ __host__ static inline void rng_choice(RNG* rng, int* a, int* out, size_t a_len, size_t size, int replace) {
#ifdef __CUDA_ARCH__
  rng_choice_device(rng, a, out, a_len, size, replace);
#else
  rng_choice_host(rng, a, out, a_len, size, replace);
#endif
}

__host__ static inline uint64_t current_time_seed() {
  uint64_t seed = (uint64_t)time(NULL);
  seed ^= (uint64_t)clock();
  seed ^= seed >> 32;
  seed *= 0x9E3779B97F4A7C15ULL;
  return seed;
}

__device__ static inline uint64_t cuda_thread_seed() {
  uint64_t seed = blockIdx.x * blockDim.x + threadIdx.x;
  seed ^= blockIdx.y * blockDim.y + threadIdx.y;
  seed ^= blockIdx.z * blockDim.z + threadIdx.z;
  seed *= 0x9E3779B97F4A7C15ULL;
  seed ^= seed >> 32;
  return seed;
}

#ifdef  __cplusplus
}
#endif

#endif