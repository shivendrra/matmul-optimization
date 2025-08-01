#ifndef __HELPERS__H__
#define __HELPERS__H__

#include <stddef.h>
#include <stdio.h>
#include "random.h"

static RNG global_rng;
static int rng_initialized = 0;

static inline void ensure_rng_initialized() {
  if (!rng_initialized) {
    rng_state(&global_rng, current_time_seed());
    rng_initialized = 1;
  }
}

void fill_randn(float* out, size_t size) {
  if (!out) return;
  ensure_rng_initialized();
  rng_randn(&global_rng, out, size);
}

float* randn_array(int* shape, size_t size, size_t ndim) {
  float* out = (float*)malloc(size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  fill_randn(out, size);
  return out;
}

void print_array(float* a, int* shape) {
  if (a == NULL) {
    printf("array(NULL)\n");
    return;
  }
  int rows = shape[0], cols = shape[1];
  for (int i = 0; i < rows; i++) {
    printf("| ");
    for (int j = 0; j < cols; j++) {
      int index = i * cols + j;
      printf("%3d ", a[index]);
    }
    printf("|\n");
  }
}


#endif  //!__HELPERS__H__