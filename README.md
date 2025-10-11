# MatrixMultiplication Optimization

I developed a matrix-multiplication trick in high school which by chance can improve computation speeds by many times than regular matrix multiplication techniques. So in this repository I've created certain scripts to check & benchmark the preformances of each method.

## Performance Results

During my testing I found out transposing matrix worked quite good for OpenMP, SIMD & Blocked matmul but lacked certain times in Hybrid matmul. Check out the logs, run it yourself on your system.
This was tested on HP Envy x360: AMD ryzen 5 4500u 8GB ram.

![1.png](https://github.com/shivendrra/matmul-optimization/blob/main/media/1.png)
![2.png](https://github.com/shivendrra/matmul-optimization/blob/main/media/2.png)
![4.png](https://github.com/shivendrra/matmul-optimization/blob/main/media/4.png)
![3.png](https://github.com/shivendrra/matmul-optimization/blob/main/media/3.png)

*Interactive performance analysis showing execution time and GFLOPS across different matrix sizes and implementations.*

## Project Structure

```bash
Matmul/
├── log/
│   ├── debug.log
│   ├── block.log
│   └── test.log
├── media/                    # media files, pics, etc
├── src/
│   ├── naive_matmul.c          # Basic matrix multiplication
│   ├── trans_matmul.c          # Transposed matrix operations
│   ├── matmul.h
│   ├── cuda/                # cuda-kernels, not yet written
│   ├── inc/                # some header files
│   └── kernels/
│       ├── blocked.c           # Cache-optimized blocked algorithms
│       ├── openmp.c            # OpenMP parallel implementations
│       └── simd.c              # AVX2 SIMD vectorized kernels
├── test/                      # testing codes
├── debug.c                    # Precision benchmarking
└── README.md
```

## Build Requirements

- **Compiler:** GCC with C11 support
- **Required Flags:**
  - `-O3`: Maximum optimization
  - `-fopenmp`: OpenMP support
  - `-mavx2`: AVX2 instruction set
  - `-mfma`: Fused multiply-add operations
  - `-std=c11`: C11 standard

## Do it on your own

### Build the Project

```bash
# Clone the repository
git clone https://github.com/shivendrra/matmul-optimization
cd matmul

# Compile with optimizations
gcc -O3 -fopenmp -mavx2 -mfma -std=c11 -o test test/test_runtime.c \
    src/naive_matmul.c src/trans_matmul.c \
    src/kernels/blocked.c src/kernels/openmp.c src/kernels/simd.c -lm

# Run the benchmark suite
./test
```

Benchmark results: [Benchmark.md](https://github.com/shivendrra/matmul-optimization/blob/main/Benchmark.md)

**System Requirements:** x86-64 processor with AVX2 support, GCC compiler, OpenMP library

**Performance Note:** Results may vary based on CPU architecture, cache size, and system configuration. The provided benchmarks were obtained on a 6-core system with AVX2 support.

a research project by Shivendra