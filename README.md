# MatrixMultiplication Optimization

I developed a matrix-multiplication trick in high school which by chance can improve computation speeds by many times than regular matrix multiplication techniques. So in this repository I've created certain scripts to check & benchmark the preformances of each method.

## Performance Results

During my testing I found out transposing matrix worked quite good for OpenMP, SIMD & Blocked matmul but lacked certain times in Hybrid matmul. Check out the logs, run it yourself on your system.
This was tested on HP Envy x360: AMD ryzen 5 4500u 8GB ram.

![1.png](https://github.com/shivendrra/matmul-optimization/blob/main/media/1.png)
![2.png](https://github.com/shivendrra/matmul-optimization/blob/main/media/2.png)
![3.png](https://github.com/shivendrra/matmul-optimization/blob/main/media/3.png)

*Interactive performance analysis showing execution time and GFLOPS across different matrix sizes and implementations.*

## 🏗️ Project Structure

```bash
Matmul/
├── log/
│   ├── debug.log
│   └── test.log
├── src/
│   ├── naive_matmul.c          # Basic matrix multiplication
│   ├── trans_matmul.c          # Transposed matrix operations
│   ├── matmul.h                # matmul header file
│   └── kernels/
│       ├── blocked.c           # Cache-optimized blocked algorithms
│       ├── openmp.c            # OpenMP parallel implementations
│       └── simd.c              # AVX2 SIMD vectorized kernels
├── test.c                      # Main benchmark suite
├── debug.c                    # Precision benchmarking
└── README.md                   # This file
```

## Build Requirements

- **Compiler:** GCC with C11 support
- **Required Flags:**
  - `-O3`: Maximum optimization
  - `-fopenmp`: OpenMP support
  - `-mavx2`: AVX2 instruction set
  - `-mfma`: Fused multiply-add operations
  - `-std=c11`: C11 standard

## Quick Start

### Building the Project

```bash
# Clone the repository
git clone https://github.com/shivendrra/matmul-optimization
cd matmul

# Compile with optimizations
gcc -O3 -fopenmp -mavx2 -mfma -std=c11 -o test test.c \
    src/naive_matmul.c src/trans_matmul.c \
    src/kernels/blocked.c src/kernels/openmp.c src/kernels/simd.c -lm

# Run the benchmark suite
./test
```

## Benchmark Results Overview

Based on our comprehensive testing, here are the key performance insights:

### Performance Hierarchy

1. **Hybrid (OpenMP+SIMD+Blocked)**: Up to **95x speedup**
   - Best overall performance across all matrix sizes
   - Combines parallel processing, vectorization, and cache optimization

2. **SIMD (AVX2)**: Up to **9.55x speedup**
   - Excellent for smaller to medium matrices
   - Processes 8 floats simultaneously

3. **OpenMP**: Up to **5.67x speedup**
   - Effectiveness increases with matrix size
   - Multi-core parallelization

4. **Blocked**: Up to **2.20x speedup**
   - Improves cache locality
   - Foundation for hybrid approaches

5. **Naive**: Baseline reference (1.00x)

### Standard vs Transposed Performance

| Matrix Size | Standard Winner | Transposed Winner | Performance Difference |
|-------------|----------------|-------------------|----------------------|
| 128×128     | Hybrid (3.00x) | SIMD (∞x)        | Transposed dominates |
| 256×256     | Hybrid (14.67x)| SIMD Blocked (7.00x) | Standard wins |
| 512×512     | Hybrid (45.94x)| Hybrid (8.14x)   | Standard wins |
| 1024×1024   | Hybrid (85.33x)| Hybrid (6.74x)   | Standard wins |
| 4096×1024   | Hybrid (95.00x)| Hybrid (11.11x)  | Standard wins |

## Test Cases

The benchmark suite includes 7 comprehensive test cases:

1. **128×128** - Small square matrices
2. **256×256** - Medium square matrices  
3. **512×256** - Rectangular matrices
4. **512×512** - Large square matrices
5. **1024×512** - Large rectangular matrices
6. **1024×1024** - Very large square matrices
7. **4096×1024** - Massive rectangular matrices

Each test case evaluates:

- All implementation variants
- Execution time and speedup calculations
- GFLOPS performance metrics
- Correctness verification
- Standard vs transposed comparisons

## Acknowledgments

- OpenMP specification contributors
- Intel AVX2 instruction set documentation
- Matrix multiplication optimization research community
- Performance benchmarking best practices

## References

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [OpenMP Specification](https://www.openmp.org/specifications/)
- [Cache-Oblivious Algorithms](https://en.wikipedia.org/wiki/Cache-oblivious_algorithm)
- [BLAS (Basic Linear Algebra Subprograms)](http://www.netlib.org/blas/)

---

**System Requirements:** x86-64 processor with AVX2 support, GCC compiler, OpenMP library

**Performance Note:** Results may vary based on CPU architecture, cache size, and system configuration. The provided benchmarks were obtained on a 6-core system with AVX2 support.
