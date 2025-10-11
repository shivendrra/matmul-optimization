# Benchmark Results Overview

Based on our comprehensive testing, here are the key performance insights:

## Performance Hierarchy

1. **Hybrid (OpenMP+SIMD+Blocked)**: Up to **66.69x speedup**
   - Best overall performance across all matrix sizes
   - Combines parallel processing, vectorization, and cache optimization
   - Peak performance: 52.88 GFLOPS (transposed hybrid on 4096×4096)

2. **SIMD (AVX2)**: Up to **8.18x speedup**
   - Excellent for smaller to medium matrices
   - Processes 8 floats simultaneously
   - Consistent performance across different matrix sizes

3. **OpenMP**: Up to **4.72x speedup**
   - Effectiveness increases with matrix size
   - Multi-core parallelization with 6 threads
   - Better performance with transposed implementations

4. **Blocked**: Up to **2.00x speedup**
   - Mixed results when used alone
   - Critical component for cache optimization in hybrid approaches
   - Uses 64×64 tiles for cache efficiency

5. **Naive**: Baseline reference (1.00x)
   - Disabled automatically for matrices >512 in any dimension

## Standard vs Transposed Performance

| Matrix Size | Standard Best | Transposed Best | Winner | Performance Gap |
|-------------|---------------|-----------------|--------|-----------------|
| 128×128     | Hybrid (∞x)   | Hybrid (∞x)     | Tie    | Both sub-millisecond |
| 256×256     | Hybrid (33.55 GFLOPS) | Hybrid (16.78 GFLOPS) | Standard | 2.0x advantage |
| 512×256     | Hybrid (16.78 GFLOPS) | Hybrid (16.78 GFLOPS) | Tie | Equal performance |
| 512×512     | Hybrid (20.65 GFLOPS) | Hybrid (33.56 GFLOPS) | Transposed | 1.6x advantage |
| 1024×512    | Hybrid (22.37 GFLOPS) | Hybrid (20.65 GFLOPS) | Standard | 1.1x advantage |
| 1024×1024   | Hybrid (21.05 GFLOPS) | Hybrid (41.30 GFLOPS) | Transposed | 2.0x advantage |
| 4096×1024   | Hybrid (22.41 GFLOPS) | Hybrid (47.55 GFLOPS) | Transposed | 2.1x advantage |
| 4096×4096   | Hybrid (17.66 GFLOPS) | Hybrid (51.73 GFLOPS) | Transposed | 2.9x advantage |

## Key Findings

- **Transposed implementations** win in **5 out of 8** test cases, with dominant performance on large matrices
- **Standard implementations** excel on smaller matrices (256×256 and 1024×512) but not by a very big margin
- **Cache locality** benefits of transposition are most apparent in medium-sized matrices
- **Correctness issues** appear in some highly optimized implementations at large matrix sizes (small <e-5 floating point errors)
- **Performance scaling** is non-linear, with sweet spots around 1024×1024 matrices

## Performance Notes

**Accuracy Warnings**: Some optimized implementations show correctness issues (marked as "NO") at very large matrix sizes, particularly:

- SIMD implementations on matrices >= 1024×1024
- Hybrid transposed implementations on matrices >= 512×512

**Reliability**: OpenMP and basic blocked implementations maintain correctness across all tested sizes.

## Test Cases

The benchmark suite includes 7 comprehensive test cases:

1. **128×128** - Small square matrices
2. **256×256** - Medium square matrices  
3. **512×256** - Rectangular matrices
4. **512×512** - Large square matrices
5. **1024×512** - Large rectangular matrices
6. **1024×1024** - Very large square matrices
7. **4096×1024** - Massive rectangular matrices
8. **4096×4096** - Massive square matrices

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
