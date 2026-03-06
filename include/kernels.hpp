#pragma once
#include <cuda_runtime.h>

// Vector
void launch_axpy(int n, float alpha, const float* x, float* y, cudaStream_t stream=0);
void launch_vadd(int n, const float* x, const float* y, float* z, cudaStream_t stream=0);
void launch_vcopy(int n, const float* x, float* z, cudaStream_t stream=0);
// Reduction
float gpu_reduce_sum(const float* d_x, int n, cudaStream_t stream=0);

// GEMM
void launch_gemm_naive(int M, int N, int K, const float* A, const float* B, float* C, cudaStream_t stream=0);
void launch_gemm_tiled16(int M, int N, int K, const float* A, const float* B, float* C, cudaStream_t stream=0);
