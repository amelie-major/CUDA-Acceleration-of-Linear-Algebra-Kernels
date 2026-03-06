#pragma once
void cpu_axpy(int n, float alpha, const float* x, float* y);
void cpu_vadd(int n, const float* x, const float* y, float* z);
void cpu_vcopy(int n, const float* x, float* z);
float cpu_reduce_sum(int n, const float* x);
void cpu_gemm(int M, int N, int K, const float* A, const float* B, float* C);
float max_abs_diff(int n, const float* a, const float* b);
