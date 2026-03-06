#include "kernels.hpp"
#include "check_cuda.hpp"
#include <cuda_runtime.h>

// AXPY baseline
__global__ void k_axpy(int n, float alpha, const float* __restrict__ x, float* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = alpha * x[i] + y[i];
}

// AXPY launcher
void launch_axpy(int n, float alpha, const float* x, float* y, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    k_axpy<<<grid, block, 0, stream>>>(n, alpha, x, y);
    CUDA_CHECK_LAST("k_axpy");
}

// Vector addition
__global__ void k_vadd(int n, const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = x[i] + y[i];
}

// Vector addition launcher
void launch_vadd(int n, const float* x, const float* y, float* z, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    k_vadd<<<grid, block, 0, stream>>>(n, x, y, z);
    CUDA_CHECK_LAST("k_vadd");
}
// Vector copy
__global__ void k_vcopy(int n, const float* __restrict__ x, float* __restrict__ z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = x[i];
}
// Vector copy launcher

void launch_vcopy(int n, const float* x, float* z, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    k_vcopy<<<grid, block, 0, stream>>>(n, x, z);
    CUDA_CHECK_LAST("k_vcopy");
}

// Reduction: two-stage (intra-block tree + warp-level shuffle, then iterative global accumulation)

__global__ void k_reduce_block_sum(const float* __restrict__ x, float* __restrict__ partials, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? x[i] : 0.0f;
    __syncthreads();

    // Tree reduction down to 32 threads (one warp), syncing after each step
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // Warp-level reduction via shuffle – no __syncthreads() needed within a warp
    if (tid < 32) {
        float val = sdata[tid];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (tid == 0) partials[blockIdx.x] = val;
    }
}

float gpu_reduce_sum(const float* d_x, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    float* d_partials = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partials, grid * sizeof(float)));
    k_reduce_block_sum<<<grid, block, 0, stream>>>(d_x, d_partials, n);
    CUDA_CHECK_LAST("k_reduce_block_sum");

    int cur = grid;
    while (cur > 1) {
        int next_grid = (cur + block - 1) / block;
        float* d_next = nullptr;
        CUDA_CHECK(cudaMalloc(&d_next, next_grid * sizeof(float)));
        k_reduce_block_sum<<<next_grid, block, 0, stream>>>(d_partials, d_next, cur);
        CUDA_CHECK_LAST("k_reduce_block_sum(final)");
        CUDA_CHECK(cudaFree(d_partials));
        d_partials = d_next;
        cur = next_grid;
    }

    float h = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&h, d_partials, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_partials));
    return h;
}

// Naive GEMM
__global__ void k_gemm_naive(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k=0;k<K;++k) acc += A[row*K + k] * B[k*N + col];
        C[row*N + col] = acc;
    }
}

void launch_gemm_naive(int M, int N, int K, const float* A, const float* B, float* C, cudaStream_t stream) {
    dim3 block(16,16);
    dim3 grid((N+15)/16, (M+15)/16);
    k_gemm_naive<<<grid, block, 0, stream>>>(M,N,K,A,B,C);
    CUDA_CHECK_LAST("k_gemm_naive");
}

// Tiled GEMM (16x16 baseline)
template<int TILE>
__global__ void k_gemm_tiled(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t=0; t < (K + TILE - 1)/TILE; ++t) {
        int a_col = t*TILE + threadIdx.x;
        int b_row = t*TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row*K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row*N + col] : 0.0f;

        __syncthreads();
        #pragma unroll
        for (int k=0;k<TILE;++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row*N + col] = acc;
}

void launch_gemm_tiled16(int M, int N, int K, const float* A, const float* B, float* C, cudaStream_t stream) {
    dim3 block(16,16);
    dim3 grid((N+15)/16, (M+15)/16);
    k_gemm_tiled<16><<<grid, block, 0, stream>>>(M,N,K,A,B,C);
    CUDA_CHECK_LAST("k_gemm_tiled<16>");
}
