/*
 * COMP3741 – Coursework: self-contained correctness tests
 * Run with:  mpirun -np <P> ./mpi_cuda_tests
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <mpi.h>
#include <cuda_runtime.h>

#include "mpi_utils.hpp"
#include "check_cuda.hpp"
#include "kernels.hpp"
#include "cpu_reference.hpp"

static int g_pass = 0, g_fail = 0;

static void report(const char* name, bool ok) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
    ok ? ++g_pass : ++g_fail;
}

// Helper: run a GEMM kernel, compare against CPU reference, return max error
static float test_gemm_kernel(int m, int n, int k,
                              const float* A, const float* B,
                              const float* Cref,
                              bool use_tiled, cudaStream_t stream) {
    float *dA=nullptr, *dB=nullptr, *dC=nullptr;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)m*k*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)k*n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)m*n*sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(dA, A, (size_t)m*k*sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dB, B, (size_t)k*n*sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(dC, 0, (size_t)m*n*sizeof(float), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (use_tiled) launch_gemm_tiled16(m, n, k, dA, dB, dC, stream);
    else           launch_gemm_naive(m, n, k, dA, dB, dC, stream);

    std::vector<float> C(m*n);
    CUDA_CHECK(cudaMemcpyAsync(C.data(), dC, (size_t)m*n*sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float err = max_abs_diff(m*n, C.data(), Cref);
    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    return err;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    auto info = mpi_info();

    // All ranks use GPU 0 (single-GPU assumption)
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) { std::cerr << "No GPU found\n"; MPI_Abort(MPI_COMM_WORLD,1); }
    CUDA_CHECK(cudaSetDevice(0));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int N = 1 << 18;   // 256 K elements – quick to run
    auto d = dist_1d(N, info.rank, info.size);

    // ── Test 1: AXPY ─────────────────────────────────────────────────────────
    {
        std::vector<float> x(d.N_local, 1.f), y(d.N_local, 2.f), yref(d.N_local, 2.f);
        const float alpha = 3.f;
        cpu_axpy((int)d.N_local, alpha, x.data(), yref.data());  // ref: y = 3*1+2 = 5

        float *dx = nullptr, *dy = nullptr;
        CUDA_CHECK(cudaMalloc(&dx, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(dx, x.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dy, y.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        launch_axpy((int)d.N_local, alpha, dx, dy, stream);
        CUDA_CHECK(cudaMemcpyAsync(y.data(), dy, d.N_local*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float err = max_abs_diff((int)d.N_local, y.data(), yref.data());
        if (info.rank == 0) report("AXPY correctness", err < 1e-5f);
        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy));
    }

    // ── Test 2: ADD (z = x + y) ──────────────────────────────────────────────
    {
        std::vector<float> x(d.N_local, 1.5f), y(d.N_local, 2.5f);
        std::vector<float> z(d.N_local), zref(d.N_local);
        cpu_vadd((int)d.N_local, x.data(), y.data(), zref.data());  // ref: 4.0

        float *dx=nullptr, *dy=nullptr, *dz=nullptr;
        CUDA_CHECK(cudaMalloc(&dx, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dz, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(dx, x.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dy, y.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        launch_add((int)d.N_local, dx, dy, dz, stream);
        CUDA_CHECK(cudaMemcpyAsync(z.data(), dz, d.N_local*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float err = max_abs_diff((int)d.N_local, z.data(), zref.data());
        if (info.rank == 0) report("ADD correctness", err < 1e-5f);
        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dz));
    }

    // ── Test 3: COPY (y = x) ─────────────────────────────────────────────────
    {
        std::vector<float> x(d.N_local, 3.14f), y(d.N_local, 0.f);

        float *dx=nullptr, *dy=nullptr;
        CUDA_CHECK(cudaMalloc(&dx, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(dx, x.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        launch_copy((int)d.N_local, dx, dy, stream);
        CUDA_CHECK(cudaMemcpyAsync(y.data(), dy, d.N_local*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float err = max_abs_diff((int)d.N_local, y.data(), x.data());
        if (info.rank == 0) report("COPY correctness", err == 0.0f);  // must be bitwise exact
        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy));
    }

    // ── Test 4: Global reduction (constant) ──────────────────────────────────
    {
        // All elements == 1  →  global sum == N
        std::vector<float> x(d.N_local, 1.f);
        float* dx = nullptr;
        CUDA_CHECK(cudaMalloc(&dx, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(dx, x.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float local  = gpu_reduce_sum(dx, (int)d.N_local, stream);
        float global = 0.f;
        MPI_Allreduce(&local, &global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        if (info.rank == 0)
            report("Reduction global sum (constant)", std::fabs(global - (float)N) < 0.5f);
        CUDA_CHECK(cudaFree(dx));
    }

    // ── Test 5: Reduction with random data, multiple seeds ───────────────────
    {
        const int seeds[] = {42, 123, 7777, 99999};
        bool all_ok = true;
        for (int seed : seeds) {
            std::mt19937 gen(seed + info.rank);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            std::vector<float> x(d.N_local);
            for (auto& v : x) v = dist(gen);

            float* dx = nullptr;
            CUDA_CHECK(cudaMalloc(&dx, d.N_local*sizeof(float)));
            CUDA_CHECK(cudaMemcpyAsync(dx, x.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            float gpu_local = gpu_reduce_sum(dx, (int)d.N_local, stream);
            float gpu_global = 0.f;
            MPI_Allreduce(&gpu_local, &gpu_global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

            float cpu_local = cpu_reduce_sum((int)d.N_local, x.data());
            float cpu_global = 0.f;
            MPI_Allreduce(&cpu_local, &cpu_global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

            // Tolerance: O(sqrt(N)) * eps for random float summation
            float tol = std::sqrt((float)N) * 1e-3f;
            if (std::fabs(gpu_global - cpu_global) >= tol) all_ok = false;

            CUDA_CHECK(cudaFree(dx));
        }
        if (info.rank == 0)
            report("Reduction multi-seed GPU vs CPU", all_ok);
    }

    // ── Test 6: Naive GEMM correctness ───────────────────────────────────────
    {
        const int m = 64, n = 64, k = 64;
        std::vector<float> A(m*k, 0.f), B(k*n, 0.f), Cref(m*n, 0.f);
        for (int i = 0; i < m && i < k; ++i) A[i*k+i] = 1.f;
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < n; ++j) B[i*n+j] = (float)(i+j);
        cpu_gemm(m, n, k, A.data(), B.data(), Cref.data());

        float err = test_gemm_kernel(m, n, k, A.data(), B.data(), Cref.data(), false, stream);
        if (info.rank == 0) report("Naive GEMM correctness (64x64)", err < 1e-3f);
    }

    // ── Test 7: Tiled GEMM correctness (tile-aligned) ────────────────────────
    {
        const int m = 64, n = 64, k = 64;
        std::vector<float> A(m*k, 0.f), B(k*n, 0.f), Cref(m*n, 0.f);
        for (int i = 0; i < m && i < k; ++i) A[i*k+i] = 1.f;
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < n; ++j) B[i*n+j] = (float)(i+j);
        cpu_gemm(m, n, k, A.data(), B.data(), Cref.data());

        float err = test_gemm_kernel(m, n, k, A.data(), B.data(), Cref.data(), true, stream);
        if (info.rank == 0) report("Tiled GEMM correctness (64x64)", err < 1e-3f);
    }

    // ── Test 8: Tiled GEMM non-tile-aligned dimensions ───────────────────────
    {
        const int m = 65, n = 33, k = 17;  // none divisible by 16
        std::mt19937 gen(555);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> A(m*k), B(k*n), Cref(m*n, 0.f);
        for (auto& v : A) v = dist(gen);
        for (auto& v : B) v = dist(gen);
        cpu_gemm(m, n, k, A.data(), B.data(), Cref.data());

        float err = test_gemm_kernel(m, n, k, A.data(), B.data(), Cref.data(), true, stream);
        if (info.rank == 0) report("Tiled GEMM non-aligned (65x33x17)", err < 1e-3f);
    }

    // ── Test 9: Naive GEMM non-tile-aligned dimensions ───────────────────────
    {
        const int m = 65, n = 33, k = 17;
        std::mt19937 gen(555);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> A(m*k), B(k*n), Cref(m*n, 0.f);
        for (auto& v : A) v = dist(gen);
        for (auto& v : B) v = dist(gen);
        cpu_gemm(m, n, k, A.data(), B.data(), Cref.data());

        float err = test_gemm_kernel(m, n, k, A.data(), B.data(), Cref.data(), false, stream);
        if (info.rank == 0) report("Naive GEMM non-aligned (65x33x17)", err < 1e-3f);
    }

    if (info.rank == 0)
        std::cout << "\nResults: " << g_pass << " passed, " << g_fail << " failed.\n";

    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Finalize();
    return g_fail > 0 ? 1 : 0;
}
