#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <cmath>

#include "cli.hpp"
#include "mpi_utils.hpp"
#include "mpi_distribution.hpp"
#include "check_cuda.hpp"
#include "timer.hpp"
#include "kernels.hpp"
#include "cpu_reference.hpp"
#include "benchmarks.hpp"

static void select_gpu_for_rank(int rank) {
    int ndev=0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev==0) MPI_Abort(MPI_COMM_WORLD, 1);
    CUDA_CHECK(cudaSetDevice(rank % ndev));
}

static void fill_random(std::vector<float>& v, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(gen);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    auto info = mpi_info();
    Args args = Args::parse(argc, argv);

    std::string mode = args.get("mode", "");
    if (mode.empty()) {
        if (info.rank==0) std::cout << "Use --mode axpy|vadd|vcopy|reduce|gemm\n";
        MPI_Finalize();
        return 0;
    }

    select_gpu_for_rank(info.rank);
    // A single CUDA stream serialises GPU work on this rank.
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));

    std::string csv = args.get("csv", "results.csv");

    if (mode == "axpy") {
        long long N = args.get_ll("N", 10'000'000);
        float alpha = (float)args.get_double("alpha", 2.0);
        auto d = dist_1d(N, info.rank, info.size);

        std::vector<float> x(d.N_local), y(d.N_local), yref(d.N_local);
        fill_random(x, 1000 + info.rank);
        fill_random(y, 2000 + info.rank);
        yref = y;

        float *dx=nullptr, *dy=nullptr;
        CUDA_CHECK(cudaMalloc(&dx, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(dx, x.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dy, y.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        GpuTimer gt; gt.start(stream);
        launch_axpy((int)d.N_local, alpha, dx, dy, stream);
        float ms = gt.stop(stream);

        CUDA_CHECK(cudaMemcpyAsync(y.data(), dy, d.N_local*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cpu_axpy((int)d.N_local, alpha, x.data(), yref.data());
        float err = max_abs_diff((int)d.N_local, y.data(), yref.data());

        double bytes = 12.0 * (double)d.N_local;
        double gbs = (bytes/1e9) / (ms/1e3);

        if (info.rank==0) std::cout << "[AXPY] ms="<<ms<<" GB/s="<<gbs<<" err="<<err<<"\n";
        append_csv(csv, info.rank, "axpy", "axpy", N, 0, 0, 0, ms, 0.0, gbs);

        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy));
    }
    else if (mode == "vadd") {
        long long N = args.get_ll("N", 10'000'000);
        auto d = dist_1d(N, info.rank, info.size);

        std::vector<float> x(d.N_local), y(d.N_local), z(d.N_local), zref(d.N_local);
        fill_random(x, 1000 + info.rank);
        fill_random(y, 2000 + info.rank);

        float *dx=nullptr, *dy=nullptr, *dz=nullptr;
        CUDA_CHECK(cudaMalloc(&dx, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dz, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(dx, x.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dy, y.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        GpuTimer gt; gt.start(stream);
        launch_vadd((int)d.N_local, dx, dy, dz, stream);
        float ms = gt.stop(stream);

        CUDA_CHECK(cudaMemcpyAsync(z.data(), dz, d.N_local*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // CPU reference computed inline (single addition, no separate function needed)
        for (int i = 0; i < (int)d.N_local; ++i) zref[i] = x[i] + y[i];
        float err = max_abs_diff((int)d.N_local, z.data(), zref.data());

        double bytes = 12.0 * (double)d.N_local;  // 2 reads + 1 write × 4 bytes
        double gbs   = (bytes/1e9) / (ms/1e3);

        if (info.rank==0) std::cout << "[VADD] ms="<<ms<<" GB/s="<<gbs<<" err="<<err<<"\n";
        append_csv(csv, info.rank, "vadd", "vadd", N, 0, 0, 0, ms, 0.0, gbs);

        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dz));
    }
    else if (mode == "vcopy") {
        long long N = args.get_ll("N", 10'000'000);
        auto d = dist_1d(N, info.rank, info.size);

        std::vector<float> x(d.N_local), z(d.N_local);
        fill_random(x, 1000 + info.rank);

        float *dx=nullptr, *dz=nullptr;
        CUDA_CHECK(cudaMalloc(&dx, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dz, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(dx, x.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        GpuTimer gt; gt.start(stream);
        launch_vcopy((int)d.N_local, dx, dz, stream);
        float ms = gt.stop(stream);

        CUDA_CHECK(cudaMemcpyAsync(z.data(), dz, d.N_local*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // z should be a bitwise copy of x — max_abs_diff must be 0
        float err = max_abs_diff((int)d.N_local, z.data(), x.data());

        double bytes = 8.0 * (double)d.N_local;  // 1 read + 1 write × 4 bytes
        double gbs   = (bytes/1e9) / (ms/1e3);

        if (info.rank==0) std::cout << "[VCOPY] ms="<<ms<<" GB/s="<<gbs<<" err="<<err<<"\n";
        append_csv(csv, info.rank, "vcopy", "vcopy", N, 0, 0, 0, ms, 0.0, gbs);

        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dz));
    }
    else if (mode == "reduce") {
        long long N = args.get_ll("N", 50'000'000);
        auto d = dist_1d(N, info.rank, info.size);

        std::vector<float> x(d.N_local);
        fill_random(x, 3000 + info.rank);

        float* dx=nullptr;
        CUDA_CHECK(cudaMalloc(&dx, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(dx, x.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        GpuTimer gt; gt.start(stream);
        float local = gpu_reduce_sum(dx, (int)d.N_local, stream);
        float ms = gt.stop(stream);

        float global = 0.0f;
        MPI_Allreduce(&local, &global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        float local_ref = cpu_reduce_sum((int)d.N_local, x.data());
        float global_ref = 0.0f;
        MPI_Allreduce(&local_ref, &global_ref, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        float err = std::fabs(global - global_ref);

        if (info.rank==0) std::cout << "[REDUCE] ms="<<ms<<" err="<<err<<"\n";
        append_csv(csv, info.rank, "reduce", "reduce", N, 0, 0, 0, ms, 0.0, 0.0);

        CUDA_CHECK(cudaFree(dx));
    }
    else if (mode == "gemm") {
        int M = args.get_int("M", 2048);
        int Nmat = args.get_int("N", 2048);
        int K = args.get_int("K", 2048);
        std::string kernel = args.get("kernel", "tiled"); // naive|tiled

        auto d = dist_rows(M, Nmat, K, info.rank, info.size);

        std::vector<float> A_full, B;
        if (info.rank==0) {
            A_full.resize((size_t)M*K);
            B.resize((size_t)K*Nmat);
            fill_random(A_full, 111);
            fill_random(B, 222);
        } else {
            B.resize((size_t)K*Nmat);
        }

        std::vector<int> countsA, displsA;
        build_counts_displs_rows(M, K, info.size, countsA, displsA);

        std::vector<float> A_local((size_t)d.M_local*K);
        MPI_Scatterv(info.rank==0 ? A_full.data() : nullptr, countsA.data(), displsA.data(), MPI_FLOAT,
                     A_local.data(), (int)A_local.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

        MPI_Bcast(B.data(), (int)B.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

        std::vector<float> C_local((size_t)d.M_local*Nmat, 0.0f);
        std::vector<float> C_ref((size_t)d.M_local*Nmat, 0.0f);

        float *dA=nullptr,*dB=nullptr,*dC=nullptr;
        CUDA_CHECK(cudaMalloc(&dA, A_local.size()*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, B.size()*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, C_local.size()*sizeof(float)));

        CUDA_CHECK(cudaMemcpyAsync(dA, A_local.data(), A_local.size()*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dB, B.data(), B.size()*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(dC, 0, C_local.size()*sizeof(float), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        GpuTimer gt; gt.start(stream);
        if (kernel == "naive") launch_gemm_naive(d.M_local, Nmat, K, dA, dB, dC, stream);
        else launch_gemm_tiled16(d.M_local, Nmat, K, dA, dB, dC, stream);
        float ms = gt.stop(stream);

        CUDA_CHECK(cudaMemcpyAsync(C_local.data(), dC, C_local.size()*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float err = 0.0f;
        if ((long long)M * (long long)Nmat <= 1024LL*1024LL) {
            cpu_gemm(d.M_local, Nmat, K, A_local.data(), B.data(), C_ref.data());
            err = max_abs_diff((int)C_local.size(), C_local.data(), C_ref.data());
        }

        double flops = 2.0 * (double)d.M_local * (double)Nmat * (double)K;
        double gflops = (flops/1e9) / (ms/1e3);

        if (info.rank==0) std::cout << "[GEMM] kernel="<<kernel<<" ms="<<ms<<" GFLOP/s(local)="<<gflops
                                    << (err>0 ? " err="+std::to_string(err) : "") << "\n";
        append_csv(csv, info.rank, "gemm", kernel, 0, M, Nmat, K, ms, gflops, 0.0);

        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Finalize();
    return 0;
}
