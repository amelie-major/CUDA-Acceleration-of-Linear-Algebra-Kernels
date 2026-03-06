#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>

#include "mpi_utils.hpp"
#include "check_cuda.hpp"
#include "kernels.hpp"

static void select_gpu_for_rank(int rank) {
    int ndev=0; CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev==0) MPI_Abort(MPI_COMM_WORLD, 1);
    CUDA_CHECK(cudaSetDevice(rank % ndev));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    auto info = mpi_info();
    select_gpu_for_rank(info.rank);
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));

    const long long N = 100000;
    auto d = dist_1d(N, info.rank, info.size);

    std::vector<float> x(d.N_local, 1.0f);
    float* dx=nullptr; CUDA_CHECK(cudaMalloc(&dx, d.N_local*sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(dx, x.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float local = gpu_reduce_sum(dx, (int)d.N_local, stream);
    float global = 0.0f;
    MPI_Allreduce(&local, &global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    if (info.rank==0) std::cout << "Reduction test: got="<<global<<" expected="<<(float)N
                               << " diff="<<std::fabs(global-(float)N) << "\n";

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Finalize();
    return 0;
}
