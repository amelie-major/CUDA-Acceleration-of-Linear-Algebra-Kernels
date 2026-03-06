#pragma once
#include <cuda_runtime.h>
#include <mpi.h>

struct CpuTimer {
    double t0 = 0.0;
    void start() { t0 = MPI_Wtime(); }
    double stop() const { return MPI_Wtime() - t0; }
};

struct GpuTimer {
    cudaEvent_t start_evt{}, stop_evt{};
    GpuTimer() { cudaEventCreate(&start_evt); cudaEventCreate(&stop_evt); }
    ~GpuTimer() { cudaEventDestroy(start_evt); cudaEventDestroy(stop_evt); }
    void start(cudaStream_t stream = 0) { cudaEventRecord(start_evt, stream); }
    float stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_evt, stream);
        cudaEventSynchronize(stop_evt);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_evt, stop_evt);
        return ms;
    }
};
