# COMP3741 MPI + CUDA Coursework — Cheat Sheet

## Core pattern
MPI distribute → H2D copy → kernel → D2H copy → MPI reduce/gather.

## NCC build & run
```bash
module load cuda openmpi cmake
mkdir -p build && cd build
cmake ..
cmake --build . -j
mpirun -np 2 ./mpi_cuda_tests
sbatch ../scripts/ncc_run.slurm
```

## One rank per GPU
```cpp
int ndev; cudaGetDeviceCount(&ndev);
cudaSetDevice(rank % ndev);
```

## Timing
- GPU: cuda events (sync stop event)
- CPU/MPI: MPI_Wtime()
- Always synchronize before stopping timers.

## AXPY bandwidth
Per float element: read x (4B) + read y (4B) + write y (4B) = 12B.
GB/s = (12*N / time_s)/1e9.

## GEMM GFLOP/s
FLOPs = 2*M*N*K.
GFLOP/s = (2*M*N*K / time_s)/1e9.

## Debug / profile
```bash
compute-sanitizer --tool memcheck ./mpi_cuda_coursework --mode axpy --N 1000000
compute-sanitizer --tool racecheck ./mpi_cuda_coursework --mode reduce --N 1000000
ncu ./mpi_cuda_coursework --mode gemm --M 2048 --N 2048 --K 2048 --kernel tiled
```

## MPI collectives used
```cpp
MPI_Allreduce(&local, &global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
MPI_Bcast(B, K*N, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Scatterv(A_full, counts, displs, MPI_FLOAT, A_local, local_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
```
