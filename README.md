# COMP3741 MPI + CUDA Coursework — Starter Repo (v2)

This repo is designed to be **handy**: it takes care of MPI distribution, timing,
error-checking, and correctness scaffolding so you can focus on CUDA kernels and analysis.

## Build
```bash
module load cuda openmpi cmake
mkdir -p build && cd build
cmake ..
cmake --build . -j
```

## Run examples
```bash
./mpi_cuda_coursework --mode axpy --N 10000000 --alpha 2.0
./mpi_cuda_coursework --mode reduce --N 10000000
./mpi_cuda_coursework --mode gemm --M 2048 --N 2048 --K 2048 --kernel tiled --tile 16
```

## Tests
```bash
mpirun -np 2 ./mpi_cuda_tests
```

## NCC batch jobs
- scripts/ncc_run.slurm
- scripts/ncc_scaling.slurm
