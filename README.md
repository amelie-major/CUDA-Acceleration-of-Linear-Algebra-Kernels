# COMP3741 MPI + CUDA Coursework

CUDA acceleration of linear algebra kernels with MPI distributed-memory parallelism.
Tested on the Durham University NCC cluster (Turing GPUs).

## Project structure

```
src/
  cuda_kernels.cu       – CUDA kernels (AXPY, ADD, COPY, reduction, naive/tiled GEMM)
  cpu_reference.cpp     – CPU reference implementations for validation
  mpi_distribution.cpp  – MPI row-distribution helpers (Scatterv counts/displs)
  benchmarks.cpp        – CSV result logging
  main.cpp              – CLI driver for all modes
include/                – Header files (kernels, timers, error checking, CLI, MPI utils)
tests/
  test_main.cpp         – Self-contained correctness test suite (9 tests)
scripts/                – SLURM batch scripts for NCC
```

## Build

```bash
module load cuda openmpi cmake
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

## Run examples

```bash
# Part A1: vector operations
mpirun -np 2 ./mpi_cuda_coursework --mode axpy --N 50000000 --alpha 2.0
mpirun -np 2 ./mpi_cuda_coursework --mode add  --N 50000000
mpirun -np 2 ./mpi_cuda_coursework --mode copy --N 50000000

# Part A2: parallel reduction
mpirun -np 2 ./mpi_cuda_coursework --mode reduce --N 50000000 --seed 3000

# Part B: GEMM (naive and tiled)
mpirun -np 1 ./mpi_cuda_coursework --mode gemm --M 2048 --N 2048 --K 2048 --kernel naive
mpirun -np 1 ./mpi_cuda_coursework --mode gemm --M 2048 --N 2048 --K 2048 --kernel tiled

# Part C: strong scaling (1, 2, 4 ranks)
for NP in 1 2 4; do
  mpirun -np $NP ./mpi_cuda_coursework --mode gemm --M 4096 --N 4096 --K 4096 --kernel tiled
done
```

## Tests

```bash
mpirun -np 2 ./build/mpi_cuda_tests
```

The test suite covers:
1. AXPY correctness
2. ADD correctness
3. COPY correctness (bitwise exact)
4. Reduction with constant input
5. Reduction with random data across 4 seeds (GPU vs CPU)
6. Naive GEMM correctness (64x64)
7. Tiled GEMM correctness (64x64, tile-aligned)
8. Tiled GEMM with non-tile-aligned dimensions (65x33x17)
9. Naive GEMM with non-tile-aligned dimensions (65x33x17)

## NCC batch scripts

| Script | Purpose |
|--------|---------|
| `scripts/ncc_run_A1.slurm` | Vector operations benchmarks |
| `scripts/ncc_run_A2.slurm` | Reduction with multiple seeds |
| `scripts/ncc_run_B1.slurm` | Naive GEMM benchmarks |
| `scripts/ncc_run_B2_profiled.slurm` | Tiled vs naive GEMM comparison + `ncu` profiling |
| `scripts/ncc_run_C1.slurm` | Strong scaling experiment (1, 2, 4 MPI ranks) |
| `scripts/ncc_run_tests.slurm` | Run correctness test suite |

## System details

- **GPU**: NVIDIA Turing (NCC `ug-gpu-small` partition)
- **MPI**: OpenMPI
- **CUDA architectures**: sm_70, sm_80
- **Compiler**: GCC + nvcc (C++17)
- **Timing**: CUDA events for GPU kernels, `MPI_Wtime` for communication
