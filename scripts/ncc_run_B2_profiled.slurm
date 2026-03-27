#!/bin/bash
#SBATCH --job-name=gemm-comparison
#SBATCH --partition=ug-gpu-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm-gemm-comparison-%j.out

module purge
module load cuda/12.4

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
cd ..

BIN=./build/mpi_cuda_coursework
CSV=results_gemm.csv


echo "=== GEMM naive  NP=1 M=N=K=4096 ==="
mpirun -np 1 $BIN --mode gemm --M 4096 --N 4096 --K 4096 --kernel naive --csv $CSV

echo "=== GEMM tiled  NP=1 M=N=K=4096 ==="
mpirun -np 1 $BIN --mode gemm --M 4096 --N 4096 --K 4096 --kernel tiled --csv $CSV

export TMPDIR="${HOME}/tmp-ncu-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

echo ""
echo "=== ncu: naive kernel (M=N=K=1024) ==="
mpirun -np 1 ncu --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --kernel-name k_gemm_naive \
    --launch-count 1 \
    $BIN --mode gemm --M 1024 --N 1024 --K 1024 --kernel naive

echo ""
echo "=== ncu: tiled kernel (M=N=K=1024) ==="
mpirun -np 1 ncu --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --kernel-name "regex:k_gemm_tiled" \
    --launch-count 1 \
    $BIN --mode gemm --M 1024 --N 1024 --K 1024 --kernel tiled

rm -rf "$TMPDIR"
