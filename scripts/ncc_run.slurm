#!/bin/bash
#SBATCH --job-name=mpi-cuda-run
#SBATCH --partition=ug-gpu-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.out

module purge
module load cuda openmpi cmake

mkdir -p build && cd build
cmake ..
cmake --build . -j

CSV=results.csv

# Strong scaling: fixed global problem size M=N=K=2048
for NP in 1 2 4; do
    echo "=== GEMM naive  NP=$NP ==="
    mpirun -np $NP ./mpi_cuda_coursework --mode gemm --M 2048 --N 2048 --K 2048 --kernel naive  --csv $CSV

    echo "=== GEMM tiled  NP=$NP ==="
    mpirun -np $NP ./mpi_cuda_coursework --mode gemm --M 2048 --N 2048 --K 2048 --kernel tiled  --csv $CSV
done