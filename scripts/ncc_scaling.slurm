#!/bin/bash
#SBATCH --job-name=mpi-cuda-scaling
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=scaling-%j.out

module purge
module load cuda openmpi cmake

mkdir -p build && cd build
cmake ..
cmake --build . -j

for NP in 1 2 4; do
  echo "=== NP=$NP ==="
  mpirun -np $NP ./mpi_cuda_coursework --mode gemm --M 4096 --N 4096 --K 4096 --kernel tiled --csv results.csv
done
