#!/bin/bash
#SBATCH --job-name=mpi-cuda-run
#SBATCH --partition=ug-gpu-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out

module purge
module load cuda openmpi cmake

mkdir -p build && cd build
cmake ..
cmake --build . -j

mpirun -np 2 ./mpi_cuda_coursework --mode gemm --M 4096 --N 4096 --K 4096 --kernel tiled --csv results.csv