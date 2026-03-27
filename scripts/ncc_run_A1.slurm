#!/bin/bash
#SBATCH --job-name=mpi-cuda-run
#SBATCH --partition=ug-gpu-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm-task.out

module purge
module load cuda openmpi cmake

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
cd ..


mpirun -np 2 ./build/mpi_cuda_coursework --mode axpy --N 50000000 --csv results.csv
mpirun -np 2 ./build/mpi_cuda_coursework --mode add --N 50000000 --csv results.csv
mpirun -np 2 ./build/mpi_cuda_coursework --mode copy --N 50000000 --csv results.csv