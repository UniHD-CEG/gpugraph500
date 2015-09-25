#!/bin/bash
#SBATCH -J test_reduce
#SBATCH --get-user-env
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 10:00
source ~/.bashrc
date
export SCOREP_CUDA_ENABLE=yes
export SCOREP_CUDA_BUFFER=500000
export SCOREP_ENABLE_PROFILING=true
export SCOREP_ENABLE_TRACING=true
mpirun ./../cpu_2d/g500 -s 23 -C 2 -gpus 1 -qs 1

