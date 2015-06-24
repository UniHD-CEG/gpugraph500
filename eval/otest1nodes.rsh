#!/bin/bash
#SBATCH -J test_reduce
#SBATCH --get-user-env
#SBATCH --tasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 30:00

date
mpirun ./../cpu_2d/g500 -s 21 -C 1 -gpus 1 -qs 1

