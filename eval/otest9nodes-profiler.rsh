#!/bin/bash
#SBATCH -J test_reduce
#SBATCH --get-user-env
#SBATCH --tasks=9
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH -t 10:00


date
mpirun tau_exec ./../cpu_2d/g500 -s 21 -C 3 -gpus 1 -qs 2

