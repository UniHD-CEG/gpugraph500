#!/bin/bash
#SBATCH -J test_reduce
#SBATCH --get-user-env
# #SBATCH --tasks=4
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 10:00


date
mpirun -np 4 -mca btl tcp,self ./../cpu_2d/g500 -s 21 -C 2 -gpus 1 -qs 1

