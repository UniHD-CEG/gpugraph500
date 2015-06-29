#!/bin/bash
#SBATCH -J test_reduce
#SBATCH --get-user-env
# #SBATCH --tasks=9
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 10:00


date
mpirun -np 9 -mca btl tcp,self ./../cpu_2d/g500 -s 21 -C 3 -gpus 1 -qs 2

