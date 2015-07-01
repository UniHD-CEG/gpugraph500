#!/bin/bash
#SBATCH -J test_reduce
#SBATCH --get-user-env
#SBATCH --tasks=4
# #SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH -t 10:00


date
mpirun -np 4 --display-map -rf hosts-noptimum -mca btl tcp,self ./../cpu_2d/g500 -s 21 -C 2 -gpus 1 -qs 1

