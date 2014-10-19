#!/bin/bash
#SBATCH -J test_reduce
#SBATCH --get-user-env
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 10:00
source ~/.bashrc
date
mpirun ./../cpu_2d/g500 -s 23 -C 2 -gpus 1 -qs 1

