#!/bin/bash
#SBATCH -J test_reduce
#SBATCH --get-user-env
#SBATCH --tasks=16
# #SBATCH -N 8
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH -t 10:00

MAX_SF=21
if [ "x$1" != "x" ]; then
  echo "Using SCALE-FACTOR $1"
  scale_factor=${1}
else
  echo "Using SCALE-FACTOR $MAX_SF"
  scale_factor=$MAX_SF
fi

date
mpirun -np 16 -mca btl tcp,self ./../cpu_2d/g500 -s $scale_factor -C 4 -gpus 1 -qs 2

