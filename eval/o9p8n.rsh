#!/bin/bash
#SBATCH -J test_reduce
#SBATCH --get-user-env
# #SBATCH --tasks=9
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
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
mpirun=mpirun
mpirun=/home/jromera/openmpi/bin/mpirun
valgrind=valgrind

if [ "x$SCALASCA_ENABLE_RUNTIME" = "xyes" ]; then
  scalasca="scan -s"
fi

date
$scalasca $mpirun -np 9 --display-map ./../cpu_2d/g500 -s $scale_factor -C 3 -gpus 1 -qs 2.1

