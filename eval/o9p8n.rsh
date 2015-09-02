#!/bin/bash
#SBATCH -J test_reduce
#SBATCH --get-user-env
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 10:00

MAX_SF=21

if [ "x$G500_SCALE_FACTOR" != "x" ]; then
  echo "Using SCALE-FACTOR $G500_SCALE_FACTOR"
  scale_factor=$G500_SCALE_FACTOR
elif [ "x$1" != "x" ]; then
  echo "Using SCALE-FACTOR $1"
  scale_factor=${1}
else
  echo "Using SCALE-FACTOR $MAX_SF"
  scale_factor=$MAX_SF
fi
mpirun=mpirun
mpirun=/home/jromera/openmpi/bin/mpirun
valgrind=

data
if [ "x$G500_ENABLE_RUNTIME_SCALASCA" = "xyes" ]; then
  scalasca="scalasca -analyze -f filter.scorep -e scorep_g500_testreduce`date +"%F-%s"`"
  $scalasca $mpirun -np 9 --display-map ./../cpu_2d/g500 -s $scale_factor -C 3 -gpus 1 -qs 2.1
else
  $mpirun -np 9 --display-map ../cpu_2d/g500 -s $scale_factor -C 3 -gpus 1 -qs 2.1
fi

