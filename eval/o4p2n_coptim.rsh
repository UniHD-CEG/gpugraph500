#!/bin/bash
#SBATCH -J test_reduce
#SBATCH --get-user-env
#SBATCH --tasks=4
#SBATCH --ntasks-per-node=2
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
export mpirun=mpirun
export mpirun=/home/jromera/openmpi/bin/mpirun

date
if [ "x$G500_ENABLE_RUNTIME_SCALASCA" = "xyes" ]; then
  scalasca="scalasca -analyze -f filter.scorep -e scorep_g500_testreduce`date +"%F-%s"`"
  $scalasca $mpirun -np 4 --display-map "-rf hosts-coptimum" ../cpu_2d/g500 -s $scale_factor -C 2 -gpus 1 -qs 1
else
  # $mpirun -np 4 --display-map -rf hosts-coptimum valgrind --leak-check=full --track-origins=yes --show-reachable=yes -v ../cpu_2d/g500 -s $scale_factor -C 2 -gpus 1 -qs 1
  $mpirun -np 4 --display-map -rf hosts-coptimum valgrind --leak-check=yes --gen-suppressions=no --suppressions=./valgrind.supp ../cpu_2d/g500 -s $scale_factor -C 2 -gpus 1 -qs 1
fi

