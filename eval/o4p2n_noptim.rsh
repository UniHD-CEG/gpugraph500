#!/bin/bash
#SBATCH -J test_reduce
#SBATCH --get-user-env
#SBATCH --tasks=4
# #SBATCH -N 2
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
mpirun=mpirun
mpirun=/home/jromera/openmpi/bin/mpirun

if [ "x$SCALASCA_ENABLE_RUNTIME" = "xyes" ]; then
  scalasca="scalasca -analyze -f filter.scorep -e scorep_g500_testreduce`date +"%F-%s"`"
fi

date
$scalasca $mpirun -np 4 --display-map "-rf hosts-noptimum" ./../cpu_2d/g500 -s $scale_factor -C 2 -gpus 1 -qs 1

