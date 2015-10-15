#!/bin/bash
#SBATCH -J test_reduce
#SBATCH --get-user-env
#SBATCH --tasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1

MAX_SF=21
CODEC=s4-bp128-d4
THRESHOLD=512

if [ "x$G500_SIMDCOMPRESSION_CODE" != "x" ]; then
  echo "Using CPUSIMD-CODEC $G500_SIMDCOMPRESSION_CODE"
  codec=$G500_SIMDCOMPRESSION_CODE
else
  echo "Using CPUSIMD-CODEC $CODEC"
  codec=$CODEC
fi

if [ "x$G500_COMPRESSION_THRESHOLD" != "x" ]; then
  echo "Using COMPRESSION-THRESHOLD $G500_COMPRESSION_THRESHOLD"
  threshold=$G500_COMPRESSION_THRESHOLD
else
  echo "Using COMPRESSION-THRESHOLD $THRESHOLD"
  threshold=$THRESHOLD
fi


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

date
if [ "x$G500_ENABLE_RUNTIME_SCALASCA" = "xyes" ]; then
  scalasca="scalasca -analyze -f filter.scorep -e scorep_g500_testreduce`date +"%F-%s"`"
  $scalasca $mpirun -np 4 --display-map "-rf hosts-noptimum" ../cpu_2d/g500 -s $scale_factor -C 2 -gpus 1 -qs 1
else
  $mpirun -np 4 --display-map -rf hosts-noptimum ../cpu_2d/g500 -s $scale_factor -C 2 -gpus 1 -qs 1 -be $codec -bt $threshold
fi

