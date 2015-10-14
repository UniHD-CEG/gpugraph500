#!/bin/bash
#SBATCH -J test_reduce
#SBATCH --get-user-env
#SBATCH --tasks=16
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH -t 10:00

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
valgrind=

date
if [ "x$G500_ENABLE_RUNTIME_SCALASCA" = "xyes" ]; then
  scalasca="scalasca -analyze -f filter.scorep -e scorep_g500_testreduce`date +"%F-%s"`"
  $scalasca $mpirun --display-map -np 16 $valgrind ../cpu_2d/g500 -s $scale_factor -C 4 -gpus 1 -qs 2
else
  $mpirun --display-map -np 16 $valgrind ../cpu_2d/g500 -s $scale_factor -C 4 -gpus 1 -qs 2 -be $codec -bt $threshold
fi


