#!/bin/bash
#SBATCH -J test_reduce
#SBATCH --get-user-env
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

MAX_SF=21
CODEC=s4-bp128-d4
ROWTHRESHOLD=64
COLUMNTHRESHOLD=64

if [ "x$G500_SIMDCOMPRESSION_CODE" != "x" ]; then
  echo "Using CPUSIMD-CODEC $G500_SIMDCOMPRESSION_CODE"
  codec=$G500_SIMDCOMPRESSION_CODE
else
  echo "Using CPUSIMD-CODEC $CODEC"
  codec=$CODEC
fi

if [ "x$G500_ROW_COMPRESSION_THRESHOLD" != "x" ]; then
  echo "Using ROW COMPRESSION-THRESHOLD $G500_ROW_COMPRESSION_THRESHOLD"
  rowthreshold=$G500_ROW_COMPRESSION_THRESHOLD
else
  echo "Using ROW COMPRESSION-THRESHOLD $ROWTHRESHOLD"
  rowthreshold=$ROWTHRESHOLD
fi

if [ "x$G500_COLUMN_COMPRESSION_THRESHOLD" != "x" ]; then
  echo "Using COLUMN COMPRESSION-THRESHOLD $G500_COLUMN_COMPRESSION_THRESHOLD"
  columnthreshold=$G500_COLUMN_COMPRESSION_THRESHOLD
else
  echo "Using COLUMN COMPRESSION-THRESHOLD $COLUMNTHRESHOLD"
  columnthreshold=$COLUMNTHRESHOLD
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
  $scalasca $mpirun -np 9 --display-map ./../cpu_2d/g500 -s $scale_factor -C 3 -gpus 1 -qs 2.1
else
  $mpirun -np 9 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -x PATH=$PATH --display-map ../cpu_2d/g500 -s $scale_factor -C 3 -gpus 1 -qs 2.1 -be $codec -btr $rowthreshold -btc $columnthreshold
fi

