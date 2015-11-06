#!/bin/bash
#SBATCH -J test_reduce
#SBATCH --get-user-env
#SBATCH --tasks=16
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1

MAX_SF=22
# CODEC=s4-bp128-d4
CODEC=frameofreference
ROWTHRESHOLD=64
COLUMNTHRESHOLD=64

if [ "x$G500_SIMDCOMPRESSION_CODEC" != "x" ]; then
  echo "Using CPUSIMD-CODEC $G500_SIMDCOMPRESSION_CODEC"
  codec=$G500_SIMDCOMPRESSION_CODEC
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


date
if [ "x$G500_ENABLE_RUNTIME_SCALASCA" = "xyes" ]; then
  scalasca="scalasca -analyze -e scorep_g500_testreduce`date +"%F-%s"`"
  $scalasca mpirun --display-map -np 16 ../cpu_2d/g500 -s $scale_factor -C 4 -gpus 1 -qs 2 -be $codec -btr $rowthreshold -btc $columnthreshold
else
  mpirun --display-map -np 16 ../cpu_2d/g500 -s $scale_factor -C 4 -gpus 1 -qs 2 -be $codec -btr $rowthreshold -btc $columnthreshold
fi


