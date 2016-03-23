#!/bin/bash
SF=$1
SRP=$2
GPUS=$3
BTR=$4
BTC=$5

export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
#echo "../cpu_2d/g500 -s $SF -C $SRP -gpus $GPUS -qs 1 -be s4-bp128-d4 -btr $BTR -btc $BTC"
../cpu_2d/g500 -s $SF -C $SRP -gpus $GPUS -qs 1 -be s4-bp128-d4 -btr $BTR -btc $BTC 
