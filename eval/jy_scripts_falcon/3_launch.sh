#!/bin/bash
SF=$1
SRP=$2
GPUS=$3
BTR=$4
BTC=$5

export CUDA_VISIBLE_DEVICES=0
NUMAEXEC="numactl --membind=0 --cpunodebind=0"

#echo "$NUMAEXEC ../cpu_2d/g500 -s $SF -C $SRP -gpus $GPUS -qs 1 -be s4-bp128-d4 -btr $BTR -btc $BTC"
$NUMAEXEC ../cpu_2d/g500 -s $SF -C $SRP -gpus $GPUS -qs 1 -be s4-bp128-d4 -btr $BTR -btc $BTC 
