#!/bin/bash

codec="s4-bp128-d4"
threshold=64

if [ x$1 = x ] || [ x$2 = x ]; then
  echo "error: $0 <SF> <C> [codec] [threshold]"
  exit 1
fi

if [ x$3 != x ]; then
  codec=${3}
fi

if [ x$4 != x ]; then
  threshold=${4}
fi

export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
../cpu_2d/g500 -s $1 -C $2 -gpus 1 -qs 2 -be $codec -btr $threshold -btc $threshold
