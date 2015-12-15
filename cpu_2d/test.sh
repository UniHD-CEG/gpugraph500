#!/bin/bash

if [ x$1 = x ]; then
  echo "error: $0 <SF>"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
../cpu_2d/g500 -s $1 -C 2 -gpus 1 -qs 2 -be "frameofreference" -btr 64 -btc 64
