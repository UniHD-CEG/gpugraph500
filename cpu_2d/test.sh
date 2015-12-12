#!/bin/bash
export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
../cpu_2d/g500 -s 26 -C 4 -gpus 1 -qs 2 -be "frameofreference" -btr 4096 -btc 4096
