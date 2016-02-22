#!/bin/bash

codec="s4-bp128-d4"
threshold=64
g500=../cpu_2d/g500

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

for i in `seq 1 10`; do
        ls $g500 > /dev/null 2>&1
done

export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
if [ -x $g500 ]; then
	$g500 -s $1 -C $2 -gpus 1 -qs 1 -be $codec -btr $threshold -btc $threshold
fi
