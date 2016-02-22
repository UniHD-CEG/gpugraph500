#!/bin/bash

MAX_SF=26

if [ x"$1" != x ]; then
  MAX_SF=$1
fi
echo "Testing Scale Factors from 8 to ${MAX_SF}.."

for i in `seq 1 ${MAX_SF}`; do

  echo "mpirun -np 36 ../cpu_2d/g500 -s ${i} -C 6 -gpus 1 -qs 2 -be 's4-bp128-d4' -btr 128 -btc 128  > falcon_nd36_scale${i}_btr128_btc128 2>&1"

done

for i in `seq 1 ${MAX_SF}`; do

  echo "mpirun -np 49 ../cpu_2d/g500 -s ${i} -C 7 -gpus 1 -qs 2 -be 's4-bp128-d4' -btr 128 -btc 128  > falcon_nd49_scale${i}_btr128_btc128 2>&1"

done

for i in `seq 1 ${MAX_SF}`; do

  echo "mpirun -np 64 ../cpu_2d/g500 -s ${i} -C 8 -gpus 1 -qs 2 -be 's4-bp128-d4' -btr 128 -btc 128  > falcon_nd64_scale${i}_btr128_btc128 2>&1"

done


