#!/bin/bash

numNodes="1 4 9 16 25 36 49 64"
gpusPerNode=3
prefixOfKidsNode=KIDS0000



for nn in `echo $numNodes`; do
 NODES=$(expr $nn / $gpusPerNode + 1 );

 file=hosts${NODES}.KIDS
 rm -rf $file 2> /dev/null
 echo "Creating file: $file with $NODES nodes and ${gpusPerNode} gpus/node"
 for n in `seq 0 $NODES`; do
   echo "$prefixOfKidsNode${n} slots=$gpusPerNode" 
   # uncomment this line if its correct:
   # echo "$prefixOfKidsNode${n} slots=$gpusPerNode" | tee $file
 done
done
 
