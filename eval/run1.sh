#!/bin/bash

if [ ! -d result ]
then
	mkdir result
fi

if [ ! -d rtlog ]
then
	mkdir rtlog
fi

#Default values for testing
#MPI processes
n=1
#Scale
msc=20

if test -z "$1" || test -z "$2"
then
echo "Please pass in the number of nodes and the scale (20-26)"
  exit
fi

n=$1
msc=$2

qsub -v p=`expr $n \* $n`,srp=$n,scale=$msc -l nodes=`expr $n \* $n` keeneland.sh

