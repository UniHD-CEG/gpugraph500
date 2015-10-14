#!/bin/bash

echo "This script will run your BFS application using 4 SLURM processes and 4 MPI Nodes."
echo "It uses gdb to debug the code. Edit the Makefile, set code_debug=yes and run make."
echo ""
echo "ps. set 'debug_code=yes' in Makefile"

if [ "$#" != "2" ]; then
	echo "usage: $0 <N> <SF>"
	echo "   - where <N> is the numbre of MPI Nodes/ GPUs"
	echo "   - where <SF> is Scale Factor for the Graph500"
	exit 1
fi

number='^[0-9]+$'
if ! [[ $1 =~ $number ]] || [ $1 -gt 4 ] || [ $1 -lt 2 ]; then
	echo "error:: the maximun supported Nodes is 4^2"
	exit 1
else
	N=$1
fi

if ! [[ $2 =~ $number ]]; then
	echo "error:: The Scale factor must be a number."
	exit 1
else
	SF=$2
fi

PROCESSES=$(expr $N \* $N)

mpirun=`which mpirun`
mpirun=/home/jromera/openmpi/bin/mpirun

iseval=`pwd | grep "/eval"`

if [ ! -d $iseval ];then
  echo "error: run this script from eval directory."
  exit 1
fi

# 2^2 (4 proceses, 4 GPUs/ Nodes)
# srun --tasks=PROCESSES --ntasks-per-node=2 $mpirun -np 4 --display-map gdb -ex run -ex "set confirm off" --args ../cpu_2d/g500 -s $SF -C $N -gpus 1 -qs 2

# 3^2 (9 proceses, 9 GPUs/ Nodes)
# srun --tasks=9 --ntasks-per-node=2 $mpirun -np 9 --display-map gdb -ex run -ex "set confirm off" --args ../cpu_2d/g500 -s $SF -C $N -gpus 1 -qs 2.1

# 4^2 (16 proceses, 16 GPUs/ Nodes)
# srun --tasks=16 --ntasks-per-node=2 $mpirun -np 16 --display-map gdb -ex run -ex "set confirm off" --args ../cpu_2d/g500 -s $SF -C $N -gpus 1 -qs 2

# N^2 (N^2 processes, N^2 GPUs/ Nodes)
srun --tasks=$PROCESSES --ntasks-per-node=2 $mpirun -np $PROCESSES --display-map gdb -ex run -ex "set confirm off" --args ../cpu_2d/g500 -s $SF -C $N -gpus 1 -qs 2
