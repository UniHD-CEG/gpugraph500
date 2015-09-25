#!/bin/sh

echo "This script will run your BFS application using 4 SLURM processes and 4 MPI Nodes."
echo "It uses gdb to debug the code. Edit the Makefile, set code_debug=yes and run make."
echo ""

mpirun=`which mpirun`
mpirun=/home/jromera/openmpi/bin/mpirun

iseval=`pwd | grep "/eval"`

if [ ! -d $iseval ];then
  echo "error: run this script from eval directory."
  exit 1
fi

srun --tasks=4 --ntasks-per-node=2 $mpirun -np 4 --display-map gdb -ex run -ex "set confirm off"  --args ../cpu_2d/g500 -s 14 -C 2 -gpus 1 -qs 1

