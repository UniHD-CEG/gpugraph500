#!/bin/sh

echo "This script will run your BFS application using 4 SLURM processes and 4 MPI Noded."
echo "Since GDB will be used to debug, edit the Makefile and set code_debug=yes. Then run make"
echo ""

mpirun=`which mpirun`
mpirun=/home/jromera/openmpi/bin/mpirun

srun --tasks=4 --ntasks-per-node=2 $mpirun -np 4 --display-map gdb -ex run -ex "set confirm off"  --args ../cpu_2d/g500 -s 14 -C 2 -gpus 1 -qs 1

