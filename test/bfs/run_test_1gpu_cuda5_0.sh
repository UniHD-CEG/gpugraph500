#!/bin/sh
#This script is meant as a simple example of running BFS
#with the simple grid input sets
OPTIONS="--i=2 --src=randomize --device=2 --quick"
SUFFIX="mpi_test"

mkdir -p eval/$SUFFIX

echo mpirun -np 2 ./bin/test_bfs_5.0_x86_64 grid2d 5000 --queue-sizing=0.15 $OPTIONS 
     mpirun -np 2 ./bin/test_bfs_5.0_x86_64 grid2d 5000 --queue-sizing=0.15 $OPTIONS > eval/$SUFFIX/grid2d.5000.$SUFFIX.txt	
sleep 10 

#echo ./bin/test_bfs_5.0_x86_64 grid2d 5000 --queue-sizing=0.15 $OPTIONS 
#     ./bin/test_bfs_5.0_x86_64 grid2d 5000 --queue-sizing=0.15 $OPTIONS > eval/$SUFFIX/grid2d.5000.$SUFFIX.txt	
#sleep 10 

#echo ./bin/test_bfs_5.0_x86_64 grid3d 300 --queue-sizing=0.15 $OPTIONS 
#     ./bin/test_bfs_5.0_x86_64 grid3d 300 --queue-sizing=0.15 $OPTIONS > eval/$SUFFIX/grid3d.300.$SUFFIX.txt	
#sleep 10 
