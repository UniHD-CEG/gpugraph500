#!/bin/bash

logs=`find . -name '*.log'`

rm *.dat

for log in $logs
do
scale=$(sed -n 's/SCALE: //p' < $log)
mpi_proc=$(sed -n 's/num_mpi_processes: //p' < $log)
mean_teps=$(sed -n 's/harmonic_mean_TEPS: //p' < $log)
mean_time=$(sed -n 's/mean_time: //p' < $log)
mean_exp=$(sed -n 's/mean_local_bfs_time: //p' < $log)
mean_queue=$(sed -n 's/mean_local_queue_time: //p' < $log)
mean_rest=$(sed -n 's/mean_rest_time: //p' < $log)

echo $log $scale $mpi_proc

if [ ! -f "mprocesses_${mpi_proc}.dat" ];
then
    echo \# G500 runs with ${mpi_proc} mpi processes `date` > "mprocesses_${mpi_proc}.dat"
    echo \# "scale" "mean TEPS" "mean time" "mean expansion" "mean queue handling" "mean rest" >> "mprocesses_${mpi_proc}.dat"
fi

if [ ! -f ${scale}.dat ];
then
    echo \# G500 runs with scalefactor ${scale} `date` > "scale_${scale}.dat"
    echo \# "mpi processes" "mean TEPS" "mean time" "mean expansion" "mean queue handling" "mean rest" >> "scale_${scale}.dat"
fi

echo ${scale} ${mean_teps} ${mean_time} ${mean_exp} ${mean_queue} ${mean_rest} >> "mprocesses_${mpi_proc}.dat"
echo ${mpi_proc} ${mean_teps} ${mean_time} ${mean_exp} ${mean_queue} ${mean_rest} >> "scale_${scale}.dat"

done