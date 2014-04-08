#!/bin/bash

logs=`find result -name '*.log' | sort`

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
mean_validation=$(sed -n 's/mean_validation_time: //p' <$log)
mean_row_com=$(sed -n 's/mean_row_com_time: //p' <$log)
mean_col_com=$(sed -n 's/mean_column_com_time: //p' <$log)
mean_predlistred=$(sed -n 's/mean_predecessor_list_reduction_time: //p' <$log)

#echo $log $scale $mpi_proc

if [ -n "$scale" ];
then

	if [ ! -f "mprocesses_`printf "%04d" ${mpi_proc}`.dat" ];
	then
		echo \# G500 runs with ${mpi_proc} mpi processes `date` > "mprocesses_`printf "%04d" ${mpi_proc}`.dat"
		echo \# \"scale\" \"mean TEPS\" \"mean time\" \"mean expansion\" \"mean queue handling\" \"mean rest\" \"mean validation\" \"mean row com.\"  \"mean column com.\" \"pred. list red.\" >> "mprocesses_`printf "%04d" ${mpi_proc}`.dat"
	fi

	if [ ! -f "scale_`printf "%02d" ${scale}`.dat" ];
	then
		echo \# G500 runs with scalefactor ${scale} `date` > "scale_`printf "%02d" ${scale}`.dat"
		echo \# \"mpi processes\" \"mean TEPS\" \"mean time\" \"mean expansion\" \"mean queue handling\" \"mean rest\" \"mean validation\" \"mean row com.\"  \"mean column com.\" \"pred. list red.\" >> "scale_`printf "%02d" ${scale}`.dat"
	fi

	echo ${scale} ${mean_teps} ${mean_time} ${mean_exp} ${mean_queue} ${mean_rest} ${mean_validation} ${mean_row_com} ${mean_col_com} ${mean_predlistred} >> "mprocesses_`printf "%04d" ${mpi_proc}`.dat"
	echo ${mpi_proc} ${mean_teps} ${mean_time} ${mean_exp} ${mean_queue} ${mean_rest} ${mean_validation} ${mean_row_com} ${mean_col_com} ${mean_predlistred} >> "scale_`printf "%02d" ${scale}`.dat"

else
	echo No valid entrie in file $log.
fi

done