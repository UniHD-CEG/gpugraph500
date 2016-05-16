#!/bin/bash
N=$1
SF=$3
SRP=$2
GPUS=$4
BTR=$5
BTC=$6
LOGFILE=$7
SCOREPDIR=$8
HOSTS=~/scripts/hostfile_falcon_64


LD_LIBRARY_PATH=/opt/rh/devtoolset-2/root/usr/lib64:/opt/rh/devtoolset-2/root/usr/lib:/usr/lib64:/usr/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/lib:/usr/lib64/boost148/
PATH=/opt/rh/devtoolset-2/root/usr/bin:/net/rd6/jyoung9/mpi/ompi184/bin:/usr/bin:/usr/sbin:/bin:/usr/local/bin:/bin:/usr/local/sbin:/usr/local/cuda/bin:/usr/include/boost148/
MPIRUN=/net/rd6/jyoung9/mpi/ompi184/bin/mpirun

#Control for GPU and socket
#CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
#Bind the CPU and memory to the socket with the GPU being used
cudavisible="CUDA_VISIBLE_DEVICES=0"


source /net/rd6/jyoung9/.bashrc
echo "$MPIRUN -v -x $cudavisible -x PATH=$PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -x SCOREP_EXPERIMENT_DIRECTORY=$SCOREPDIR -hostfile $HOSTS -np $N -map-by node --display-map -mca btl openib,self -mca mpi_warn_on_fork 0 ./3_launch.sh $SF $SRP $GPUS $BTR $BTC &> ${LOGFILE}"
$MPIRUN -v -x $cudavisible -x PATH=$PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -x SCOREP_EXPERIMENT_DIRECTORY=$SCOREPDIR -hostfile $HOSTS -np $N -map-by node --display-map -mca btl openib,self -mca mpi_warn_on_fork 0 ./3_launch.sh $SF $SRP $GPUS $BTR $BTC &> ${LOGFILE}
#$MPIRUN -v -x PATH=$PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -hostfile $HOSTS -np $N -map-by slot --display-map -mca btl openib,self -mca mpi_warn_on_fork 0 ./3_launch.sh $SF $SRP $GPUS $BTR $BTC &> ${LOGFILE}
