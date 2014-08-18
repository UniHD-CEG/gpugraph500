#PBS -S /bin/bash
#PBS -N g500
#PBS -j oe
#PBS -o rtlog/run_$PBS_JOBID.log
#PBS -m a
#PBS -m e
#PBS -A UT-NTNL0208
#PBS -l walltime=00:30:00

cd $PBS_O_WORKDIR

#Initializes a default BASH environment
. /opt/modules/default/init/bash
#Loads experiment-specific paths
. ~/env_vars/intel_cuda6.sh

#mpirun -np ${p} ./../cpu_2d/g500 -s ${scale} -C ${srp} -gpus ${gpus} > result/p`printf "%04d" ${p}`_c`printf "%03d" ${srp}`_s`printf "%02d" ${scale}`_gpus${gpus}.log#Just expose 2 GPUs per node, so that peer-to-peer can be used
cudavisible="CUDA_VISIBLE_DEVICES=1,2"

mpirun -x $cudavisible -np ${p} ./../cpu_2d/g500 -s ${scale} -C ${srp} -gpus ${gpus} > result/p`printf "%04d" ${p}`_c`printf "%03d" ${srp}`_s`printf "%02d" ${scale}`_gpus${gpus}.log
