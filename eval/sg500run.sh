#!/bin/sh
#SBATCH -J g500
#SBATCH --ntasks-per-node=1
#SBATCH -o rtlog/run_%j.log
#SBATCH -t 30:00
#SBATCH -c 8
date
echo Scalefactor ${scale} Nodes ${p} Configuration ${srp} x ${srp} GPUs ${gpus}
echo Job ID: %j > result/p`printf "%04d" ${p}`_c`printf "%03d" ${srp}`_s`printf "%02d" ${scale}`_gpus${gpus}.log
mpiexec -np ${p} ./../cpu_2d/g500 -s ${scale} -C ${srp} -gpus ${gpus} >> result/p`printf "%04d" ${p}`_c`printf "%03d" ${srp}`_s`printf "%02d" ${scale}`_gpus${gpus}.log
