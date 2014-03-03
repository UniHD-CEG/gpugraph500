#PBS -S /bin/bash
#PBS -N g500
#PBS -j oe
#PBS -o test.$PBS_JOBID.log
#PBS -m a
#PBS -m e
#PBS -M matthias.hauck@stud.uni-heidelberg.de
#PBS -A UT-NTNL0208
#PBS -l walltime=00:20:00
#PBS -l nodes=16:ppn=1

cd $PBS_O_WORKDIR

#mpirun -n 16 ./g500 -s 20 -C 4
#mpirun -n 16 ./g500 -s 22 -C 4
#mpirun -n 16 ./cpu_2d/g500 -s 24 -C 4
mpirun --mca btl openib,self,sm -n 16 ./cpu_2d/g500 -s 26 -C 4
