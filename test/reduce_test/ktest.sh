#PBS -S /bin/bash
#PBS -N msRedTest
#PBS -j oe
#PBS -o test.$PBS_JOBID.log
#PBS -m a
#PBS -m e
#PBS -A UT-NTNL0208
#PBS -l walltime=00:12:00
#PBS -l nodes=8:ppn=1:gpus=1

cd $PBS_O_WORKDIR
#Initializes a default BASH environment
. /opt/modules/default/init/bash > /dev/null
#Loads experiment-specific path
. ~/env_vars/intel_cuda6.sh > /dev/null
module load valgrind/3.9.0

mpiexec -verbose ./reduce_test
