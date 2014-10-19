#!/bin/bash

if [ ! -d result ]
then
	mkdir result
fi

if [ ! -d rtlog ]
then
	mkdir rtlog
fi

gpus=(1)
#scale factors
msize=(18 19 20 21 22 23 24 25)
#square root of nodes
snodes=(1  2  3)
#maximum scale factor
nmax=( 22 23 25)

for msc in ${msize[@]}
do 

for i in ${!snodes[@]}
do

for ngpus in ${gpus[@]}
do 
	n=${snodes[$i]}
	if [ $msc -le ${nmax[$i]} ]
	then
		sbatch  --export=p=`expr $n \* $n`,srp=$n,scale=$msc,gpus=$ngpus --nodes=`expr $n \* $n` --gres=gpu:$ngpus   ./sg500run.sh
	fi
done

done 

done