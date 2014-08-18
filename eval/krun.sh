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
msize=(20 21 22 23 24 25 26)
#square root of nodes
snodes=(1  2  3	 4  5  6  7)
#maximum scale factor
nmax=( 23 23 25 26 26 26 26)

for msc in ${msize[@]}
do 

for i in ${!snodes[@]}
do

for ngpus in ${gpus[@]}
do 
	n=${snodes[$i]}
	if [ $msc -le ${nmax[$i]} ]
	then
		qsub -v p=`expr $n \* $n`,srp=$n,scale=$msc,gpus=$ngpus -l nodes=`expr $n \* $n`:ppn=1:gpus=$ngpus keeneland.sh
	fi
done

done 

done