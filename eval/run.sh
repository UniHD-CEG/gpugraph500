#!/bin/bash

if [ ! -d result ]
then
	mkdir result
fi

if [ ! -d rtlog ]
then
	mkdir rtlog
fi


msize=(20 21 22 23 24 25 26)
snodes=(1  2  3	 4  5  6  7)
nmax=( 23 23 25 26 26 26 26)

for msc in ${msize[@]}
do 

for i in ${!snodes[@]}
do
	n=${snodes[$i]}
	if [ $msc -le ${nmax[$i]} ]
	then
		qsub -v p=`expr $n \* $n`,srp=$n,scale=$msc -l nodes=`expr $n \* $n`:ppn=1 keeneland.sh
	fi
done

done 