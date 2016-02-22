#!/bin/bash

if [ x"$1" = x ]; then
	echo "error, string needed"
	echo "$0 <idOrCompileSting> <hostfile> <MaxSF>"
	exit 1
fi

if [ x"$2" = x ] && [ ! -f $2 ]; then
        echo "error, hostfile needed"
        echo "$0 <idOrCompileSting> <hostfile> <MAxSF>"
        exit 1
fi

if [ x"$3" = x ]; then
        echo "error, Maximum SF needed"
        echo "$0 <idOrCompileSting> <hostfile> <MaxSF>"
        exit 1
fi

cluster=`hostname`
str="$1"
hostfile=$2
maxsf=$3
numNodes="2"
sfs=`seq 16 ${maxsf}`

for c in `echo $numNodes`; do
        np=$((c * c))
        if [ x"$c" = "x1" ]; then
          maxsf1=21
          sfs=`seq 13 ${maxsf1}`
        elif [ x"$c" = "x2" ]; then
          maxsf2=22
          sfs=`seq 13 ${maxsf2}`
        else
          sfs=`seq 13 ${maxsf}`
        fi
	for sf in `echo ${sfs}`; do
		file="${cluster}_np${np}_sf${sf}_${str}.log"
		result=""
		if [ -f ${file} ]; then
			result=`cat ${file} | grep ": passed"` 
		fi
		if [ x"$result" != x ];then
                        echo "Skipping np: $np scale: $sf type: $str on [$cluster] ..."
			continue
		fi	
		mpirun -np ${np} -hostfile ${hostfile} --display-map -bynode ./test2.sh ${sf} ${c} "s4-bp128-d4" 64 | tee ${file}
	done
done
