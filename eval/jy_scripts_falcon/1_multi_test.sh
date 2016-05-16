#!/bin/bash

if [ ! -d result ]
then
	mkdir result
fi

gpus=(1)
#scale factors
#msize=(15)
#msize=(15 16 17 18 19 20 21 22 23 24 25 26 27)
msize=(22 27)
#square root of nodes
snodes=(8 7 6 5)
#snodes=(3 4)
#BTR
btrs=(64)
#btcs=(128 256 512)

#All test variations
#declare -a g500ver=("g500_CompressionNoOptimizedNoOMP" "g500_CompressionOptimizedNoOMP" "g500_NoCompressionNoOptimizedNoOMP" "g500_NoCompressionOptimizedNoOMP" "g500_CompressionNoOptimizeOMP" "g500_CompressionOptimizedOMP" "g500_NoCompressionNoOptimizedOMP" "g500_NoCompressionOptimizedOMP")

#Restrict tests for ScoreP
#declare -a g500ver=("g500_CompressionNoOptimizedNoOMP" "g500_CompressionOptimizedNoOMP" "g500_NoCompressionNoOptimizedNoOMP" "g500_NoCompressionOptimizedNoOMP")
#declare -a g500ver=("scp_CompOptOMP" "scp_CompOptNoOMP" "scp_NoCompOptNoOMP" "scp_NoCompOptOMP" "scp_CompNoOptOMP" "scp_CompNoOptNoOMP" "scp_NoCompNoOptNoOMP" "scp_NoCompNoOptOMP")
declare -a g500ver=("scp_CompOptOMP")

for g500 in "${g500ver[@]}"
do
echo "Copying over a new g500 version, $g500ver"
cd ../cpu_2d/
cp -f g500_versions/${g500} g500
#Sleep to try and get NFS to sync properly across nodes
sleep 20
cd -
log="${g500}_may16_numa"

for msc in ${msize[@]}
do 
    for i in ${snodes[@]}
    do
        #for ngpus in ${gpus[@]}
        #do
        ngpus=1
        
        #Restrict max scale factor based on node count
        if [ x"$i" = "x1" ]; then
          maxsf=22
        elif [ x"$i" = "x2" ]; then
          maxsf=23
        elif [ x"$i" = "x3" ]; then
          maxsf=25
        elif [ x"$i" = "x4" ]; then
          maxsf=25
        elif [ x"$i" = "x5" ]; then
          maxsf=26
        else
          maxsf=27
        fi

        if [ "$msc" -le "$maxsf" ]; then
            
            for btr in ${btrs[@]}
            do 
                # score-p
		# before script execution:
		export G500_ENABLE_RUNTIME_SCALASCA=yes

		# pass score-p's $SCOREP_EXPERIMENT_DIRECTORY to mpirun
                
		#mpirun -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -x SCOREP_EXPERIMENT_DIRECTORY=$SCOREP_EXPERIMENT_DIRECTORY -np ${np} -hostfile ${hostfile} --display-map -bynode ./test2.sh ${sf} ${c} "s4-bp128-d4" 64 | tee ${file}
                #for btc in ${btcs[@]}
                #do
	            srp=$i
                    n=`expr $i \* $i` 
                    logfile="result/falcon_${log}_nd${n}_scale${msc}_btr${btr}_btc${btr}.log"
                    scorepdir="scorep_results/falcon_${log}_nd${n}_scale${msc}_btr${btr}.scorep"
		    if [ "x$G500_ENABLE_RUNTIME_SCALASCA" = "xyes" ]; then
			export SCOREP_EXPERIMENT_DIRECTORY=${scorepdir}
                        echo "SCOREPDIR is ${SCOREP_EXPERIMENT_DIRECTORY}"
         	    fi	
                    echo "./2_test_falcon.sh nd$n srp$srp msc$msc gpu$ngpus btr$btr btc$btr $logfile $scorepdir"
                    echo ""
                    ./2_test_falcon.sh $n $srp $msc $ngpus $btr $btr $logfile $scorepdir
                    sleep 1 
                #done

            done
        fi
        
        #done

    done 

done
done
