#!/bin/bash

if [ ! -d result ]
then
	mkdir result
fi

gpus=(1)
#scale factors
#msize=(15)
msize=(15 16 17 18 19 20 21 22 23 24 25 26 27)
#square root of nodes
snodes=(1 2 3 4 5 6 7 8)
#snodes=(3 4)
#BTR
btrs=(64)
#btcs=(128 256 512)

declare -a g500ver=("g500_CompressionNoOptimizedNoOMP" "g500_CompressionOptimizedNoOMP" "g500_NoCompressionNoOptimizedNoOMP" "g500_NoCompressionOptimizedNoOMP" "g500_CompressionNoOptimizeOMP" "g500_CompressionOptimizedOMP" "g500_NoCompressionNoOptimizedOMP" "g500_NoCompressionOptimizedOMP")

for g500 in "${g500ver[@]}"
do
echo "Copying over a new g500 version, $g500ver"
cd ../cpu_2d/
cp -f g500_versions/${g500} g500
cd -
log="${g500}_mar16"

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
                #for btc in ${btcs[@]}
                #do
	            srp=$i
                    n=`expr $i \* $i` 
                    logfile="result/falcon_${log}_nd${n}_scale${msc}_btr${btr}_btc${btr}.log"
                    echo "./2_test_falcon.sh nd$n srp$srp msc$msc gpu$ngpus btr$btr btc$btr $logfile"
                    echo ""
                    ./2_test_falcon.sh $n $srp $msc $ngpus $btr $btr $logfile
                    sleep 2
                #done

            done
        fi
        
        #done

    done 

done
done
