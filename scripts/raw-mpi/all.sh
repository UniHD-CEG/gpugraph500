#!/bin/bash

file=hosts
maxsf=23
runall=./runall.sh

if [ ! -f $runall ] || [ ! -x $runall ]; then
	echo "edit script and set the runall varable"
	exit 1
fi

string="CompressionOptimizedOMP"
echo "---------------------------------------------------------------"
echo "-- $string"
echo "---------------------------------------------------------------"
echo "Configuring && making ..."
./configure --enable-aggressive-optimizations --enable-cuda-math-optimizations --enable-ptxa-optimizations --enable-openmp && make
echo "Updating directory ..."
ls -r >/dev/null ; ls -r > /dev/null
echo "Running [#1] ..."
$runall "$string" $file $maxsf
echo "Running [#2] ..."
$runall "$string" $file $maxsf


string="CompressionOptimizedNoOMP"
echo "---------------------------------------------------------------"
echo "-- $string"
echo "---------------------------------------------------------------"
echo "Configuring && making ..."
./configure --enable-aggressive-optimizations --enable-cuda-math-optimizations --enable-ptxa-optimizations --disable-openmp && make
echo "Updating directory ..."
ls -r >/dev/null ; ls -r > /dev/null
echo "Running [#1] ..."
$runall "$string" $file $maxsf
echo "Running [#2] ..."
$runall "$string" $file $maxsf

string="CompressionNoOptimizeOMP"
echo "---------------------------------------------------------------"
echo "-- $string"
echo "---------------------------------------------------------------"
echo "Configuring && making ..."
./configure --disable-aggressive-optimizations --disable-cuda-math-optimizations --disable-ptxa-optimizations --enable-openmp && make
echo "Updating directory ..."
ls -r >/dev/null ; ls -r > /dev/null
echo "Running [#1] ..."
$runall "$string" $file $maxsf
echo "Running [#2] ..."
$runall "$string" $file $maxsf


string="CompressionNoOptimizedNoOMP"
echo "---------------------------------------------------------------"
echo "-- $string"
echo "---------------------------------------------------------------"
echo "Configuring && making ..."
./configure --disable-aggressive-optimizations --disable-cuda-math-optimizations --disable-ptxa-optimizations --disable-openmp && make
echo "Updating directory ..."
ls -r >/dev/null ; ls -r > /dev/null
echo "Running [#1] ..."
$runall "$string" $file $maxsf
echo "Running [#2] ..."
$runall "$string" $file $maxsf


string="NoCompressionOptimizedOMP"
echo "---------------------------------------------------------------"
echo "-- $string"
echo "---------------------------------------------------------------"
echo "Configuring && making ..."
./configure --enable-aggressive-optimizations --enable-cuda-math-optimizations --enable-ptxa-optimizations --enable-openmp --disable-compression && make
echo "Updating directory ..."
ls -r >/dev/null ; ls -r > /dev/null
echo "Running [#1] ..."
$runall "$string" $file $maxsf
echo "Running [#2] ..."
$runall "$string" $file $maxsf

string="NoCompressionOptimizedNoOMP"
echo "---------------------------------------------------------------"
echo "-- $string"
echo "---------------------------------------------------------------"
echo "Configuring && making ..."
./configure --enable-aggressive-optimizations --enable-cuda-math-optimizations --enable-ptxa-optimizations --disable-openmp --disable-compression && make
echo "Updating directory ..."
ls -r >/dev/null ; ls -r > /dev/null
echo "Running [#1] ..."
$runall "$string" $file $maxsf
echo "Running [#2] ..."
$runall "$string" $file $maxsf

string="NoCompressionNoOptimizedOMP"
echo "---------------------------------------------------------------"
echo "-- $string"
echo "---------------------------------------------------------------"
echo "Configuring && making ..."
./configure --disable-aggressive-optimizations --disable-cuda-math-optimizations --disable-ptxa-optimizations --enable-openmp --disable-compression && make
echo "Updating directory ..."
ls -r >/dev/null ; ls -r > /dev/null
echo "Running [#1] ..."
$runall "$string" $file $maxsf
echo "Running [#2] ..."
$runall "$string" $file $maxsf

string="NoCompressionNoOptimizedNoOMP"
echo "---------------------------------------------------------------"
echo "-- $string"
echo "---------------------------------------------------------------"
echo "Configuring && making ..."
./configure --disable-aggressive-optimizations --disable-cuda-math-optimizations --disable-ptxa-optimizations --disable-openmp --disable-compression && make
echo "Updating directory ..."
ls -r >/dev/null ; ls -r > /dev/null
echo "Running [#1] ..."
$runall "$string" $file $maxsf
echo "Running [#2] ..."
$runall "$string" $file $maxsf



