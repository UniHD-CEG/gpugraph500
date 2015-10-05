#!/bin/bash


astyle=`which astyle`

if [ ! -f "$astyle" ] || [ ! -x "$astyle" ]; then
    echo "error:: astyle is not installed"
    exit 1
else
	$astyle  --style=allman --recursive --options=../cpu_2d/astyle.conf ../cpu_2d/\*.h ../cpu_2d/\*.hpp ../cpu_2d/\*.hh ../cpu_2d/\*.c ../cpu_2d/\*.cpp ../cpu_2d/\*.cu
fi



