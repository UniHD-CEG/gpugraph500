#!/bin/bash


astyle=`which astyle`

if [ ! -f "$astyle" ] || [ ! -x "$astyle" ]; then
    echo "error:: astyle is not installed."
    exit 1
fi

if [ ! -f "./astyle.conf" ]; then
    echo "error:: astyle.conf not found in this directory."
    exit 1
fi

$astyle  --style=allman --recursive --options=./astyle.conf ./\*.h ./\*.hpp ./\*.hh ./\*.c ./\*.cpp ./\*.cu



