#!/bin/bash


if [ $# -lt 1 ] || [ "x$1" = "x-h" ] || [ "x$1" = "x--help" ]; then
    echo "This script will search for the file slurm-<JOBID>.out and will generate R-Code for with the extracted data."
    echo "Usage: $0 <JOBID> [JOBID] [JOBID] [JOBID] ..."
    exit 1
fi

main "$*" $#
