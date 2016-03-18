#!/bin/bash

function banner {
    local text="$1"
    echo
    echo "============================================================================"
    echo "== $text"
    echo "============================================================================"
}

function section_banner {
    local text="$1"
    echo
    echo "--- $text"
}

function error {
	msg="$1"
	echo ""
	echo "error:: $msg"
	exit 1
}

function print_stats {
	local directory="$1"
	local regions="$2"
	local cubex="${directory}/profile.cubex"

	banner "Profiling regions in .cubex file"
	for metric in ${metrics[*]}; do
		section_banner "${metric}:"
		cube_stat -p -m $metric -r $regions $cubex
	done
}



if [ $# -lt 1 ] || [ "x$1" = "x-h" ] || [ "x$1" = "x--help" ]; then
    echo "This script will print out the statisctics of a scalasca/scorep .cubex file"
    echo "Usage: $0 <directory with profile.cubex>"
    exit 1
fi

if [ ! -d $1 ];then
	error "$1 must be a directory."
fi

if [ ! -f "$1/profile.cubex" ]; then
	error "$1/profile.cubex is not a file."
fi

cube_stat=`which cube_stat`

if [ ! -x $cube_stat ];then
	error "scalasca, scorep and CUBE (non Qt version) must be installed."
fi


# declare -a metrics=("time" "bytes_sent" "bytes_received")
declare -a metrics=("time")
regions="BFSRUN_region_vertexBroadcast,BFSRUN_region_allReduceBC,BFSRUN_region_localExpansion,BFSRUN_region_columnCommunication,BFSRUN_region_rowCommunication"

print_stats $1 $regions
