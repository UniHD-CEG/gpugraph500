#!/bin/bash

#
# This script will generate code to display a comparative plot of the mean-time of 1-N JOBIDS in 1-M SCALE_FACTOR
# Usage: $0 <SCALE_FACTOR> [SCALE_FACTOR] [SCALE_FACTOR] [SCALE_FACTOR] ...
#

function filename {
    jobid=$1
    if [ ! -f "slurm-$1.out" ]; then
      echo "Error. file slurm-$1.out must exist."
      exit 1
    else
      file="slurm-$1.out"
    fi
    eval "$2=$file"
}

function verify_file {
    jobid=$1
    sf=$2
    filename $jobid file
    is_valid=`grep "Validation: passed" $file`
    scale_factor=`tail -125 $file | grep "SCALE"`
    scale_factor=${scale_factor##*:\ }

    if [ "x$is_valid" = "x" ]; then
        echo "Error. execution ended with error. File $file"
        exit 1
    fi
    if [ "x$scale_factor" = "x$sf" ]; then
        echo "-> File $file. Validation passed. "
    else
        echo "Error. Typed SF does not match file's SF. File $file"
        exit 1
    fi
}

function create_r_file {
    local r_file="file$1.r"
    rm -f $r_file
    touch $r_file
    res=$?
    if [ $res -ne 0 ]; then
        echo "Error creating $r_file"
        exit 1
    fi
    eval "$2=$r_file"
}

function write_r {
    if [ ! -f "$2" ]; then
        echo "Write error."
        exit 1    
    fi

    echo_null=`echo "$1" | tee -a "$2"`
    res=$?
    echo_null=`echo "" | tee -a "$2"`

    if [ $res -ne 0 ]; then
        echo "Write error."
        exit 1
    fi
}

function read_jobids {
    sf=$1

    echo -n "Enter a list of JOBIDs for SF $sf. Separate them with spaces: "
    read jobids < /dev/tty
    n_jobids=`echo "$jobids" | wc -w`
    if [ $n_jobids -eq 0  ]; then
       echo "Error entering JOBID(s). 0 JOBID(s) were entered."
       exit 1
    fi

    eval "$2=\"$jobids\""
    eval "$3=$n_jobids"
}

function read_variable {
    jobid="$1"
    variable="$2"
    filename $jobid file

    var=`tail -121 $file | grep $variable | head -1`
    if [ "x$var" = "x" ]; then
        echo "Error. Variable $variable in file $file could not be read."
        exit 1
    fi
    var=`echo "$var" | sed 's/ //g' | sed 's/:/ <- /g' | sed 's/^\(.*\)/id__'$jobid'__\1/'`
    eval "$3=\"$var\""
}

function generate_r_variables {
    sf="$1"
    results_file=$2

    read_jobids $sf jobids n_jobids
    for jobid in $jobids; do
         verify_file $jobid $sf
    done
 
    for jobid in $jobids; do
         read_variable $jobid "mean_time:" mean_time
         write_r "$mean_time" $results_file
         read_variable $jobid "total_gpus:" gpus
         write_r "$gpus" $results_file
    done

    get_labels "$jobids" $sf labels
    write_r "labels_x <- $labels" $results_file

    eval "$3=$n_jobids"
}


function get_labels {
    jobids="$1"
    sf=$2
    n_jobids=`echo "$jobids" | wc -w`
    default_labels=""
    for jobid in $jobids; do
        #read_variable $jobid "Tasks" default_label
        default_label="paste(get(paste('id__',$jobid,'__total_gpus',sep='')),'-Process',sep='')"
	default_labels="$default_label $default_labels"
    done
    default_labels=`echo $default_labels | sed 's/ /,/g'`
    default_labels="c($default_labels)"


    continue_loop="y"
    while [ "x$continue_loop" = "xy" ]; do
       echo -n "Enter labels for the X-Axe? (y/n) [n] "
       read yesno < /dev/tty
       if [ "x$yesno" = "xy" ];then
          echo -n "Enter a total $n_jobids label(s) between quoutes. Separate them with spaces: "
          read labels_temp < /dev/tty
          n_labels_temp=`echo "$labels_temp" | wc -w`
          if [ $n_labels_temp != "$n_jobids"  ]; then
             echo "Error entering label(s). $n_labels_temp label(s) were entered."
	     echo -n "Re-"	
          else
             continue_loop="n"
             labels_x="c(`echo "$labels_temp" | sed 's/ /,/g'`)"
          fi

       else
          continue_loop="n"
          labels_x=$default_labels
       fi
    done
    eval "$3=\"$labels_x\""
}

function generate_r_plotcode {
    result_file="$1"

    write_r "line()" $result_file
}

function build {
    echo ""
    sfs="$1"
    n_sfs=$2
    token="-`echo "$sfs" | sed 's/ /-/g'`"

    create_r_file "$token" result_file
    old_number_jobids="0"
    for sf in $sfs; do

        generate_r_variables $sf $result_file new_number_jobids

        if [ $old_number_jobids = "0" ]; then
            old_number_jobids=$new_number_jobids
        else
            if [ "$old_number_jobids" != "$new_number_jobids" ]; then
                echo "Error. The number of JOBIDs should be the same for each SF.."
                exit 1
            fi
        fi

    done

    generate_r_plotcode "$result_file"

    echo "-> Created file \"$result_file\". "
    echo "-> R-Code successfully generated.";
    echo ""
}



if [ $# -lt 1 ] || [ "x$1" = "x-h" ] || [ "x$1" = "x--help" ]; then
    echo "This script will generate code to display a comparative plot of the mean-times between cale-factors"
    echo ""
    echo "Usage: $0 <SCALE_FACTOR> [SCALE_FACTOR] [SCALE_FACTOR] ..."
    exit 1
fi
build "$*" $#
