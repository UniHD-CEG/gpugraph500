#!/bin/bash -x

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
    filename $jobid file
    is_valid=`grep "Validation: passed" $file`

    if [ "x$is_valid" != "x" ]; then
                echo "-> File $file. Validation passed (jobid=$jobid). "
    else
        echo "Error. execution ended with error. File $file"
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
    echo_null=`echo "$1" | tee -a "$2"`
    res=$?
    echo_null=`echo "" | tee -a "$2"`

    if [ $res -ne 0 ]; then
        echo "Write error."
        exit 1
    fi
}

function read_jobids {

    echo -n "Enter a a list of Slurm's JOBIDs. Separate them with spaces: "
    read jobids < /dev/tty
    n_jobids=`echo "$jobids" | wc -w`
    if [ $n_jobids = "0"  ]; then
       echo "Error entering label(s). 0 label(s) were entered."
    else
       jobids="c(`echo "$jobids" | sed 's/ /,/g'`)"
    fi

    eval "$1=$jobids"
    eval "$2=$n_jobids"
}

function read_variable {
    jobid=$1
    variable=$2
    filename $jobid file

    var=`grep $variable $file`
    if [ "x$var" = "x" ]; then
        echo "Error. Variable $variable in file $file could not be read."
        exit 1
    fi
    var=`echo "$var" | sed 's/ //g' | sed 's/:/<-/g' | sed 's/^\(.*\)/id__'$jobid'__\1/'`
    eval "$3=$var"
}

function iterate_sf {
    sf=$1

    reads_jobids jobids n_jobids
    variables=""
    for jobid in "$jobids" {
         verify_file jobid
         read_variable jobid "mean_time" variable
         variables="$variables $variable"
         get_labels "$jobids" $sf labels
    }
    var=`echo "$variables" | sed 's/ /;/g'`

    eval "$2=$labels"
    eval "$3=$variables"
    eval "$4=$new_number_jobids"
}


function get_labels {
    jobids="$1"
    sf=$2
    n_jobids=`echo "$jobids" | wc -w`
    default_labels=""
    for jobid in "$jobids"; do
        read_variable jobid "Tasks" default_label
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
          read labels < /dev/tty
          n_labels=`echo "$labels" | wc -w`
          if [ $n_labels != "$n_jobids"  ]; then
             echo "Error entering label(s). $n_labels label(s) were entered."
          else
             continue_loop="n"
             labels_x="c(`echo "$labels" | sed 's/ /,/g'`)"
          fi

       else
          continue_loop="n"
          labels_x=$default_labels
       fi
    done
    eval "$3=$labels_x"
}


function generate_r_variables {
    variables="$1"
    labels="$2"
    result_file="$3"

    write_r $result_file "$variables"
    write_r $result_file "labels_x <- $labels"
}

function generate_r_plotcode {
    result_file="$1"

    write_r $result_file "line()"
}

function build {
    echo ""
    sfs="$1"
    n_sfs=$2
    token="-`echo "$sfs" | sed 's/ /-/g'`"

    create_r_file "$token" result_file
    old_number_jobids=0
    for sf in $sfs; do

        iterate_sf $sf labels variables new_number_jobids

        if [ $old_number_jobids -eq 0 ]; then
            old_number_jobids=new_number_jobids
        else
            if [ $old_number_jobids -ne $new_number_jobids ]; then
                echo "Error. The number of JOBIDs should be the same for each Scale Factor."
                exit 1
            fi
        fi

        generate_r_variables "$variables" "$labels" "$result_file"
    done

    generate_r_plotcode "$result_file"

    echo "-> Created file \"$result_file\". "
    echo "-> R-Code successfully generated.";
    echo ""
}



if [ $# -lt 1 ] || [ "x$1" = "x-h" ] || [ "x$1" = "x--help" ]; then
    echo "This script will generate code to display a comparative plot of the mean-time of 1-N JOBIDS in 1-M SCALE_FACTOR"
    echo "Usage: $0 <SCALE_FACTOR> [SCALE_FACTOR] [SCALE_FACTOR] [SCALE_FACTOR] ..."
    exit 1
fi
build "$*" $#
