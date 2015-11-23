#!/bin/bash

#
# This script will generate code to display a diagram comparing the selected value in variable $VARIABLE (below) for different scale-factors (how they scale)
# Usage: $0 <SCALE_FACTOR> [SCALE_FACTOR] [SCALE_FACTOR] [SCALE_FACTOR] ...
#

function filename {
    jobid=$1
    local prefix=`echo $VARIABLE | awk '{print tolower($0)}'`
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
    local prefix=`echo $VARIABLE | sed -e 's/://g' -e 's/_/-/g' -e 's/=//g' -e 's/ //g' | awk '{print tolower($0)}'`
    local r_file="${prefix}$1.r"
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

    echo -n "Enter a JOBIDs for SF ${sf}: "
    read jobids < /dev/tty
    n_jobids=`echo "$jobids" | wc -w`
    if [ $n_jobids -eq 0 ]; then
       echo "Error entering the JOBID. none was entered."
       exit 1
    fi
    if [ $n_jobids -ne 1 ]; then
       echo "Error entering the JOBID. One JOBID must be entered."
       exit 1
    fi

    eval "$2=\"$jobids\""
    eval "$3=$n_jobids"
}

function read_variable {
    jobid="$1"
    sf=$2
    variable="$3"
    filename $jobid file

    var=`tail -121 $file | grep $variable | head -1`
    if [ "x$var" = "x" ]; then
        echo "Error. Variable $variable in file $file could not be read."
        exit 1
    fi
    token="id__${sf}__${jobid}__"
    var=`echo "$var" | sed 's/ //g' | sed 's/:/ <- /g' | sed 's/^\(.*\)/'$token'\1/'`
    eval "$4=\"$var\""
}

function generate_r_variables {
    sf="$1"
    results_file=$2
    labels_x_set=$3

    read_jobids $sf jobids n_jobids
    for jobid in $jobids; do
         verify_file $jobid $sf
    done

    for jobid in $jobids; do
         read_variable $jobid $sf "${VARIABLE}:" variable
         write_r "$variable" $results_file
         read_variable $jobid $sf "total_gpus:" gpus
         write_r "$gpus" $results_file
    done

    values=""
    for jobid in $jobids; do
        token="id__${sf}__${jobid}__"
        if [ "x$values" = "x" ]; then
            values="get(paste('$token','$VARIABLE',sep=''))"
        else
            values="$values,get(paste('$token','$VARIABLE',sep=''))"
        fi
    done
    values="c($values)"
    write_r "vect__${sf} <- $values" $results_file

    if [ $labels_x_set -eq 0 ]; then
        get_labels "$jobids" $sf labels_x
        write_r "labels_x <- $labels_x" $results_file
        labels_x_set=1
    fi

    eval "$4=\"$jobids\""
    eval "$5=$n_jobids"
    eval "$6=$labels_x_set"
}

function get_labels {
    jobids="$1"
    sf=$2

    n_jobids=`echo "$jobids" | wc -w`
    default_labels=""
    for jobid in $jobids; do
        if [ "x$default_labels" = "x" ]; then
            default_labels="paste(get(paste('id__',$sf,'__',$jobid,'__total_gpus',sep='')),'-Process',sep='')"
        else
            default_label="paste(get(paste('id__',$sf,'__',$jobid,'__total_gpus',sep='')),'-Process',sep='')"
            default_labels="$default_labels $default_label"
        fi
    done
    default_labels=`echo $default_labels | sed 's/ /,/g'`
    default_labels="c($default_labels)"

#    continue_loop="y"
#    while [ "x$continue_loop" = "xy" ]; do
#       echo -n "Enter labels for the X-Axe? (y/n) [n] "
#       read yesno < /dev/tty
#       if [ "x$yesno" = "xy" ];then
#          echo -n "Enter a total $n_jobids label(s) between quoutes. Separate them with spaces: "
#          read labels_temp < /dev/tty
#          n_labels_temp=`echo "$labels_temp" | wc -w`
#          if [ $n_labels_temp != "$n_jobids" ]; then
#             echo "Error entering label(s). $n_labels_temp label(s) were entered."
#         echo -n "Re-"
#          else
#              continue_loop="n"
#              labels_x=""
#              for label_x in $labels_temp; do
#                  if [ "x$labels_x" = "x" ]; then
#                      labels_x="'$label_x'"
#                  else
#                      labels_x="$labels_x,'$label_x'"
#                  fi
#              done
#             labels_x="c($labels_x)"
#          fi
#       else
#          continue_loop="n"
#          labels_x="$default_labels"
#       fi
#    done
    labels_x="labels_x <- c('${LABEL}')"

    eval "$3=\"$labels_x\""
}

function generate_r_plotcode {
    result_file="$1"
    sfs="$2"
    n_sfs="$3"
    jobids="$4"

    labels_y=`echo "$sfs" | sed 's/ / sf/g' | sed 's/^/sf/'`
    new_labels_y=""
    for label_y in $labels_y; do
        if [ "x$new_labels_y" = "x" ]; then
            new_labels_y="\"$label_y\""
        else
            new_labels_y="$new_labels_y,\"$label_y\""
        fi
    done

    values=""
    for sf in $sfs; do
        token="vect__${sf}"
        if [ "x$values" = "x" ]; then
            values="$token"
        else
            values="$values,$token"
        fi
    done
    values="c($values)"

    matrix="mat <- matrix($values, nrow = $n_sfs, ncol = length(labels_x), byrow = TRUE, dimnames = list(labels_y, labels_x))"

    plotcode="par(xpd = TRUE)
    matplot(mat, axes = FALSE, xlab='scale factor (#)', ylab = '${YAXIS} ${UNIT}',
        main = 'Execution on Fermi. Scale factor comparison', col = topo.colors(length(rownames(mat)), alpha=1),
        type='o', lty=1, lwd=1, pch=3)
        axis(1, at = 1:length(colnames(t(mat))), labels= labels_y)
        axis(2)
        legend('top', fill = topo.colors(length(rownames(mat)), alpha=1),
        legend = labels_x)"

    write_r "labels_y <- c($new_labels_y)" $result_file
    write_r "$matrix" $result_file
    write_r "$plotcode" $result_file
}

function build {
    echo ""
    sfs="$1"
    n_sfs=$2
    token=`echo "$sfs" | sed 's/ /\-/g'`
    token="-$token"

    create_r_file "$token" result_file
    old_number_jobids="0"
    jobid_list=""
    labels_x_set=0
    for sf in $sfs; do

        generate_r_variables $sf $result_file $labels_x_set jobids new_number_jobids labels_x_set

        if [ "x$old_number_jobids" = "x0" ]; then
            old_number_jobids=$new_number_jobids
        else
            if [ "x$old_number_jobids" != "x$new_number_jobids" ]; then
                echo "Error. The number of JOBIDs should be the same for each SF."
                exit 1
            fi
        fi

        if [ "x$jobid_list" = "x" ]; then
            jobid_list="$jobids"
        else
            jobid_list="$jobid_list $jobids"
        fi
    done

    generate_r_plotcode "$result_file" "$sfs" $n_sfs "$jobid_list"

    echo "-> Created file \"$result_file\". "
    echo "-> R-Code successfully generated.";
    echo ""
}

# no trailing ':'
VARIABLE="harmonic_mean_TEPS"
DESCRIPTION="Total Edges Per Second (EPSs)"
LABEL="TEPS"
YAXIS="edges per second"
UNIT="(s)"

if [ $# -lt 1 ] || [ "x$1" = "x-h" ] || [ "x$1" = "x--help" ]; then
    echo "This script will generate code to display a diagram comparing the ${DESCRIPTION} for different scale-factors (How they scale)."
    echo ""
    echo "Usage: $0 <SCALE_FACTOR> [SCALE_FACTOR] [SCALE_FACTOR] ..."
    exit 1
fi


build "$*" $#


