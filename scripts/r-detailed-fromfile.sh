#!/bin/bash

#
# This script will generate code to display a comparative plot of the mean-time of 1-N JOBIDS in 1-M SCALE_FACTOR
# Usage: $0 <SCALE_FACTOR> [SCALE_FACTOR] [SCALE_FACTOR] [SCALE_FACTOR] ...
#

function confirm {
    local message=$1

    if [ x$force = xno ]; then
        echo -n "${message} Do you want to continue? [Y/n] "
        read yesno < /dev/tty
        if [ "x$yesno" = "xn" ] || [ "x$yesno" = "xN" ];then
          echo "Quiting ..."
          exit 1
        fi
    fi
}

function verify_file {
    local file=$1
    local sf=$2
    local np=$3

    # filename $jobid file
    local is_valid=`grep "Validation: passed" $file`
    local scale_factor=`tail -125 $file | grep "SCALE"`
    local scale_factor=${scale_factor##*:\ }

    local processes=`tail -125 $file | grep "total_gpus"`
    local processes=${processes##*:\ }

    if [ "x$is_valid" = "x" ]; then
        echo "Error. execution ended with error. File $file"
        exit 1
    fi
    if [ "x$scale_factor" = "x$sf" ] && [ "x$processes" = "x$np" ]; then
        echo "-> File $file. Validation passed. "
    else
        echo "Error. Wrong SF:${sf} or NP:${np} in file ${file}."
        exit 1
    fi
}

function create_r_file {
    # local r_file="file$1.r"
    local r_file="detail$1.r"

    if [ -f $r_file ]; then
        confirm "File $r_file is about to be deleted."
    fi

    rm -f $r_file
    touch $r_file
    res=$?
    if [ $res -ne 0 ]; then
        echo "Error creating $r_file"
        exit 1
    fi
    eval "$2='$r_file'"
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


function read_variable {
    local file="$1"
    local variable="$2"
    local sf=$3
    local np=$4
    local scenario="$5"

    var=`tail -121 "$file" | grep $variable | head -1`
    if [ "x$var" = "x" ]; then
        echo "Error. Variable $variable in file $file could not be read."
        exit 1
    fi
    token="variable__${np}_${sf}_${scenario}__"


    var=`echo "$var" | sed 's/ //g' | sed 's/:/ <- /g' | sed 's/^\(.*\)/'$token'\1/'`
    eval "$6=\"$var\""
}

function generate_r_variables {
    local results_file=$1
    local labels_x_set=$2
    local files="$3"
    local scenario="$4"

    # available y-axis variables. The selected y-axis is set in $variable_token
    local i=1
    for file in $files; do
        read_variable "$file" "harmonic_mean_TEPS:" ${sfi[$i]} ${npi[$i]} "${scenario}" tepsvar
        write_r "$tepsvar" $results_file
        read_variable "$file" "mean_time:" ${sfi[$i]} ${npi[$i]} "${scenario}" mean_time
        write_r "$mean_time" $results_file
        read_variable "$file" "total_gpus:" ${sfi[$i]} ${npi[$i]} "${scenario}" gpus
        write_r "$gpus" $results_file
        read_variable "$file" "mean_local_bfs_time:" ${sfi[$i]} ${npi[$i]} "${scenario}" local_time
        write_r "$local_time" $results_file
        read_variable "$file" "mean_row_com_time:" ${sfi[$i]} ${npi[$i]} "${scenario}" row_time
        write_r "$row_time" $results_file
        read_variable "$file" "mean_column_com_time:" ${sfi[$i]} ${npi[$i]} "${scenario}" col_time
        write_r "$col_time" $results_file
        read_variable "$file" "mean_predecessor_list_reduction_time:" ${sfi[$i]} ${npi[$i]} "${scenario}" pred_time
        write_r "$pred_time" $results_file
    i=$((i+1))
    done

    local variables="mean_local_bfs_time mean_row_com_time mean_column_com_time mean_predecessor_list_reduction_time mean_time"

    for variable in ${variables}; do
        values=""
        i=1
        np=0
        sf=0
        for file in $files; do
            token="variable__${npi[$i]}_${sfi[$i]}_${scenario}__${variable}"
            np=${npi[$i]}
            sf=${sfi[$i]}
            if [ "x$values" = "x" ]; then
                values="get(paste('$token',sep=''))"
            else
                values="$values,get(paste('$token',sep=''))"
            fi
            i=$((i+1))
        done
        values="c($values)"

        i=1
        for file in $files; do
            verify_file $file ${sfi[$i]} ${npi[$i]}
            i=$((i+1))
        done

        echo ""

        if [ x"$isset_sf" = xyes ]; then
            write_r "vector__${sf0}_${scenario}__${variable} <- $values" $results_file
        fi
        if [ x"$isset_np" = xyes ]; then
            write_r "vector__${np0}_${scenario}__${variable} <- $values" $results_file
        fi
    done

 # create X-AXIS labels only once
 labels_x=""
 if [ $labels_x_set -eq 0 ]; then
     i=1
     labels_x=""
     for file in $files; do
       if [ "x$labels_x" = "x" ]; then
           labels_x="'${xlabels[$i]}'"
       else
           labels_x="$labels_x,'${xlabels[$i]}'"
       fi
       i=$((i+1))
     done
     labels_x="c($labels_x)"

     write_r "labels_x <- $labels_x" $results_file
     labels_x_set=1
 fi

    labels_x_set=1
    eval "$5=$labels_x_set"
}

function generate_r_plotcode {
    local result_file="$1"
    local scenario=""

    declare -a colors=('red', 'cornflowerblue' 'darkred' 'azure4' 'blue4' 'brown3' 'blue1')
    declare -a pointshapes=('1', '18' '15' '16' '17' '19' '20')

    colormatrix=""
    pointmatrix=""
    for index in ${!scenarios[@]}; do
        if [ x"${scenarios[$index]}" != x ]; then
            if [ x"${colormatrix}" = x ]; then
                colormatrix="'${colors[$index]}'"
                pointmatrix="${pointshapes[$index]}"
            else
                colormatrix="${colormatrix}, '${colors[$index]}'"
                pointmatrix="${pointmatrix}, ${pointshapes[$index]}"
            fi
        fi
    done
    colormatrix="c(${colormatrix})"
    pointmatrix="c(${pointmatrix})"

    values=""
    xvariable=""
    yvariable=""
    if [ x"$isset_sf" = xyes ]; then
        token="vector__${sf0}__${scenario}"
        xvariable="Problem size (Scale factor)"
    fi
    if [ x"$isset_np" = xyes ]; then
        token="vector__${np0}__${scenario}"
        xvariable="Number of Processors"
    fi

    if [ x"$teps" = xyes ]; then
        yvariable="Traversed Edges Per Second (TEPS)"
    fi
    if [ x"$ttime" = xyes ]; then
        yvariable="Total time (s)"
    fi

    local title="${yvariable}"
    if [ x"$isset_title" = xyes ]; then
        title="${gtitle}"
    fi

    if [ "x$values" = "x" ]; then
        values="$token"
    else
        values="$values,$token"
    fi
    values="c($values)"

    # Legend labels
    labelscenario=""
    if [ x"$isset_ci" = xyes ]; then
        i=1
        for index in ${!scenarios[@]}; do
            if [ x"${scenarios[$index]}" != x ]; then
                if [ x"$labelscenario" = x ]; then
                    labelscenario="'${yvariable} (${scenarios[$index]})'"
                else
                    labelscenario="${labelscenario}, '${yvariable} (${scenarios[$index]})'"
                fi
            fi
            i=$((i+1))
        done
    else
        labelscenarios="'${yvariable}'"
    fi
    labelscenario="c(${labelscenario})"

    totals_points="points(x=p, y=totals, col='firebrick', pch = 17)
                 lines(x=p, y=totals, col='firebrick', lty = 4)"

    local values_total_time="'mean_time'"
    local values="'mean_local_bfs_time' 'mean_row_com_time' 'mean_column_com_time' 'mean_predecessor_list_reduction_time'"
    local labels_x_text="'Expansion','Row com' ,'Col com','Pred list red' ,'Total time'"
    local num_labels_x=`echo "$values" | wc -w`
    local rows="c('Expansion','Row com' ,'Col com','Pred list red')"

    criteria=""
    if [ x"$isset_sf" = xyes ]; then
        criteria=${sf0}
    fi
    if [ x"$isset_np" = xyes ]; then
        criteria=${np0}
    fi

    labels_y=""
    for var in $values; do
        for index in ${!scenarios[@]}; do
            labels_y_paste="get(paste('vector__',${criteria},'_','${scenarios[$index]}','__',${var},sep=''))"
            if [ "x$labels_y" = "x" ]; then
                labels_y="$labels_y_paste"
            else
                labels_y="$labels_y, $labels_y_paste"
            fi
        done
        labels_total_time="get(paste('vector__',${criteria},'_','${scenarios[$index]}','__','mean_time',sep=''))"
    done
    labels_y="c($labels_y)"
    num_labels_y=`echo "$values" | wc -w`




    local ccolors="c('blue4', 'lightskyblue', 'cornflowerblue', 'azure3', 'brown3', 'blue1')"

    local plot="par(xpd = F, mar = c(5.1, 4.1, 4.1, 7.1))
        acc <- 0
        mm <- apply(m,1,max)
        for(i in 1:${num_labels_x}){
            acc<-acc+mm[i]
        }
        maximumacumulated <- round(as.numeric(acc), digits = 1)
        maximumacumulated <- min(maximumacumulated, 1.0)

        p <-  barplot(m, axes = F, axisnames = F, ylim = c(0, maximumacumulated), cex.lab = 0.9,
            main = '${title}', border = NA, ylab = '${yvariable}', xaxt = 'n', yaxt = 'n', xlab='${xvariable}',
            col = ${ccolors})
        axis(1, at = barplot(m, plot = F), labels = $labels_x,  cex.axis=0.60)
        axis(2, seq(0, maximumacumulated, 0.05), cex.axis=0.75)
        abline(h=seq(0, maximumacumulated, by = 0.05),col=c('grey25'), lty = 3)

        par(xpd=T)
        legend('topright', inset = c(-0.17, 0),
        legend = c($labels_x_text), fill = ${ccolors}, cex = 0.75)"


    write_r "m <- matrix($labels_y, nrow = length(${rows}), ncol = length(${labels_x}), byrow = TRUE, \
        dimnames = list(${rows}, ${labels_x}))" $result_file

    write_r "totals<-${labels_total_time}" $result_file
    write_r "$plot" $result_file
    write_r "$totals_points" $result_file

}

function build {
    echo ""
    local inputfile="$1"
    local np=$2
    local sf=$3
    local t=$4
    local filter_string="$5"
    local labels_x_set=0

    create_r_file $filter_string result_file

    local grep_filter_np=""
    local grep_filter_sf=""
    local grep_filter_t=""
    if [ $np -ne 0 ]; then
        grep_filter_np="np:${np} "
    fi
    if [ $sf -ne 0 ]; then
        grep_filter_sf="sf:${sf} "
    fi
    if [ $t -ne 0 ]; then
        grep_filter_t="t:${t} "
    fi

    for index in ${!scenarios[*]}; do

        local lines="`cat $inputfile | grep \"$grep_filter_np\" | grep \"$grep_filter_sf\" | grep \"$grep_filter_t\" | grep \"compareid:${scenarios[$index]}$\"`"


        nlines=`echo "$lines" | wc -c`
        if [ ${nlines} -eq 1 ];then
            echo "'comparisonid:${scenarios[$index]}' not found. skiping..."
            scenarios[$index]=''
            continue
        fi

        # array $files is declared global
        files=""
        for line in "$lines"; do
            file=`echo "$line" | cut -d ' ' -f4`
            files="${files}${file}"
        done

        npss=()
        sfss=()
        for line in "$lines"; do

            nps=`echo "$line" | cut -d ' ' -f1 2>/dev/null | cut -d ':' -f2 2>/dev/null`
            sfs=`echo "$line" | cut -d ' ' -f2 2>/dev/null | cut -d ':' -f2 2>/dev/null`

            npss+=("$nps")
            sfss+=("$sfs")
        done


        nlabels=`echo "$npss" | wc -w`
        for i in `seq 1 $nlabels`; do
            npss0=`echo ${npss} | cut -d ' ' -f${i}`
            sfss0=`echo ${sfss} | cut -d ' ' -f${i}`
            if [ "x${isset_sf}" = "xyes" ] && [ "x${isset_np}" = "xno" ]; then
                xlabels[$i]="gpus:$npss0"
            elif [ "x${isset_sf}" = "xno" ] && [ "x${isset_np}" = "xyes" ]; then
                xlabels[$i]="scale:$sfss0"
            else
                xlabels[$i]="gpus:$npss0 \\n scale:$sfss0"
            fi
            # xlabels[$i]="gpus:$npss0 \\n scale:$sfss0"
            npi[$i]="$npss0"
            sfi[$i]="$sfss0"
        done

        generate_r_variables $result_file $labels_x_set "$files" "${scenarios[$index]}" labels_x_set
    done

    generate_r_plotcode "$result_file"

    echo ""
    echo "-> Created file \"$result_file\". "
    echo "-> R-Code successfully generated.";
    echo ""
}

function printhelp {
    echo ""
    echo "this script will generate code to display a comparative plot of the mean-times between scale-factors"
    echo ""
    echo "Usage: $0 [-y] -f <file> -np <# processes> -sf <scale factor> [-t <threshold>] [-title 'graph title'] [[-ci <compareid>][-ci <compareid>]...]"
}

# sf, np
filters=0
# tesps, ttime
variables=0
force=no
ci=""
gtitle=""

# X-AXIS
isset_file=no
isset_np=no
isset_sf=no
isset_ci=no
isset_t=yes
isset_title=no

np0=0
sf0=0
t0=128

filter_string_t="_t128"
option_string=""
ci_string=""

# Y-AXIS
# globals (as options will increase. not suitable passing as parameter)
teps=no
ttime=no
variable_token=""

# Bash: it is not possible to pass an array as parameter. passing globally
declare -a xlabels
declare -a sfi
declare -a npi
declare -a scenarios

while [[ $# > 0 ]]
do
key="$1"

case $key in
    -f|--file)
    inputfile0="$2"
    isset_file=yes
    shift
    ;;
    -np|--processes)
    np0="$2"
    filters=$((filters+1))
    filter_string_np="_np${np0}"
    isset_np=yes
    shift
    ;;
    -sf|--scale)
    sf0="$2"
    filters=$((filters+1))
    filter_string_sf="_sf${sf0}"
    isset_sf=yes
    shift
    ;;
    -ci|--compareid)
    if [ x"$ci" = x ]; then
        ci="$2"
    else
        ci="${ci}__$2"
    fi
    ci_string="_${ci}"
    isset_ci=yes
    shift
    ;;
    -t|--threshold)
    t0="$2"
    # this one is not a filter, but an option
    filter_string_t="_t${t0}"
    shift
    ;;
    -y|--yes|--force)
    force=yes
    ;;
    -title)
    gtitle="$2"
    isset_title=yes
    shift
    ;;
    -h|--help)
    printhelp
    exit 1
    ;;
    *)
    # unknown option
    ;;
esac
shift
done

if [ x"$isset_file" = "xno" ]; then
    echo "You must provide an input file. Option: -f <file>"
    printhelp
    exit 1
fi

if [ ! -f $inputfile0 ]; then
    echo "Input file does not exist."
    printhelp
    exit 1
fi

if [ $filters -ne 1 ]; then
    echo "You must provide one filter. Options are: -np <# processes> -sf <scale factor>"
    printhelp
    exit 1
fi

number='^[0-9]+$'

if [ x$isset_np = "xyes" ] && [[ ! $np0 =~ $number ]] && [ $np0 -gt 0 ]; then
    echo "Value provided in -np <# processes> must be a number."
    printhelp
    exit 1
fi

if [ x$isset_sf = "xyes" ] && [[ ! $sf0 =~ $number ]] && [ $sf0 -gt 0 ]; then
    echo "Value provided in -sf <scale factor> must be a number."
    printhelp
    exit 1
fi

if [ x$isset_t = "xyes" ] && [[ ! $t0 =~ $number ]] && [ $t0 -gt 0 ]; then
    echo "Value provided in -t <threshold> must be a number."
    printhelp
    exit 1
fi

variables=1
ttime=yes
variable_token="mean_time"
option_string="_time"

# suffix of output R file
filter_string0="${filter_string_np}${filter_string_sf}${filter_string_t}${option_string}${ci_string}"

cis=""
if [ x"$isset_ci" = xyes ]; then
    cis=`echo $ci | sed 's/__/ /g'`
    i=1
    for c in $cis; do
        scenarios[$i]="$c"
        i=$((i+1))
    done
else
    scenarios[1]=""
fi

# start!
build $inputfile0 $np0 $sf0 $t0 $filter_string0





