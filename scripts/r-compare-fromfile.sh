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
    # local scale_factor=`tail -125 $file | grep "SCALE"`
    local scale_factor=`cat $file | grep "SCALE"`
    local scale_factor=${scale_factor##*:\ }

    # local processes=`tail -125 $file | grep "total_gpus"`
    local processes=`cat $file | grep "total_gpus"`
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
    local r_file="comparison$1.r"

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

    #var=`tail -121 "$file" | grep $variable | head -1`
    var=`cat "$file" | grep $variable | head -1`
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
    i=$((i+1))
    done

    values=""
    gpus_array=""
    i=1
    np=0
    sf=0
    for file in $files; do
        token="variable__${npi[$i]}_${sfi[$i]}_${scenario}__"
        np=${npi[$i]}
        sf=${sfi[$i]}
        if [ "x$gpus_array" = "x" ]; then
            gpus_array="get(paste('$token','total_gpus',sep=''))"
        else
            gpus_array="$gpus_array,get(paste('$token','total_gpus',sep=''))"
        fi
        if [ "x$values" = "x" ]; then
            values="get(paste('$token','$variable_token',sep=''))"
        else
            values="$values,get(paste('$token','$variable_token',sep=''))"
        fi
        i=$((i+1))
    done
    values="c($values)"
    gpus_array="c($gpus_array)"

    i=1
    for file in $files; do
        verify_file $file ${sfi[$i]} ${npi[$i]}
        i=$((i+1))
    done

    echo ""

    if [ x"$isset_sf" = xyes ]; then
        write_r "vector__${sf0}__${scenario} <- $values" $results_file
    fi
    if [ x"$isset_np" = xyes ]; then
        write_r "vector__${np0}__${scenario} <- $values" $results_file
    fi
    write_r "vector__gpus__${scenario} <- $gpus_array" $results_file

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
    if [ "x${speedup}" = "xyes" ];then
        colormatrix="${colormatrix}, '${colors[$index+1]}'"
        pointmatrix="${pointmatrix}, ${pointshapes[$index+1]}"
    fi

    colormatrix="c(${colormatrix})"
    pointmatrix="c(${pointmatrix})"

    yvariable_pernode=""
    if [ "x${weak}" = "xyes" ]; then
        yvariable_pernode="per Processor"
    fi

    restofscenariosplotcode=""
    for index in ${!scenarios[@]}; do

        scenario=${scenarios[$index]}
        if [ $index -eq 1 ]; then
            values=""
            xvariable=""
            yvariable=""
            if [ x"$isset_sf" = xyes ]; then
                token="vector__${sf0}__${scenario}"
                xvariable="Number of Processors"
            elif [ x"$isset_np" = xyes ]; then
                token="vector__${np0}__${scenario}"
                xvariable="Problem size (Scale factor)"
            fi

            yvariable_subscale=""
            if [ x"$teps" = xyes ]; then
                yvariable_subscale="TEPS"
            elif [ x"$ttime" = xyes ]; then
                yvariable_subscale="time"
            fi

            if [ x"$speedup" = xyes ]; then
                yvariable="Speedup (T1 / Tp)"
            elif [ x"$efficiency" = xyes ]; then
                yvariable="Efficiency (%) (idealScaling_p / scaling_p)"
            elif [ x"$teps" = xyes ]; then
                yvariable="Traversed Edges per Second (TEPS) ${yvariable_pernode}"
            elif [ x"$ttime" = xyes ]; then
                yvariable="Total time (s) ${yvariable_pernode}"
            fi

            local title="${yvariable}"
            if [ x"$isset_title" = xyes ]; then
                title="${gtitle} - ${scaling_title}"
            else
                title="${title} - ${scaling_title}"
            fi

            if [ "x$values" = "x" ]; then
                values="$token"
            else
                values="$values,$token"
            fi
            values="c($values)"

            criteria=""
            if [ x"$isset_sf" = xyes ]; then
                criteria="${sf0}"
            fi
            if [ x"$isset_np" = xyes ]; then
                criteria="${np0}"
            fi

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
            if [ "x${speedup}" = "xyes" ];then
                labelscenario="${labelscenario}, '${yvariable} (Ideal scaling)'"
            fi
            labelscenario="c(${labelscenario})"

            #
            # variable "firstscenarioplotcode" for scenario 1 (only) should be generated here
            #
        else

            # TEPS scale increases, time decreaseas (as performance increases)
            invert_scale_start=""
            invert_scale_end=""
            if [ x"$teps" = xyes ]; then
                invert_scale_start="(1 / "
                invert_scale_end=")"
            fi

            y_formula=""
            ideal_speedup=""
            ideal_speedup_plot=""
            if [ "x$efficiency" = "xyes" ]; then
            ideal_speedup="
                ideal_speedup <- 1:length(processors)
                for (i in 1:length(processors)) {
                    ideal_speedup[i] <- processors[i] / processors[1]
                }
            "
            y_formula="
                y__${scenario}_0 <- y__${scenario}[1]
                y__${scenario}_n <- y__${scenario}[length(y__${scenario})]
                processors_0 <- processors[1]
                for (i in 1:length(y__${scenario})) {
                  y__${scenario}[i] = (${invert_scale_start}  y__${scenario}_0 ${invert_scale_end}  / ${invert_scale_start} y__${scenario}[i]) ${invert_scale_end} / ideal_speedup[i] * 100
                }
                "
            elif [ "x$speedup" = "xyes" ]; then
            ideal_speedup="
                ideal_speedup <- 1:length(processors)
                for (i in 1:length(processors)) {
                    ideal_speedup[i] <- processors[i] / processors[1]
                }
            "
            ideal_speedup_plot="
            points(x, ideal_speedup, col = ('${colors[$index+1]}') , pch = c(${pointshapes[$index+1]}))
            lines(x, ideal_speedup, lty=2, col = c('${colors[$index+1]}'))
            "

            y_formula="
                y__${scenario}_0 <- y__${scenario}[1]
                y__${scenario}_n <- y__${scenario}[length(y__${scenario})]
                for (i in 1:length(y__${scenario})) {
                  y__${scenario}[i] = (${invert_scale_start}  y__${scenario}_0 ${invert_scale_end} / ${invert_scale_start} y__${scenario}[i] ${invert_scale_end})
                }
                "
            elif [ "x$weak" = "xyes" ]; then
            y_formula="
                y__${scenario}_0 <- y__${scenario}[1]
                processors_0 <- processors[1]
                for (i in 1:length(y__${scenario})) {
                  y__${scenario}[i] = y__${scenario}[i] * (processors_0 / processors[i])
                }
                "
            fi

        restofscenariosplotcodevars="
            ${restofscenariosplotcodevars}
                df__${scenario} <- data.frame('y' = vector__${criteria}__${scenario}, 'x' = seq(1:length(vector__${criteria}__${scenario})))
                x__${scenario} <- df__${scenario}\$x
                y__${scenario} <- df__${scenario}\$y

            ${y_formula}
                "

        restofscenariosplotcode="
            ${restofscenariosplotcode}
            points(x__${scenario}, y__${scenario}, col = ('${colors[$index]}') , pch = c(${pointshapes[$index]}))
            lines(x__${scenario}, y__${scenario}, col = c('${colors[$index]}'))"

    fi
    done

    processors=""
    if [ "x$speedup" = "xyes" ] || [ "x$efficiency" = "xyes" ] || [ "x$weak" = "xyes" ]; then
        processors="processors <- as.integer(vector__gpus__${scenarios[$index]})"
    fi

    y_labels=""
    if [ "x$efficiency" = "xyes" ]; then
        y_labels="
            labels_y = format(seq(from = min_y, to = max_y, by = ((max_y - min_y)/(length(y) - 1))), scientific = F, digits = 3);
            "
    elif [ "x$speedup" = "xyes" ]; then
        y_labels="
            min_y <- min(ideal_speedup)
            max_y <- max(ideal_speedup)
            labels_y = format(seq(from = min_y, to = max_y, by = ((max_y - min_y)/(length(y) - 1))), scientific = F, digits = 3);
            "
    elif [ "x$weak" = "xyes" ]; then
        y_labels="
            labels_y = format(seq(from = min_y, to = max_y, by = ((max_y - min_y)/(length(y) - 1))), scientific = T, digits = 3);
            "
    else
        y_labels="
            ${fix_maxmin_y}

            for (i in 1:length(y)) {
              labels_y[i] = format(y[i], scientific = T, digits = 3);
            }"
    fi

    calculatemaxmin_axisy=""
    if [ ${#scenarios[@]} -ge 2 ]; then
        for index in `seq 2 ${#scenarios[@]}`; do
            scenario=${scenarios[$index]}
            calculatemaxmin_axisy="
                ${calculatemaxmin_axisy}
                max_y <- max(max_y, y__${scenario}[i])
                min_y <- min(min_y, y__${scenario}[i])"
        done
    fi

    calculatemaxmin_axisy="
        max_y <-y[1]
        min_y <-y[1]
        for (i in 1:length(y)) {
            max_y <- max(max_y, y[i])
            min_y <- min(min_y, y[i])
            ${calculatemaxmin_axisy}
        }"

    fix_maxmin_y=""
    if [ x"$ttime" = xyes ]; then
        fix_maxmin_y="# max_y=max(y)
        min_y=min(y)"
    fi

    y_formula=""
    if [ "x$efficiency" = "xyes" ]; then
        y_formula="
            y_0 <- y[1]
            y_n <- y[length(y)]
            processors_0 <- processors[1]
            for (i in 1:length(y)) {
              y[i] = (${invert_scale_start} y_0 ${invert_scale_end} / ${invert_scale_start} y[i] ${invert_scale_end}) / ideal_speedup[i] * 100
            }
        "
    elif [ "x$speedup" = "xyes" ]; then
        y_formula="
            y_0 <- y[1]
            y_n <- y[length(y)]
            for (i in 1:length(y)) {
              y[i] = ${invert_scale_start} y_0 ${invert_scale_end} / ${invert_scale_start} y[i] ${invert_scale_end}
            }
        "
    elif [ "x$weak" = "xyes" ]; then
        y_formula="
            processors_0 <- processors[1]
            for (i in 1:length(y)) {
              y[i] = y[i] * (processors_0 / processors[i])
            }
            "
    fi

        firstscenarioplotcode="df <- data.frame('y' = ${token}, 'x' = seq(1:length(${token})))
            x <- df\$x
            y <- df\$y

        ${processors}
            labels_y <- seq(1, length(y))
            ${ideal_speedup}
            ${y_formula}

        ${restofscenariosplotcodevars}
            ${calculatemaxmin_axisy}
            ${y_labels}

            par(xpd=F, yaxs = 'r')
            df.bar <-  plot(x,y, axes = F, ylim=c(min_y, max_y), xaxt = 'n', yaxt = 'n', xlab='${xvariable}', ylab='${yvariable}', cex.lab = 0.9,
                            main = '${title}', type='p', lty=1, lwd=1, pch = 18, col = c('cornflowerblue'))
            lines(x, y, col = c('cornflowerblue'))
        ${restofscenariosplotcode}

        ${ideal_speedup_plot}
            axis(1, at = 1:length(x), labels = F)
            text(x = seq(1, length(x)) , par('usr')[3], labels = labels_x, srt = 360, pos = 1, xpd = T, cex = 0.75)
            axis(2, at= seq(min_y, max_y, by=(max_y-min_y)/(length(y)-1)), labels = F)
            text(y=seq(min_y, max_y, by=(max_y-min_y)/(length(y)-1)), par('usr')[1], srt = 45, labels=labels_y ,pos = 2, xpd = T, cex=0.75)


            spy=(length(y)-1)
            spx=(length(x)-1)
            grid(nx=spx,ny=spy,col=c('black'),equilogs=T)
            par(xpd=T)
        legend('top', legend = ${labelscenario}, bty= 'n', cex=0.75, col=${colormatrix}, pch=${pointmatrix})"

    write_r "$firstscenarioplotcode" $result_file
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
            if [ "x${isset_sf}" = "xyes" ]; then
                xlabels[$i]="gpus:$npss0"
            elif [ "x${isset_np}" = "xyes" ]; then
                xlabels[$i]="scale:$sfss0"
            else
                xlabels[$i]="gpus:$npss0 \\n scale:$sfss0"
            fi
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
    echo "Usage: $0 [-y] -f <file> -np <# processes> -sf <scale factor> [-t <threshold>] < -teps|-time > [-speedup|-efficiency] [-weak] [-title 'graph title'] [[-ci <compareid>][-ci <compareid>]...]"
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
scaling_string="_strongscaling"
scaling_title="Strong Scaling"
subscale_string=""

# Y-AXIS
# globals (as options will increase. not suitable passing as parameter)
teps=no
ttime=no
speedup=no
efficiency=no
weak=no # weak scaling. default: strong
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
    -teps)
    variables=$((variables+1))
    teps=yes
    variable_token="harmonic_mean_TEPS"
    option_string="_teps"
    ;;
    -time)
    variables=$((variables+1))
    ttime=yes
    variable_token="mean_time"
    option_string="_time"
    ;;
    -speedup)
    speedup=yes
    subscale_string="_speedup"
    ;;
    -efficiency)
    efficiency=yes
    subscale_string="_efficciency"
    ;;
    -weak)
    weak=yes
    scaling_string="_weaksscaling"
    scaling_title="Weak Scaling"
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

if [ $variables -ne 1 ]; then
    echo "You must provide one option. Options are: -teps -time"
    printhelp
    exit 1
fi

if [ "x$speedup" = "xyes" ] && [ "x$efficiency" = "xyes" ]; then
    echo "You must only provide one scale. Options are: -speedup -efficiency"
    printhelp
    exit 1
fi


#if [ "x$speedup" = "xyes" ]; then
#    teps=no
#    ttime=yes
#    variable_token="mean_time"
#    subscale_string="_speedup"
#fi

#if [ "x$efficiency" = "xyes" ]; then
#    teps=no
#    ttime=yes
#    variable_token="mean_time"
#    option_string="_efficciency"
#fi

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

if [ "x$speedup" = "xyes" ] && [ x$isset_np = "xyes" ]; then
    echo "-sf <scale factor> must be provided in order to use Speedup in y-axis "
    printhelp
    exit 1
fi

if [ "x$efficiency" = "xyes" ] && [ x$isset_np = "xyes" ]; then
    echo "-sf <scale factor> must be provided in order to use Parallel efficiency in y-axis "
    printhelp
    exit 1
fi

# suffix of output R file
filter_string0="${filter_string_np}${filter_string_sf}${filter_string_t}${option_string}${subscale_string}${scaling_string}${ci_string}"

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





