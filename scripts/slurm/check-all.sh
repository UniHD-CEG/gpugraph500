#!/bin/bash


function set_colors {
  colors="$1"

if [ "x$colors" = "xyes" ]; then
  fancyx='\u0058'
  fancyo='\u03BF'
  checkmark='\u221A'
  nocolor='\e[0m'
  red='\e[1;31m'
  green='\e[0;32m'
else
  fancyx='x'
  fancyo='o'
  checkmark='.'
  nocolor=''
  red=''
  green=''
fi
}

function test_result {
  local res=$1
  local error=false

  if [ $res -ne 0 ]; then
    error=true
  fi

  eval "$2='$error'"
}

function get_size {
  local file=$1

  size=`ls -l $file 2>/dev/null | awk '{ print $5 }'`
  res=$?

  test_result $res error
  if $error ; then
    size=0
  fi

  eval "$2='$size'"
}

function clean_char {
  echo -n " "
  tput cub1
}

function update_progress {
  local pos=$1
  local state=$(( $pos % 4 ))

  clean_char
  case $state in
    1)
      echo -en " [|]"
      tput cub 4
      ;;
    2)
      echo -en " [/]"
      tput cub 4
      ;;
    3)
      echo -en " [-]"
      tput cub 4
      ;;
    0)
      echo -en " [\]"
      tput cub 4
      ;;
  esac
}

function cancel_job {
  local jobid=$1
  local progress=$2

  squeue=`squeue | grep -i $jobid`
  if [ "x$squeue" != "x" ]; then
    update_progress $progress
    scancel $jobid &> /dev/null
    squeue=`squeue | grep -i $jobid`
  fi

  while [ ! "x$squeue" = "x" ]; do
    sleep 1s
    squeue=`squeue | grep -i $jobid`
    progress=$((progress + 1))
    update_progress $progress
  done
}

function wait_and_process {
  local jobid=$1
  local sf=$2
  local file="slurm-$1.out"
  local hasfinished=false
  local validation=""
  local initial_filesize=0
  local increment=$(( $sf / 5 ))
  success="no"
  local modulus=$(( 6 + $increment))

  if [ $sf -le 15 ]; then
    local modulus=2
  elif [ $sf -le 20 ]; then
    local modulus=$(($increment - 1))
  else
    local modulus=$((6 + $increment))
  fi

  echo -n " "
  tput cub1
  i=1
  while [ ! -f $file ]; do
    sleep 1s
    update_progress $i
    i=$(($i + 1))
  done

  i=1
  while ! $hasfinished ; do
    if [ $(( $i % $modulus ))  -eq 0 ]; then
        get_size $file current_filesize
        if [ $current_filesize -eq $initial_filesize ]; then
           hasfinished=true
        else
           get_size $file initial_filesize
        fi
      fi
    sleep 1s
    i=$(($i + 1))
    update_progress $i
  done
  cancel_job $jobid $i

  validation=`cat $file | grep "Validation:" | grep "passed"`
  if  [ ! "x$validation" = "x" ]; then
    success="yes"
  fi

  eval "$3='$success'"
}

function print_header {
  local scale_factors="$1"

  printf "\n"
  echo -ne "$green[ $checkmark ] = success"
  echo -ne "   $red[ $fancyx ] = error / fail"
  echo -ne "   [ $fancyo ] = error / bug$nocolor"
  printf "\n%-16s" "script / sf"
  for sf in $scale_factors; do
    if [ $sf -lt 10 ]; then
      echo -n "  $sf"
    else
      echo -n " $sf"
    fi
  done
  echo ""
}

function print_script {
  local script=`echo $1 | sed -e 's/\.rsh//'`
  printf "%-16s" "$script"
}

function iterate {
  local script="$1"
  local sf=$2
  local lock="$3"
  local error=false
  jobid_sf_stat=""

  sbatch $script $sf &> $lock
  res=$?

  test_result $res error
  jobid=`head -1 $lock | grep "Submitted batch" | sed -e 's/[^0-9]//g'`

  if $error || [ "x$jobid" = "x" ]; then
     echo -ne "  ${red}${fancyo}${nocolor}"
     jobid_sf_stat="0_${sf}__0"
  else
     wait_and_process $jobid $sf success
     clean_char
     if [ "x$success" = "xyes" ] ; then
        echo -ne "  ${green}${checkmark}${nocolor}"
        jobid_sf_stat="${jobid}_${sf}__1"
     else
        echo -ne "  ${red}${fancyx}${nocolor}"
        jobid_sf_stat="${jobid}_${sf}__0"
     fi
     clean_char
  fi
  eval "$4=$jobid_sf_stat"
}

function file_output_result_lists {
  local joid_sf_list=$1
  local sfs=$2
  local x_labels=$3
  local file=$4

  echo "$file_output_result_lists"

}

function get_status {
  stat=$1
  stat_position=$((${#stat}-1))
  stat_last_char="${stat:$stat_position:1}"

  eval "$2=$stat_last_char"
}

function main {
  local min=$1
  local max=$2
  local lock=tmp.tmp
  local scripts=`ls o*.rsh`
  local scale_factors=`seq $min $max`
  local status_array_length=0
  local status_array=()
  local rows=`echo $scripts | wc -w`
  local columns=`echo $scale_factors | wc -w`
  local tmp_jobid_list=""
  local valid_jobid_list=""
  local all_rows_valid=true
  local tmp_jobid=""

  print_header "$scale_factors"
  rm -rf $lock

  tput civis
  for script in $scripts; do
    print_script $script
    for sf in $scale_factors; do
      iterate $script $sf $lock id_sf_stat
      status_array[$status_array_length]=$id_sf_stat
      ((status_array_length+=1))
    done
    echo ""
  done
  echo ""
  tput cnorm

  local index=0
  local row=0
  local col=0
  local temp_row_list=""
  while [ $index -lt $status_array_length ]; do

    tmp_jobid=${status_array[$((col * columns + row))]}
    get_status $tmp_jobid st
    if [ $all_rows_valid ] && [ $st -eq 1 ]; then
        tmp_jobid_list+=" $tmp_jobid"
    else
        all_rows_valid=false
        tmp_jobid_list=""
    fi

    ((index+=1))
    if [ $(($index % $rows)) -eq 0 ]; then
      ((row+=1))
      col=0
      if [ $all_rows_valid ]; then
          valid_jobid_list+=" $tmp_jobid_list"
      else
          all_rows_valid=true
          tmp_jobid_list=""
      fi
    else
      ((col+=1))
    fi
  done

  file_output_result_lists "$valid_jobid_list" "a" "a" "a"

  status_array=()
  rm -rf $lock
  exit 0
}

function usage {
    echo "This script will run all tests (o*.rsh) in the 'eval/' folder for score factors {MIN..MAX}."
    echo "Usage: $0 <MIN> <MAX>"
}

number_regex='^[0-9]+$'
if [ $# -ne 2 ] || [ "x$1" = "x-h" ] || [ "x$1" = "x--help" ] || ! [[ $1 =~ $number_regex ]] || ! [[ $2 =~ $number_regex ]]; then
  usage
  exit 1
fi

if [ $1 -gt $2 ]; then
  usage
  echo ""
  echo "error: MIN should be less than or equal to MAX"
  exit 1
fi

if [ $1 -eq 0 ]; then
  usage
  echo ""
  echo "error: The value fo zero is not allowed"
  exit 1
fi

srun=`which srun`
if [ ! -x $srun ];then
  echo "error: this script requires SLURM."
  exit 1
fi

set_colors "yes"
main $1 $2
