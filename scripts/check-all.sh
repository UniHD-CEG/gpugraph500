#!/bin/bash


function test_result {
  local res=$1 
  local error=false
  if [ $res -ne 0 ]; then
    error=true
  fi
  eval "$2=$error"
}

function get_size {
  local file=$1
 
  size=`ls -l $file 2>/dev/null | awk '{ print $5 }'`
  eval "$2=$size"
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

  clean_char
  update_progress $progress
  scancel $jobid &> /dev/null
  squeue=`squeue | grep -i $jobid`

  while [ ! "x$squeue" = "x" ]; do
    sleep 1s
    squeue=`squeue | grep -i $jobid`
    let "progress+=1"
    update_progress $progress
  done
}

function wait_and_process {
  local jobid=$1
  local file="slurm-$1.out"
  local hasfinished=false
  local validation=""

  echo -n " "
  tput cub1    
  sleep 1s
  if [ ! -f $file ]; then
    hasfinished=true
  fi
  
  get_size $file initial_filesize
  i=1
  update_progress $i
  while ! $hasfinished ; do
    if [ $(( $i % 3 ))  -eq 0 ]; then
      get_size $file current_filesize 
      if [ $current_filesize -eq $initial_filesize ];then
         hasfinished=true
       else
         get_size $file initial_filesize
       fi 
    fi
    sleep 1s
    let "i+=1"
    update_progress $i
  done
 
  if [ -f $file ];then 
    validation=`grep "Validation:" $file | grep "passed"`
  fi 

  if  [ ! "x$validation" = "x" ]; then
    success=true
  else 
    success=false
  fi
  
  cancel_job $jobid $i
  eval "$2=$success"
}

function print_header {
  local scale_factors="$1"
  
  echo ""; echo -e "${green}[ . ] = success   ${red}[ x ] = error${nocolor}"
  printf "%-20s" "script / sf"
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
  printf "%-20s" "$script"
}

function iterate {
  local script="$1"
  local sf=$2
  local lock="$3"
  local error=false
  
  sbatch $script $sf &> $lock
  res=$?
  test_result $res error 
  jobid=`head -1 $lock | grep "Submitted batch" | sed -e 's/[^0-9]//g'`

  if $error || [ "x$jobid" = "x" ]; then
     echo -ne "  ${red}x${nocolor}"
  else 
     wait_and_process $jobid success
     clean_char
     if $success ; then
        echo -ne "  ${green}.${nocolor}"
     else
        echo -ne "  ${red}x${nocolor}"
     fi
     clean_char
  fi
}

function main {
  local min=$1
  local max=$2
  local lock=tmp.tmp
  local scripts=`ls o*.rsh`
  local scale_factors=`seq $min $max`

  print_header "$scale_factors"
  rm -rf $lock
  tput civis
  for script in $scripts; do
    print_script $script 
    for sf in $scale_factors; do
      iterate $script $sf $lock
    done
    echo ""
  done
  echo "" 
  tput cnorm 
  
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

red='\e[0;31m'     
green='\e[0;32m'
nocolor='\e[0m'
main $1 $2
