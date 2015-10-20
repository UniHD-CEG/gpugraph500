#!/bin/bash
#
# Installs score-P on $HOME.
#
#
#

# todo: add --help
# todo: add --installed-cleanall (rm -rf $HOME/cube $HOME/opari2 $HOME/openmpi $HOME/scorep)
# todo: add --temp-cleanall (rm -rf $HOME/tmp/cube $HOME/tmp/opari2 $HOME/tmp/openmpi $HOME/tmp/scorep)


#
# Banners
#
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
  sleep 3
}

#
# Enforces makedir.
#
function makedir {
  local dir="$1"
  if [ ! -d $dir ]; then
    mkdir -p $dir
    exit_error $? "Cannot create directory $dir"
  fi
}

#
# Check if file returns a correct header
#
function test_url {
  local url="$1"
  # curl -s --head $url | head -n 1 | grep "HTTP/1.[01] [23].." > /dev/null
  output="`wget -S --spider $url  2>&1 | grep 'HTTP/1.1 200 OK'`"
  if [ "x$output" = "x" ];then
    exit_error $? "Cannot access file $url . check that the url is correct."
  fi
}

#
# downloads and check file.
#
function download {
  local url="$1"
  local file="$2"
  # curl $url --output $file -o nul -#
  wget $url -O $file
  if [ ! -f $file ]; then
    exit_error 1 "Error downloading $file. Is the url valid?"
    exit 1
  fi

  header=`head -20 $file | grep -i "html"`
  if [ "x$header" != "x" ]; then
    exit_error 1 "Downloaded file $file is invalid. Is the url valid?"
    exit 1
  fi
}


#
# error output & exit
#
function exit_error {
  local error=$1
  local msg="error:: $2"
  if [ "x$msg" = "x" ]; then
    msg="Error detected. Quitting..."
  fi
  if [ $error -ne 0 ]; then
    echo ""
    echo $msg
    exit 1
  fi
}

#
# let user review error
#
function review_error {
  local error=$1
  local msg="error:: $2"

  if [ $yestoall ];then
    return
  fi
  if [ "x$msg" = "x" ]; then
    msg="Error detected. Quitting..."
  fi
  if [ $error -ne 0 ]; then
    echo ""
    echo $msg
    echo -n "Continue installation? [Y/n] "
    read yesno < /dev/tty
    if [ "x$yesno" = "xn" ] || [ "x$yesno" = "xN" ];then
      exit 1
    fi
  fi
}

#
# e.g: from "http://user:pass@www.domain.tld/path/path/path/filename-1.2.3.tar.gz?query&option=op" ---> "filename-1.2.3.tar.gz"
#
function get_filename {
  local url="$1"
  temporal_url=${url##*/}
  temporal_url=${temporal_url%%\?*}
  eval "$2='$temporal_url'"
}

#
# e.g: from "http://user:pass@www.domain.tld/path/path/path/filename-1.2.3.tar.gz?query&option=op" ---> "filename"
#
function get_shortfilename {
  local url="$1"
  temporal_url=${url##*/}
  temporal_url=${temporal_url%%\?*}
  temporal_url=${temporal_url%%\-*}
  temporal_url=${temporal_url%%\.*}
  eval "$2='$temporal_url'"
}

function add_to_path {
  dir="$1"
  export LD_LIBRARY_PATH=$dir/lib:$LD_LIBRARY_PATH
  export PATH=$dir/bin:$PATH
}

function confirm_install {
  counter=$1
  #
  # TODO:
  # if forceall
  #   eval "$2='$counter'"
  #   echo Setting up ${array_of_apps[$counter,1]} for installation ...

  echo -n "Do you want to install a local ${array_of_apps[$counter,1]}? [Y/n] "
  read yesno < /dev/tty
  if [ "x$yesno" = "xn" ] || [ "x$yesno" = "xN" ];then
    let counter--
  else
    get_shortfilename ${array_of_apps[$counter,2]} shortname
    add_to_path ${installdirectory_prefix}$shortname
  fi
  eval "$2='$counter'"
}


function install {
  local banner="$1"
  local url="$2"
  local configureparams="$3"
  get_filename $url filename # filename.tar.gz
  get_shortfilename $url shortname # filename
  installdirectory=${installdirectory_prefix}${shortname} #~/filename
  banner $banner
  # re-download if neccesary
  if [ ! -f ${temporaldirectory_prefix}${filename} ]; then
    section_banner "Downloading ${filename} to ${temporaldirectory_prefix}${filename}"
    test_url $url
    exit_error $? "url ($url) is invalid."
    download $url "${temporaldirectory_prefix}${filename}"
  fi
  # decompress and clean previous installation if neccesary
  configfile="`ls ${temporaldirectory_prefix}${shortname}/configure 2> /dev/null`"
  if [ ! -d ${temporaldirectory_prefix}${shortname} ] || [ "x$configfile" = "x" ]; then
    section_banner "Decompressing ${filename} to ${temporaldirectory_prefix}${shortname}"
    if [ ! -d ${temporaldirectory_prefix}${shortname} ]; then
      # directory to be deleted is inside /home/. Safety.
      if [ "x`echo ${temporaldirectory_prefix}${shortname}| grep -i "/home/"`" != "x" ]; then
      section_banner "Deleting previous directory ${temporaldirectory_prefix}${shortname}"
        rm -rf ${temporaldirectory_prefix}${shortname} 2> /dev/null
      fi
      makedir ${temporaldirectory_prefix}${shortname}
    fi
    tar -xzvf ${temporaldirectory_prefix}${filename} -C ${temporaldirectory_prefix}${shortname} --strip-components 1 2> /dev/null
    exit_error $? "Error decompressing file ${temporaldirectory_prefix}${filename} to ${temporaldirectory_prefix}${shortname}"
  else
    cd ${temporaldirectory_prefix}${shortname}
    section_banner "Un-Installing ${temporaldirectory_prefix}${shortname}"
    make -j4 uninstall 2> /dev/null
    section_banner "Cleaning ${temporaldirectory_prefix}${shortname}"
    make -j4 clean 2> /dev/null
    cd ..
  fi
  # Installation: ./configure; make; make install
  makedir ${installdirectory}
  cd ${temporaldirectory_prefix}${shortname}
  section_banner "Configuring ${temporaldirectory_prefix}${shortname}"
  if [ ! -f "${temporaldirectory_prefix}${shortname}/configure" ]; then
    exit_error $? "Error in installed application. Consider deleting ${temporaldirectory_prefix}${shortname} and reinstall."
  else
    ./configure --prefix=${installdirectory} "${configureparams}"
    exit_error $? "Error running ./config script."
  fi
  section_banner "Making ${temporaldirectory_prefix}${shortname}"
  make -j4
  # review_error exit_error $? "Error making application $banner."
  section_banner "Installing ${temporaldirectory_prefix}${shortname}"
  make -j4 install
  review_error $? "Error installing application $banner. Try removing temporal directory (rm -rf ${temporaldirectory_prefix}${shortname})"
}

#
# check available cuda libraries
#
function get_cudaconfig {
  cudas=`locate bin/nvcc | grep bin/nvcc$`
  number='^[0-9]+$'
  if [ `echo $cudas | wc -l` -eq 0 ]; then
    exit_error 1 "CUDA must be installed."
  fi
  declare -A cuda_array
  let i=0
  echo ""
  for cuda in $cudas; do
    cuda_trimmed=`echo $cuda | sed 's,/bin/nvcc$,,'`
    echo "[$i] $cuda_trimmed"
    cuda_array[$i]=$cuda_trimmed
    let i++
  done
  echo -n "Select a CUDA library: "
  read cudanum < /dev/tty
  if [[ $cudanum =~ $number ]] && [ $cudanum -lt $i ]; then
    cuda_dir=${cuda_array[$cudanum]}
  else
    exit_error 1 "The selected CUDA libary is not correct. ${cuda_array[$cudanum]}"
  fi
  if [ ! -d ${cuda_dir}/include ]; then
    exit_error 1 "CUDA includes directory (${cuda_dir}/include) must exist."
  fi
  if [ ! -d ${cuda_dir}/lib ]; then
    exit_error 1 "CUDA lib directory (${cuda_dir}/lib) must exist."
  fi
  echo ""
  echo "Using CUDA:: $cuda_dir"
  add_to_path $cuda_dir
  if [ ! -d $temporaldirectory_prefix ]; then
    makedir $temporaldirectory_prefix
  fi
  eval "$1='$cuda_dir'"
}

#
# run scripts for selected apps
#
function do_iterate {
  for i in `seq 1 $number_of_apps`; do
    install "${array_of_apps[$i,1]}" "${array_of_apps[$i,2]}" "${array_of_apps[$i,3]}"
  done

  section_banner "Installation summary"
  for i in `seq 1 $number_of_apps`; do
    get_shortfilename ${array_of_apps[$i,2]} shortname
    echo "==> Installed ${array_of_apps[$i,1]} in directory [${installdirectory_prefix}${shortname}]"
    # todo: print "export LD_LIBRARY=/app/lib:$LD_LIBRARY ... ETC" & PATH=/app/bin:$PATH ETC
  done

  for i in `seq 1 $number_of_apps`; do
    echo -n ""
    # todo: print "export LD_LIBRARY=/app/lib:$LD_LIBRARY ... ETC" & PATH=/app/bin:$PATH ETC
  done
  exit $?
}

function usage {
    echo "This script will install 3rd-party software locally."
    echo "Usage: $0 [--force]"
}


declare -A array_of_apps
number_of_apps=0

usage
get_cudaconfig cuda_dir

#
# Change as desired.
#

temporaldirectory_prefix="$HOME/tmp/"
installdirectory_prefix="$HOME/"


let number_of_apps++
# no dependencies
array_of_apps[$number_of_apps,1]="OpenMPI" # Name of application
array_of_apps[$number_of_apps,2]="http://www.open-mpi.de/software/ompi/v1.10/downloads/openmpi-1.10.0.tar.gz" # Download url
array_of_apps[$number_of_apps,3]="--enable-mpirun-prefix-by-default" # ./config script's parameters # CC=$cc CXX=$cxx
confirm_install $number_of_apps number_of_apps

let number_of_apps++
# no dependencies
array_of_apps[$number_of_apps,1]="Opari" # Name of application
array_of_apps[$number_of_apps,2]="http://www.vi-hps.org/upload/packages/opari2/opari2-1.1.2.tar.gz" # Download url
array_of_apps[$number_of_apps,3]="" # ./config script's parameters # CC=$cc CXX=$cxx
get_shortfilename ${array_of_apps[$number_of_apps,2]} shortname
opari_installdirectory=${installdirectory_prefix}$shortname
confirm_install $number_of_apps number_of_apps

let number_of_apps++
# no dependencies
array_of_apps[$number_of_apps,1]="Cube" # Name of application
array_of_apps[$number_of_apps,2]="http://apps.fz-juelich.de/scalasca/releases/cube/4.3/dist/cube-4.3.2.tar.gz" # Download url
array_of_apps[$number_of_apps,3]="--without-gui" # ./config script's parameters
get_shortfilename ${array_of_apps[$number_of_apps,2]} shortname
cube_installdirectory=${installdirectory_prefix}$shortname
confirm_install $number_of_apps number_of_apps

let number_of_apps++
# depends on Opari and Cube
array_of_apps[$number_of_apps,1]="Score-P" # Name of application
array_of_apps[$number_of_apps,2]="http://www.vi-hps.org/upload/packages/scorep/scorep-1.4.1.tar.gz" # Download url
array_of_apps[$number_of_apps,3]="--with-cube=${cube_installdirectory}/bin --with-opari2=${opari_installdirectory}/bin --with-libcudart=$cuda_dir --enable-mpi --enable-cuda" # ./config script's parameters
get_shortfilename ${array_of_apps[$number_of_apps,2]} shortname
scorep_installdirectory=${installdirectory_prefix}$shortname
confirm_install $number_of_apps number_of_apps

let number_of_apps++
# depends on ScoreP
array_of_apps[$number_of_apps,1]="Scalasca" # Name of application
array_of_apps[$number_of_apps,2]="http://apps.fz-juelich.de/scalasca/releases/scalasca/2.2/dist/scalasca-2.2.2.tar.gz" # Download url
array_of_apps[$number_of_apps,3]="--with-cube=${cube_installdirectory}/bin --with-mpi=openmpi --with-otf2=${scorep_installdirectory}/bin" # ./config script's parameters
confirm_install $number_of_apps number_of_apps



if [ "x$1" = "x--force" ];then
  yestoall=true
else
  yestoall=false
fi

do_iterate
