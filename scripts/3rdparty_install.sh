#!/bin/bash
#
# Installs score-P on $HOME.
#
#
#


# Configuration
#

temporaldirectory_prefix="$HOME/tmp/"
installdirectory_prefix="$HOME/"

#
# End Configuration




#
# error checking
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
  curl -s --head $url | head -n 1 | grep "HTTP/1.[01] [23].." > /dev/null
  exit_error $? "Cannot access file $url . check that the url is correct."
}

#
# downloads and check file.
#
function download {
  local url="$1"
  local file="$2"
  curl $url --output $file # --silent
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
# e.g: from "http://user:pass@www.domain.tld/path/path/path/filename-1.2.3.tar.gz?query&option=op" returns "filename-1.2.3.tar.gz"
#
function get_filename {
  local url="$1"
  temporal_url=${url##*/}
  temporal_url=${temporal_url%%\?*}
  eval "$2='$temporal_url'"
}

#
# e.g: from "http://user:pass@www.domain.tld/path/path/path/filename-1.2.3.tar.gz?query&option=op" returns "filename"
#
function get_shortfilename {
  local url="$1"
  temporal_url=${url##*/}
  temporal_url=${temporal_url%%\?*}
  temporal_url=${temporal_url%%\-*}
  temporal_url=${temporal_url%%\.*}
  eval "$2='$temporal_url'"
}


function install {
  local banner="$1"
  local url="$2"
  local configureparams="$3"
  get_filename $url filename # filename.tar.gz
  get_shortfilename $url shortname # filename
  installdirectory=${installdirectory_prefix}${shortname} #~/filename
  banner $banner
  if [ ! -f ${temporaldirectory_prefix}${filename} ]; then
    section_banner "Downloading ${filename} to ${temporaldirectory_prefix}${shortname}"
    test_url $url
    exit_error $? "url ($url) is invalid."
    download $url "${temporaldirectory_prefix}${shortname}"
  fi
  if [ ! -d ${temporaldirectory_prefix}${shortname} ]; then
    section_banner "Decompressing ${filename} to ${temporaldirectory_prefix}${shortname}"
    if [ ! -d ${temporaldirectory_prefix}${shortname} ]; then
      # directory to be deleted is inside /home/
      if [ "x`echo ${temporaldirectory_prefix}${shortname}| grep -i "/home/"`" != "x" ]; then
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
  makedir ${installdirectory}
  cd ${temporaldirectory_prefix}${shortname}
  section_banner "Configuring ${temporaldirectory_prefix}${shortname}"
  if [ ! -f "${temporaldirectory_prefix}${shortname}/configure" ]; then
    exit_error $? "Error in installed application. Consider deleting ${temporaldirectory_prefix}${shortname} and reinstall."
  else
    ./configure --prefix=${installdirectory} $configureparams
    exit_error $? "Error running ./config script."
  fi
  section_banner "Making ${temporaldirectory_prefix}${shortname}"
  make -j4
  # exit_error $?
  section_banner "Installing ${temporaldirectory_prefix}${shortname}"
  make -j4 install
  exit_error $? "Error installing application."
}




#
# Find binaries. used in Creek cluster.
#
cxx=`locate bin/g++- | grep "bin/g++-[0-9]" | tail -1`
cc=`locate bin/gcc- | grep "bin/gcc-[0-9]" | tail -1`
nvcc=`locate bin/nvcc | grep bin/nvcc$ | tail -1`
cuda_dir=`echo $nvcc | sed 's,/bin/nvcc$,,'`

export LD_LIBRARY_PATH=$openmpi_installdirectory/lib:$LD_LIBRARY_PATH
export PATH=$openmpi_installdirectory/bin:$PATH
export OMPI_CC=$cc
export OMPI_CXX=$cxx




if [ ! -d $temporaldirectory_prefix ]; then
  makedir $temporaldirectory_prefix
fi
declare -A array_of_apps
number_of_apps=0

let number_of_apps++
array_of_apps[$number_of_apps,1]="OpenMPI" # Name of application
array_of_apps[$number_of_apps,2]="http://www.open-mpi.de/software/ompi/v1.10/downloads/openmpi-1.10.0.tar.gz" # Download url
array_of_apps[$number_of_apps,3]="CC=$cc CXX=$cxx --enable-mpirun-prefix-by-default" # ./config script's parameters

let number_of_apps++
array_of_apps[$number_of_apps,1]="Opari" # Name of application
array_of_apps[$number_of_apps,2]="http://www.vi-hps.org/upload/packages/opari2/opari2-1.1.2.tar.gz" # Download url
array_of_apps[$number_of_apps,3]="CC=$cc CXX=$cxx" # ./config script's parameters
get_shortfilename ${array_of_apps[$number_of_apps,2]} shortname
opari_installdirectory=${installdirectory_prefix}$shortname

let number_of_apps++
array_of_apps[$number_of_apps,1]="Cube" # Name of application
array_of_apps[$number_of_apps,2]="http://apps.fz-juelich.de/scalasca/releases/cube/4.3/dist/cube-4.3.2.tar.gz" # Download url
array_of_apps[$number_of_apps,3]="--without-gui" # ./config script's parameters
get_shortfilename ${array_of_apps[$number_of_apps,2]} shortname
cube_installdirectory=${installdirectory_prefix}$shortname

let number_of_apps++
array_of_apps[$number_of_apps,1]="Score-P" # Name of application
array_of_apps[$number_of_apps,2]="http://www.vi-hps.org/upload/packages/scorep/scorep-1.4.1.tar.gz" # Download url
array_of_apps[$number_of_apps,3]="--with-cube=$cube_installdirectory --with-opari2=$opari_installdirectory --with-libcudart=$cuda_dir --enable-mpi --enable-cuda" # ./config script's parameters
get_shortfilename ${array_of_apps[$number_of_apps,2]} shortname
scorep_installdirectory=${installdirectory_prefix}$shortname

let number_of_apps++
array_of_apps[$number_of_apps,1]="Scalasca" # Name of application
array_of_apps[$number_of_apps,2]="http://apps.fz-juelich.de/scalasca/releases/scalasca/2.2/dist/scalasca-2.2.2.tar.gz" # Download url
array_of_apps[$number_of_apps,3]="--with-cube=${cube_installdirectory}/bin --with-mpi=openmpi --with-otf2=${scorep_installdirectory}/bin" # ./config script's parameters



for i in `seq 1 $number_of_apps`; do
  install "${array_of_apps[$i,1]}" "${array_of_apps[$i,2]}" "${array_of_apps[$i,3]}"
done

section_banner "Installation summary"
for i in `seq 1 $number_of_apps`; do
  get_shortfilename ${array_of_apps[$number_of_apps,2]} shortname
  echo "==> Installed ${array_of_apps[$i,1]} in directory [${installdirectory_prefix}${shortname}]"
done
exit $?
