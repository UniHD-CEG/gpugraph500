#!/bin/bash

# Installs score-P on $HOME.


function exit_error {
  local error=$1
  if [ $error -neq 0 ]; then
    echo "Error detected. Quitting..."
    exit 1
 fi
}

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

function install {

  export LD_LIBRARY_PATH=$apps/openmpi/lib64:$LD_LIBRARY_PATH
  export PATH=$apps/openmpi/bin:$PATH
  export OMPI_CC=$cc
  export OMPI_CXX=$cxx

  mkdir -p $apps &> /dev/null

  banner "OpenMPI"
  if [ ! -f ${openmpi}.tar.gz ]; then
    section_banner "Downloading"
    wget http://www.open-mpi.org/software/ompi/v1.8/downloads/${openmpi}.tar.gz
    res=$?
    exit_error $res
  fi
  if [ ! -d ${openmpi} ]; then
    section_banner "Decompressing"
    rm -rf ${openmpi}
    tar -xzvf ${openmpi}.tar.gz
  fi
  cd ${openmpi}
  section_banner "Checking"
  ./configure CC=$cc CXX=$cxx --prefix=$apps --with-cuda=$cuda_dir
  res=$?
  exit_error $res
  section_banner "Making"
  make clean
  make -j 12
  res=$?
  exit_error $res
  section_banner "Installing"
  make install
  res=$?
  exit_error $res
  return $res
}

cxx=`locate bin/g++- | grep "bin/g++-[0-9]" | tail -1`
cc=`locate bin/gcc- | grep "bin/gcc-[0-9]" | tail -1`
nvcc=`locate bin/nvcc | grep bin/nvcc$$ | tail -1`
cuda_dir=`echo $nvcc | sed 's,/bin/nvcc$$,,'`
apps="$HOMEi/openmpi"
openmpi="openmpi-1.6.5"

install
exit $?


