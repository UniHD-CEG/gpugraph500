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

function makedir {
  local dir="$1"
  if [ ! -d $dir ]; then
    mkdir -p $dir
    res=$?
    exit_error $res
  fi
}

function install {

  export LD_LIBRARY_PATH=$openmpi_prefix/lib:$LD_LIBRARY_PATH
  export PATH=$openmpi_prefix/bin:$PATH
  export OMPI_CC=$cc
  export OMPI_CXX=$cxx

  if [ 1 -eq 2]; then
  banner "OpenMPI"
  if [ ! -f ${openmpi}.tar.gz ]; then
    section_banner "Downloading"
    wget http://www.open-mpi.org/software/ompi/v1.8/downloads/${openmpi}.tar.gz
    res=$?
    exit_error $res
  fi
  if [ ! -d ${openmpi} ]; then
    section_banner "Decompressing"
    tar -xzvf ${openmpi}.tar.gz
  else
    cd ${openmpi}
    section_banner "Un-Installing"
    make uninstall 2> /dev/null
    section_banner "Cleaning"
    make clean 2> /dev/null
    cd ..
  fi
  makedir $openmpi_prefix
  cd ${openmpi}
  section_banner "Checking"
  # --with-cuda=$cuda_dir
  ./configure CC=$cc CXX=$cxx  --prefix=$openmpi_prefix --disable-mpi-f90 --enable-mpirun-prefix-by-default
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
  cd ..

  banner "Opari"
  if [ ! -f  ${opari}.tar.gz ]; then
    section_banner "Downloading"
    wget http://www.vi-hps.org/upload/packages/opari2/${opari}.tar.gz
    res=$?
    exit_error $res
  fi
  if [ ! -d  ${opari} ]; then
    section_banner "Decompressing"
    tar -xzvf ${opari}.tar.gz
  else
    cd ${opari}
    section_banner "Un-Installing"
    make uninstall 2> /dev/null
    section_banner "Cleaning"
    make clean 2> /dev/null
    cd ..
  fi
  makedir $opari_prefix
  cd ${opari}
  section_banner "Checking"
  ./configure CC=$cc CXX=$cxx  --prefix=$opari_prefix
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
  cd ..

  banner "Cube"
  if [ ! -f  ${cube}.tar.gz ]; then
    section_banner "Downloading"
    wget http://apps.fz-juelich.de/scalasca/releases/cube/4.2/dist/${cube}.tar.gz
    res=$?
    exit_error $res
  fi
  if [ ! -d  ${cube} ]; then
    section_banner "Decompressing"
    tar -xzvf ${cube}.tar.gz
  else
    cd ${cube}
    section_banner "Un-Installing"
    make uninstall 2> /dev/null
    section_banner "Cleaning"
    make clean 2> /dev/null
    cd ..
  fi
  makedir $cube_prefix
  cd ${cube}
  section_banner "Checking"
  ./configure CC=$cc CXX=$cxx  --prefix=$cube_prefix --without-gui
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
  cd ..
  fi
  banner "Score-P"
  if [ ! -f  ${scorep}.tar.gz ]; then
    section_banner "Downloading"
  	wget http://www.vi-hps.org/upload/packages/scorep/${scorep}.tar.gz
    res=$?
    exit_error $res
  fi
  if [ ! -d  ${scorep} ]; then
    section_banner "Decompressing"
    tar -xzvf ${scorep}.tar.gz
  else
    cd ${scorep}
    section_banner "Un-Installing"
    make uninstall 2> /dev/null
    section_banner "Cleaning"
    make clean 2> /dev/null
    cd ..
  fi
  makedir $scorep_prefix
  cd ${scorep}
  section_banner "Checking"
  ./configure CC=$cc CXX=$cxx --prefix=$scorep_prefix --with-cube=$cube_prefix --with-opari2=$opari_prefix --with-cuda=$cuda_dir
  res=$?
  exit_error $res
  section_banner "Making"
  make clean
  make -j12
  res=$?
  exit_error $res
  section_banner "Installing"
  make install
  return $?
}

cxx=`locate bin/g++- | grep "bin/g++-[0-9]" | tail -1`
cc=`locate bin/gcc- | grep "bin/gcc-[0-9]" | tail -1`
nvcc=`locate bin/nvcc | grep bin/nvcc$ | tail -1`
cuda_dir=`echo $nvcc | sed 's,/bin/nvcc$,,'`

openmpi="openmpi-1.6.5"
opari="opari2-1.1.2"
cube="cube-4.2.3"
scorep="scorep-1.4.2"

openmpi_prefix="$HOME/openmpi"
opari_prefix="$HOME/opari2"
cube_prefix="$HOME/cube"
scorep_prefix="$HOME/scorep"


install
exit $?

