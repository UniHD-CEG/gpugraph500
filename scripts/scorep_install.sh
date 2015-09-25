#!/bin/bash

# Installs score-P on $HOME.


function exit_error {
  local error=$1
  if [ $error -ne 0 ]; then
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
    exit_error $?
  fi
}

function install {

  export LD_LIBRARY_PATH=$openmpi_prefix/lib:$LD_LIBRARY_PATH
  export PATH=$openmpi_prefix/bin:$PATH
  export OMPI_CC=$cc
  export OMPI_CXX=$cxx

  banner "OpenMPI"
  if [ ! -f ${openmpi}.tar.gz ]; then
    section_banner "Downloading"
    wget http://www.open-mpi.org/software/ompi/v1.8/downloads/${openmpi}.tar.gz
    exit_error $?
  fi
  if [ ! -d ${openmpi} ]; then
    section_banner "Decompressing"
    tar -xzvf ${openmpi}.tar.gz
  else
    cd ${openmpi}
    section_banner "Un-Installing"
    make -j4 uninstall 2> /dev/null
    section_banner "Cleaning"
    make -j4 clean 2> /dev/null
    cd ..
  fi
  makedir $openmpi_prefix
  cd ${openmpi}
  section_banner "Checking"
  # --with-cuda=$cuda_dir
  ./configure CC=$cc CXX=$cxx  --prefix=$openmpi_prefix --enable-mpirun-prefix-by-default
  exit_error $?
  section_banner "Making"
  make -j4
  # exit_error $?
  section_banner "Installing"
  make -j4 install
  exit_error $?
  cd ..

  banner "Opari"
  if [ ! -f  ${opari}.tar.gz ]; then
    section_banner "Downloading"
    wget http://www.vi-hps.org/upload/packages/opari2/${opari}.tar.gz
    exit_error $?
  fi
  if [ ! -d  ${opari} ]; then
    section_banner "Decompressing"
    tar -xzvf ${opari}.tar.gz
  else
    cd ${opari}
    section_banner "Un-Installing"
    make -j4 uninstall 2> /dev/null
    section_banner "Cleaning"
    make -j4 clean 2> /dev/null
    cd ..
  fi
  makedir $opari_prefix
  cd ${opari}
  section_banner "Checking"
  ./configure CC=$cc CXX=$cxx  --prefix=$opari_prefix
  exit_error $?
  section_banner "Making"
  make -j4
  # exit_error $?
  section_banner "Installing"
  make -j4 install
  exit_error $?
  cd ..

  banner "TAU"
  if [ ! -f  ${tau}.tar.gz ]; then
    section_banner "Downloading"
    wget https://www.cs.uoregon.edu/research/tau/tau_releases/${tau}.tar.gz
    exit_error $?
  fi
  if [ ! -d  ${tau} ]; then
    section_banner "Decompressing"
    tar -xzvf ${tau}.tar.gz
  else
    cd ${tau}
    section_banner "Un-Installing"
    make -j4 uninstall 2> /dev/null
    section_banner "Cleaning"
    make -j4 clean 2> /dev/null
    cd ..
  fi
  makedir $tau_prefix
  cd $tau
  section_banner "Checking"
  ./configure -c++=g++ -cc=gcc -prefix=$tau_prefix -cuda=$cuda_dir
  exit_error $?
  section_banner "Making & Installing"
  make -j4 install
  exit_error $?
  cd ..

  banner "Cube"
  if [ ! -f  ${cube}.tar.gz ]; then
    section_banner "Downloading"
    wget http://apps.fz-juelich.de/scalasca/releases/cube/4.3/dist/${cube}.tar.gz
    exit_error $?
  fi
  if [ ! -d  ${cube} ]; then
    section_banner "Decompressing"
    tar -xzvf ${cube}.tar.gz
  else
    cd ${cube}
    section_banner "Un-Installing"
    make -j4 uninstall 2> /dev/null
    section_banner "Cleaning"
    make -j4 clean 2> /dev/null
    cd ..
  fi
  makedir $cube_prefix
  cd ${cube}
  section_banner "Checking"
  ./configure --prefix=$cube_prefix #--without-gui
  exit_error $?
  section_banner "Making"
  make -j4
  #exit_error $?
  section_banner "Installing"
  make -j4 install
  exit_error $?
  cd ..


  banner "Score-P"
  if [ ! -f  ${scorep}.tar.gz ]; then
    section_banner "Downloading"
    wget http://www.vi-hps.org/upload/packages/scorep/${scorep}.tar.gz
    exit_error $?
  fi
  if [ ! -d  ${scorep} ]; then
    section_banner "Decompressing"
    tar -xzvf ${scorep}.tar.gz
  else
    cd ${scorep}
    section_banner "Un-Installing"
    make -j4 uninstall 2> /dev/null
    section_banner "Cleaning"
    make -j4 clean 2> /dev/null
    cd ..
  fi
  makedir $scorep_prefix
  cd ${scorep}
  section_banner "Checking"
  ./configure   --prefix=$scorep_prefix --with-cube=$cube_prefix \
                --with-opari2=$opari_prefix  \
                --with-libcudart=$cuda_dir --enable-mpi --enable-cuda
  exit_error $?
  section_banner "Making"
  make -j4
  #exit_error $?
  section_banner "Installing"
  make -j4  install


  banner "Scalasca"
  if [ ! -f  ${scalasca}.tar.gz ]; then
    section_banner "Downloading"
    wget http://apps.fz-juelich.de/scalasca/releases/scalasca/2.2/dist/${scalasca}.tar.gz
    exit_error $?
  fi
  if [ ! -d  ${scalasca} ]; then
    section_banner "Decompressing"
    tar -xzvf ${}.tar.gz
  else
    cd ${scalasca}
    section_banner "Un-Installing"
    make -j4 uninstall 2> /dev/null
    section_banner "Cleaning"
    make -j4 clean 2> /dev/null
    cd ..
  fi
  makedir $scalasca_prefix
  cd ${scalasca}
  section_banner "Checking"
  ./configure   --prefix=$scalasca_prefix --with-cube=${cube_prefix}/bin \
                --with-mpi=openmpi --with-otf2=${scorep_prefix}/bin
  exit_error $?
  section_banner "Making"
  make -j4
  #exit_error $?
  section_banner "Installing"
  make -j4  install
  return $?
}

cxx=`locate bin/g++- | grep "bin/g++-[0-9]" | tail -1`
cc=`locate bin/gcc- | grep "bin/gcc-[0-9]" | tail -1`
nvcc=`locate bin/nvcc | grep bin/nvcc$ | tail -1`
cuda_dir=`echo $nvcc | sed 's,/bin/nvcc$,,'`

# these versions are compatible.
# Ubuntu 12. Aug 2015
# openmpi="openmpi-1.6.5"
# opari="opari2-1.1.2"
# cube="cube-4.2.3"
# scorep="scorep-1.3"
# tau="tau-2.24"

openmpi="openmpi-1.6.5"
opari="opari2-1.1.2"
cube="cube-4.3.2"
scorep="scorep-1.4.1"
tau="tau-2.24"
scalasca="scalasca-2.2.2"


openmpi_prefix="$HOME/openmpi"
opari_prefix="$HOME/opari2"
cube_prefix="$HOME/cube"
scorep_prefix="$HOME/scorep"
tau_prefix="$HOME/tau"
scalasca_prefix="$HOME/scalasca"

install
exit $?
