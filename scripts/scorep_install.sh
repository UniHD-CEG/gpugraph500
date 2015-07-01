#!/bin/bash -x

# Installs score-P on $HOME. 

export apps=$HOME/distlibs


function exit_error {

  if [ $1 -neq 0 ]; then
	echo "Error detected. Quitting..."
	exit 1
  fi

}

function install {

  if [ ! -f openmpi-1.8.3.tar.gz ]; then
  	wget http://www.open-mpi.org/software/ompi/v1.8/downloads/openmpi-1.8.3.tar.gz
  fi
  rm -rf openmpi-1.8.3
  tar -xzvf openmpi-1.8.3.tar.gz
  cd openmpi-1.8.3
  ./configure --prefix=$apps/openmpi --with-cuda=/usr/local/cuda
  res=$?
  exit_error $res
  
  make clean 
  make -j 12
  res=$?
  exit_error $res

  make install
  res=$?
  exit_error $res

  export PATH=$apps/openmpi/bin:$PATH
  export LD_LIBRARY_PATH=$apps/openmpi/lib64:$LD_LIBRARY_PATH
  cd ..

  if [ ! -f  opari2-1.1.2.tar.gz ]; then
  	wget http://www.vi-hps.org/upload/packages/opari2/opari2-1.1.2.tar.gz
  fi
  rm -rf opari2-1.1.2
  tar -xzvf opari2-1.1.2.tar.gz
  cd opari2-1.1.2
  ./configure --prefix=$apps/opari2
  res=$?
  exit_error $res

  make clean
  make -j 12
  res=$?
  exit_error $res

  make install
  res=$?
  exit_error $res
  cd ..

  if [ ! -f  cube-4.2.3.tar.gz ]; then
	wget http://apps.fz-juelich.de/scalasca/releases/cube/4.2/dist/cube-4.2.3.tar.gz
  fi
  rm -rf cube-4.2.3
  tar -xzvf cube-4.2.3.tar.gz
  cd cube-4.2.3
  ./configure --prefix=$apps/cube --without-gui
  res=$?
  exit_error $res

  make clean
  make -j 12
  res=$?
  exit_error $res
  make install
  res=$?
  exit_error $res
  cd ..

  if [ ! -f  scorep-1.3.tar.gz ]; then
  	wget http://www.vi-hps.org/upload/packages/scorep/scorep-1.3.tar.gz
  fi
  rm -rf scorep-1.3
  tar -xzvf scorep-1.3.tar.gz
  cd scorep-1.3
  ./configure --prefix=$apps/score_p --with-cube=$apps/cube --with-opari2=$apps/opari2 --with-cuda=/usr/local/cuda
  res=$?
  exit_error $res

  make clean
  make -j 12
  res=$?
  exit_error $res

  make install
}

install
exit 0


