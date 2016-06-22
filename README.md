# Table Of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
  - [downloading and decompressing](#downloading-and-decompressing)
  - [using git](#using-git)
- [Build](#build)
- [Run](#run)
  - [Run (using SLURM)](#run-using-slurm)
  - [Run (using MPI)](#run-using-mpi)
- [Profiling](#profiling)
  - [zones](#zones)
  - [system variables](#system-variables)
  - [compression benchmarking tool](#compression-benchmarking-tool)
  - [instrumentation](#instrumentation)
- [Options](#options)
  - [build options](#build-options)
  - [execution options](#execution-options)
  - [currently supported codecs in the gpugraph500 binary](#currently-supported-codecs-in-the-gpugraph500-binary)
- [Troubleshooting](#troubleshooting)
- [Future work](#future-work)
- [Author](#author)
- [License](#license)
- [Resources](#resources)


# Introduction
Recent research projects have investigated partitioning, acceleration, and data reduction techniques for improving the performance of Breadth First Search (BFS) and the related HPC benchmark, Graph500. However, few implementations have focused on cloud-based systems like Amazon's Web Services, which differ from HPC systems in several ways, most importantly in terms of network interconnect.


This work looks at optimizations to reduce the communication overhead of an accelerated, distributed BFS on an HPC system and a smaller cloud-like system that contains GPUs. We demonstrate the effects of an efficient 2D partitioning scheme and allreduce implementation, as well as different CPU-based compression schemes for reducing the overall amount of data shared between nodes. Timing and Score-P profiling results demonstrate a dramatic reduction in row and column frontier queue data (up to 91%) and show how compression can improve performance for a bandwidth-limited cluster.



# Requirements
- C compiler. C++ Compiler with c++11 support.
- A MPI implementation: OpenMPI (MPICH2 is not fully supported)
- To use CUDA-BFS or CUDA-compression: CUDA 6+ support.
- To use SIMD compression: SSE2 support (SSE4+ support recommended)
- To use SIMD+ compression SSE2 support. (Optional)
- Scalasca (Score-P) and CUBE4+ for instrumentation and profiling (Optional)
- System packages: `libtool`, `automake`

# Installation
## downloading and decompressing

```
$ wget https://github.com/UniHD-CEG/gpugraph500/archive/master.zip
$ unzip master.zip
$ cd gpugraph500-master
```

## using git

```
$ git clone https://github.com/UniHD-CEG/gpugraph500.git
$ cd gpugraph500
```

# Build
The code to compile is in the folder `cpu_2d/`. To build the binary:

First build: (or when editing `configure.ac`)

```
$ cd cpu_2d
$ ./autogen.sh [option1 option2 ...] # ./configure options: (1)
$ make
```

Consecutive builds:

```
$ cd cpu_2d
$ ./configure [option1 option2 ...]  # ./configure options (1)
$ make
```

(1) for further help check the [available options](#build-options) or run `./configure --help`

# Run

## Run (using SLURM)

```
$ cd eval/
$ sbatch o16p8n.rsh 22 # (Replace 22 with Scale Factor)
```

## Run (using MPIRUN)

```
$ cd cpu_2d/
$ mpirun -np 16 ../cpu_2d/g500 -s 22 -C 4 -gpus 1 -qs 2 -be "s4-bp128-d4" -btr 64 -btc 64
```
See a full description of the [options](#execution-options) below.

See a full description of the [available codecs](#currently-supported-codecs-in-the-gpugraph500-binary) below.


runs a test with 16 proccesses in 8 nodes, using Scale Factor 21

# Profiling
This application allows the code to be instrumented in zones using Score-P (Scalasca) with low overhead.

## zones
The names of the instrumented zones are listed below.

Zone (label)                      | Explanation
--------------------------------- | -------------------------------------------------------------:
BFSRUN_region_vertexBroadcast     |        Initial vertices broadcast (No compression)
BFSRUN_region_localExpansion      |        Predecessor List Reduction (No compression)
BFSRUN_region_columnCommunication |           Column communication phase (Compression)
BFSRUN_region_rowCommunication    |              Row communication phase (Compression)
BFSRUN_region_Compression         |      Row Compression (type convertions + compression/ encoding)
BFSRUN_region_Decompression       |    Row Decompression (type convertions + decompression/ encoding)
CPUSIMD_region_encode             |                          compression or decompression/ encoding
BFSRUN_region_vreduceCompr        |   Column Compression (type convertions + compression/ encoding)
BFSRUN_region_vreduceDecompr      | Column Decompression (type convertions + decompression/ encoding)

## system variables
The following example asumes an installation of CUBE and scalasca in `$HOME/cube` and `$HOME/scorep`

```
$ cat >> ~/.bashrc << EOF
export G500_ENABLE_RUNTIME_SCALASCA=yes

export SCOREP_CUDA_BUFFER=48M
export SCOREP_CUDA_ENABLE=no
export SCOREP_ENABLE_PROFILING=true
export SCOREP_ENABLE_TRACING=false
export SCOREP_PROFILING_FORMAT=CUBE4
export SCOREP_TOTAL_MEMORY=12M
export SCOREP_VERBOSE=no
export SCOREP_PROFILING_MAX_CALLPATH_DEPTH=330

export LD_LIBRARY_PATH=$HOME/cube/lib:$LD_LIBRARY_PATH
export PATH=$HOME/cube/bin:$PATH
export LD_LIBRARY_PATH=$HOME/scorep/lib:$LD_LIBRARY_PATH
export PATH=$HOME/scorep/bin:$PATH
export LD_LIBRARY_PATH=$HOME/scorep/lib:$LD_LIBRARY_PATH
export PATH=$HOME/scorep/bin:$PATH
EOF
```

The variable `G500_ENABLE_RUNTIME_SCALASCA` set to yes will enable the required runtime instrumentor of Scalasca.


## compression benchmarking tool
See TurboPFOR in [Resources](#resources)

## instrumentation
Results will be stored on folders with the format `scorep-*`.

Possible ways of instrumenting:

* The provided `scripts/Profiling/Statistics.sh` script. The options must be changed inside the script. Text output

* Using CUBE (text output)
```
$HOME/cube/bin/cube_stat -p -m time  -r BFSRUN_region_Compression,BFSRUN_region_Decompression,CPUSIMD_region_encode,BFSRUN_region_vreduceCompr,BFSRUN_region_vreduceDecompr profile.cubex
```
Flag -m in ´cube_stat´ may be set to: time or bytes_sent


* Using scorep-score (text ouptut)
```
$ scorep-score -r [zone1,zone2,zone3....] profile.cubex
```

See the available [zones](#zones) section, for further information.

* Using CUBE (graphical interface)
```
$ $HOME/cube/bin/cube profile.cubex
```


# Options
## currently supported codecs in the gpugraph500 binary

Lemire's SIMDCompression codecs | Notes
------------------------------- | ------------------------------------:
varintg8iu                      |
fastpfor                        |
varint                          |
vbyte                           |
maskedvbyte                     |
streamvbyte                     |
frameofreference                |
simdframeofreference            |
varintgb                        | Based on a talk by Jeff Dean (Google)
s4-fastpfor-d4                  |
s4-fastpfor-dm                  |
s4-fastpfor-d1                  |
s4-fastpfor-d2                  |
bp32                            |
ibp32                           |
s4-bp128-d1-ni                  |
s4-bp128-d2-ni                  |
s4-bp128-d4-ni                  |
s4-bp128-dm                     |                 Codec used as default
s4-bp128-d1                     |
s4-bp128-d2                     |
s4-bp128-d4                     |
for                             |                          Original FOR


## build options
```
`configure' configures gpugraph500 1.0 to adapt to many kinds of systems.

Usage: ./configure [OPTION]... [VAR=VALUE]...

To assign environment variables (e.g., CC, CFLAGS...), specify them as
VAR=VALUE.  See below for descriptions of some of the useful variables.

Defaults for the options are specified in brackets.

Configuration:
  -h, --help              display this help and exit
      --help=short        display options specific to this package
      --help=recursive    display the short help of all the included packages
  -V, --version           display version information and exit
  -q, --quiet, --silent   do not print `checking ...' messages
      --cache-file=FILE   cache test results in FILE [disabled]
  -C, --config-cache      alias for `--cache-file=config.cache'
  -n, --no-create         do not create output files
      --srcdir=DIR        find the sources in DIR [configure dir or `..']

Installation directories:
  --prefix=PREFIX         install architecture-independent files in PREFIX
                          [/usr/local]
  --exec-prefix=EPREFIX   install architecture-dependent files in EPREFIX
                          [PREFIX]

By default, `make install' will install all the files in
`/usr/local/bin', `/usr/local/lib' etc.  You can specify
an installation prefix other than `/usr/local' using `--prefix',
for instance `--prefix=$HOME'.

For better control, use the options below.

Program names:
  --program-prefix=PREFIX            prepend PREFIX to installed program names
  --program-suffix=SUFFIX            append SUFFIX to installed program names
  --program-transform-name=PROGRAM   run sed PROGRAM on installed program names

System types:
  --build=BUILD     configure for building on BUILD [guessed]
  --host=HOST       cross-compile to build programs to run on HOST [BUILD]

Optional Features:
  --disable-option-checking  ignore unrecognized --enable/--with options
  --disable-FEATURE       do not include FEATURE (same as --enable-FEATURE=no)
  --enable-FEATURE[=ARG]  include FEATURE [ARG=yes]
  --disable-dependency-tracking  speeds up one-time build
  --enable-dependency-tracking   do not reject slow dependency extractors
  --enable-bfs-basic-profiling
                          It is related with instrumentation. Displays
                          statistical data on each BFS run. (Enabled by
                          default)
  --enable-other-basic-profiling
                          It is related with instrumentation. Displays
                          gpugraph500 default statistics. (Enabled by default)
  --enable-scorep         It is related with instrumentation. Enables
                          instrumentation with Scalasca/ScoreP. ScoreP must be
                          detected by ./configure. (Disabled by default)]
  --enable-compression    It is related with data compression. Enables data
                          compression through the network. This option is
                          available only when --enable-cuda (BFS runs using
                          CUDA) is active (default). (Enabled by default)
  --enable-simd           It is related with data compression. MPI packets
                          will be sent compressed using the PFOR-delta D.
                          Lemire SIMDCompression library. It is only active if
                          --enable-compression is selected. It will be enabled
                          by default if --enable-compression is active and no
                          compression method is selected. (Enabled by default)
  --enable-simd+          It is related with data compression. MPI packets
                          will be sent compressed using a PFOR-delta improved
                          library: Turbo-PFOR. It is only active if
                          --enable-compression is selected. (Disabled by
                          default)
  --enable-simt           It is related with data compression. Use CUDA
                          implementation for data compression. Not implemented
                          yet. (Disabled by default)
  --enable-debug-compression
                          It is related with data compression. Shows
                          statistics of compression rate, time of compression,
                          codec, ETC. (Disabled by default)
  --enable-verify-compression
                          It is related with data compression. Sends both
                          compressed and decompressed data through the
                          network. Checks decompression after transmission.
                          (Disabled by default)
  --enable-aggressive-optimizations
                          It is related with optimizations. Enables aggressive
                          compiler optimizations on the compiler. (Disabled
                          by default)
  --enable-openmp         It is related with optimizations. Enables or
                          disables both --enable-cuda-openmp and
                          --enable-general-openmp. This option overrides both
                          openmp settings. (Not set by default)
  --enable-cuda-openmp    Related with optimizations. Selects whether OpenMP
                          will be enabled. This option applies to CUDA C
                          files. (Disabled by default)
  --enable-general-openmp It is related with optimizations. Selects whether
                          OpenMP will be enabled. This option applies to
                          general C and C++ files. (Disabled by default)
  --enable-cuda           Use the CUDA implementation of the BFS runs.
                          Requires NVIDIA hardware support. Enabled by default
  --enable-ptxa-optimizations
                          It is related with optimizations. Selects whether
                          CUDA assembly (PTXAS) will be optimized or not. This
                          option will only be used if --enable-cuda is present
                          (default). The default PTXAS optimization is -O3.
                          (Disabled by default)
  --enable-nvidia-architecture= fermi|kepler|auto|detect
                          Selects the NVIDIA target architecture. Requires --enable-cuda to be selected (default). Default option is 'detect'. In case detection does not succeed 'all'
                          mode is selected.
  --enable-debug          Provides extra traces at runtime. (Disabled by
                          default)
  --enable-debugging      It is related with debugging. Enables -g option on
                          compiler (debugging). (Disabled by default)
  --enable-quiet          It is related with debugging. Disable compile
                          mensages when running make. (Disabled by default)
  --enable-portable-binary
                          disable compiler optimizations that would produce
                          unportable binaries
  --enable-cc-warnings= no|minimum|yes|maximum|error
                          Turn on C compiler warnings. Default selection is
                          maximum
  --enable-iso-c          Try to warn if code is not ISO C
  --enable-cxx-warnings= no|minimum|yes|maximum|error
                          Turn on C++ compiler warnings. Default selection is
                          maximum
  --enable-iso-cxx        Try to warn if code is not ISO C++

Optional Packages:
  --with-PACKAGE[=ARG]    use PACKAGE [ARG=yes]
  --without-PACKAGE       do not use PACKAGE (same as --with-PACKAGE=no)
  --with-mpi=<path>       absolute path to the MPI root directory. It should
                          contain bin/ and include/ subdirectories.
  --with-mpicc=mpicc      name of the MPI C++ compiler to use (default mpicc)
  --with-mpicxx=mpicxx    name of the MPI C++ compiler to use (default mpicxx)
  --with-cuda=<path>      Use CUDA library. If argument is <empty> that means
                          the library is reachable with the standard search
                          path "/usr" or "/usr/local" (set as default).
                          Otherwise you give the <path> to the directory which
                          contain the library.
  --with-gcc-arch=<arch>  use architecture <arch> for gcc -march/-mtune,
                          instead of guessing
  --with-opencl=<path>    prefix to location of OpenCL include directory
                          [default=auto]
  --with-scorep=<path>    Use SCOREP profiler. If argument is <empty> that
                          means the library is reachable with the standard
                          search path (set as default). Otherwise you give the
                          <path> to the directory which contain the library.

Some influential environment variables:
  CXXFLAGS    C++ compiler flags
  CFLAGS      C compiler flags
  CXX         C++ compiler command
  LDFLAGS     linker flags, e.g. -L<lib dir> if you have libraries in a
              nonstandard directory <lib dir>
  LIBS        libraries to pass to the linker, e.g. -l<library>
  CPPFLAGS    (Objective) C/C++ preprocessor flags, e.g. -I<include dir> if
              you have headers in a nonstandard directory <include dir>
  CC          C compiler command
  CPP         C preprocessor
  DOXYGEN_PAPER_SIZE
              a4wide (default), a4, letter, legal or executive

Use these variables to override the choices made by `configure' or to help
it to find libraries and programs with nonstandard names/locations.

Report bugs to the package provider.
```

## execution options
* -s Number - (SCALE_FACTOR)
* -C Number - (2^SCALE_FACTOR) - This is also the value used in the the -np flag of `mpirun`
* -gpus Number - Number of GPUs per node. Currently, only the value 1 is fully tested.
* -qs Number - Queue size as in B40C implementation, from 1 to 2 (e.g. 1.3).
* -be "Codec" - [Codec](#currently-supported-codecs-in-the-gpugraph500-binary) used when compression is enabled (--enable-compression)
* -btc Number - Row Threshoold number: Frontier queue minimum size at which compression would start. Allows disabling compression for small queue sizes.

e.g. `g500 -s 22 -C 4 -gpus 1 -qs 1.1 -be "s4-bp128-d4" -btc 64`


# Troubleshooting
- Q: In the .out file of Slurm/ Sbatch execution I get the text:

```
S=C=A=N: Abort: No SCOREP instrumentation found in target ../cpu_2d/g500
```

- A:
The instrumentation is activated for the runtime execution (i.e: the binary is being run prefixed with scalasca).

Disable with:

```
$ export G500_ENABLE_RUNTIME_SCALASCA=no
```

# Authors

Computer Engineering Group at Ruprecht-Karls University of Heidelberg
and
School of Computer Science at Georgia Institute of Technology

# License
- Duane Merrill's BC40 (back40computing) is licenced under [Apache 2 Licence.](https://github.com/UniHD-CEG/gpugraph500/tree/master/b40c/LICENSE.TXT)
- SIMDcompressionAndIntersection is licenced under [Apache 2 Licence.](https://github.com/UniHD-CEG/gpugraph500/blob/master/cpu_2d/compression/cpusimd/LICENSE)
- Alenka GPU database engine is licensed under [Apache 2 License.](https://github.com/UniHD-CEG/gpugraph500/blob/master/cpu_2d/compression/gpusimt/cudacompress.cu)


Copyright (c) 2016, Computer Engineering Group at Ruprecht-Karls University of Heidelberg, Germany. All rights reserved. Licensed under GNU/GPL version 3 -  https://www.gnu.org/licenses/gpl-3.0

# Future work

* Embed the [compression encoding routines](https://github.com/UniHD-CEG/gpugraph500/blob/master/cpu_2d/compression/cpusimd/include/codecfactory.h#L106) in the [communication module](https://github.com/UniHD-CEG/gpugraph500/blob/master/cpu_2d/globalbfs.hh#L536) (e.g. by using lambda functions). The [current implementation](https://github.com/UniHD-CEG/gpugraph500/blob/master/cpu_2d/compression/compression.hh#L15) does not allow code inlining by the Linker (due to the use of the `virtual` keyword).

* Remove [types convertion](https://github.com/UniHD-CEG/gpugraph500/blob/master/cpu_2d/compression/cpusimd.hh#L93) in the compression calls (Using a [GPU PFOR-compression implementation?](https://github.com/UniHD-CEG/gpugraph500/blob/master/cpu_2d/compression/gpusimt/cudacompress.cu#L196)).

# Resources
[D. Lemire's SIMDCompression](https://github.com/lemire/SIMDCompressionAndIntersection)

[TurboPFOR SIMDCompression and Codec Benchmarking tool](https://github.com/powturbo/TurboPFor)

[Alenka's CUDA–PFOR Compression](https://github.com/antonmks/Alenka/blob/master/compress.cu)

