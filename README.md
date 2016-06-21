# Table Of Contents
- [Requirements](#requirements)
- [Installation](#installation)
  - [downloading and decompressing:](#downloading-and-decompressing)
  - [using git](#using-git)
- [Build](#build)
  - [Run (using SLURM)](#run-using-slurm)
  - [Run (using MPI)](#run-using-mpi)
- [Profiling](#profiling)
  - [zones](#zones)
  - [system variables](#system-variables)
  - [compression benchmarking tool](#compression-benchmarking-tool)
- [Options](#options)
  - [codecs currently supported by the gpugraph500 binary](#codecs-currently-supported-by-the-gpugraph500-binary)
- [Troubleshooting](#troubleshooting)
- [Author](#author)
- [License](#license)
- [Resources](#resources)


# Requirements
- C compiler. C++ Compiler with c++11 support.
- An MPI implementation: OpenMPI (MPICH2 is not supported)
- To use CUDA-BFS or CUDA-compression: CUDA 6+ support.
- To use SIMD compression: SSE2 support (SEE4 support recommended)
- To use SIMD+ compression SSE2 support.
- Scalasca(Score-P) and CUBE4+ for instrumentation and profiling
- System packages: `libtool`, `automake`

# Installation
## downloading and decompressing:

```
$ mkdir gpugraph500
$ cd gpugraph500
$ wget https://github.com/UniHD-CEG/gpugraph500/archive/master.zip
$ unzip master.zip
```

## using git

```
$ git clone https://github.com/UniHD-CEG/gpugraph500.git
$ cd gpugraph500
```

# Build
The code to compile is in the folder `cpu_2d/`. to build the binary:

First build: (or when editing `configure.ac`)

```
$ cd cpu_2d
$ ./autogen.sh # ./configure options: (1)
$ make
```

Consecutive builds:

```
$ cd cpu_2d
$ ./configure --enable-aggressive-optimizations --enable-ptxa-optimizations  --disable-openmp --enable-compression --enable-simd  # ./configure options (1)
$ make
```

(1) for options, run `./configure --help`

## Run (using SLURM)

```
$ cd bfs_multinode
$ cd eval
$ sbatch o16p8n.rsh 22 # (Replace 22 with Scale Factor)
```

## Run (using MPI)

```
$ # ---- In cpu_2d directory ----:
$ mpirun -np 16 ../cpu_2d/g500 -s 22 -C 4 -gpus 1 -qs 2 -be "s4-bp128-d4" -btr 64 -btc 64 # available codecs listed below
```

runs a test with 16 proccesses in 8 nodes, using Scale Factor 21

# Profiling
This application allows the code to be instrumented in zones using Score-P (Scalasca) with very low overhead.

## zones
The names of the instrumented zones are listed below.

Zone (label)                      | Explanation
--------------------------------- | -------------------------------------------------------------:
BFSRUN_region_vertexBroadcast     |        Initial vertices broadcast (No compression implemented)
BFSRUN_region_localExpansion      |        Predecessor List Reduction (No compression implemented)
BFSRUN_region_columnCommunication |           Column communication phase (Implemented Compression)
BFSRUN_region_rowCommunication    |              Row communication phase (Implemented Compression)
BFSRUN_region_Compression         |      Row Compression (type convertions + Compression encoding)
BFSRUN_region_Decompression       |    Row Compression (type convertions + Decompression encoding)
CPUSIMD_region_encode             |                          Compression or decompression encoding
BFSRUN_region_vreduceCompr        |   Column Compression (type convertions + Compression encoding)
BFSRUN_region_vreduceDecompr      | Column Compression (type convertions + Decompression encoding)

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

Results will be stored on a folder with format `scorep-*` in the `/eval` folder.

To instrument graphically with CUBE run:

```
$ cd eval/scorep-____FOLDER_NAME____
$ cube profile.cubex
```

To instrument through the console run either:

1) The provided `scripts/Profiling/Statistics.sh` script. The options must be changed inside the script.

2) manually using CUBE:

```
$HOME/cube/bin/cube_stat -p -m time  -r BFSRUN_region_Compression,BFSRUN_region_Decompression,CPUSIMD_region_encode,BFSRUN_region_vreduceCompr,BFSRUN_region_vreduceDecompr profile.cubex
```

Flag -m <time|bytes_sent>

3) manually using scorep-score

```
$ cd eval/scorep-____FOLDER_NAME____
$ scorep-score -r profile.cubex
```

## compression benchmarking tool
See TurboPFOR in Resources

# Options
## codecs currently supported by the gpugraph500 binary

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

# Author

Computer Engineering Group at Ruprecht-Karls University of Heidelberg

# License
- Duane Merrill's BC40 (back40computing) is licenced under [Apache 2 Licence.](https://github.com/UniHD-CEG/gpugraph500/tree/master/b40c/LICENSE.TXT)
- SIMDcompressionAndIntersection is licenced under [Apache 2 Licence.](https://github.com/UniHD-CEG/gpugraph500/blob/master/cpu_2d/compression/cpusimd/LICENSE)
- Alenka GPU database engine is licensed under Apache 2 License.


Copyright (c) 2016, Computer Engineering Group at Ruprecht-Karls University of Heidelberg, Germany. All rights reserved.

Licensed under GNU/GPL version 3 https://www.gnu.org/licenses/gpl-3.0



# Resources
[OpenSource R suite: RStudio](https://www.rstudio.com/products/RStudio/)

[D. Lemire's SIMDCompression](https://github.com/lemire/SIMDCompressionAndIntersection)

[TurboPFOR SIMDCompression and Codec Benchmarking tool](https://github.com/powturbo/TurboPFor)

[Alenka's CUDA–PFOR Compression](https://github.com/antonmks/Alenka/blob/master/compress.cu)
