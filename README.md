[TOC]

# BFS Application
## Requirements
- Compiler with std c++11 support (e.g GNU C/C++ v4.8.1+)
- UNIX-like OS with CUDA 6+ support

## Installation, Setup and Execution of the BFS code
### Downloading
Create an account in [https://bitbucket.org](https://bitbucket.org) Request access for bfs_multinode repository

Available repositories are:(Replace the token **Your_Bitbucket_User**)


- [https://Your_Bitbucket_User@bitbucket.org/g500optimization/bfs_multinode.git](https://Your_Bitbucket_User@bitbucket.org/g500optimization/g500optimization.git) **(Updated daily)**
- [https://Your_Bitbucket_User@bitbucket.org/jyoung3131/bfs_multinode.git](https://Your_Bitbucket_User@bitbucket.org/jyoung3131/bfs_multinode.git) **(Updated weekly)**

```shell
$ # REPLACE the token Your_Bitbucket_User
$ git clone https://Your_Bitbucket_User@bitbucket.org/julianromera/bfs_multinode.git
$ git checkout -b architectural_tuning
$ git pull origin architectural_tuning
```

### Setting up the environment
1- Add CUDA libraries and binaries to your path:

Add this text to your ~/.bashrc

```shell
# UPDATE NEXT LINE
export CUDA_PATH=/usr/local/cuda-7.0
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/lib:$CUDA_PATH/extras/CUPTI/lib64:$CUDA_PATH/extras/CUPTI/lib:$LD_LIBRARY_PATH
```

2- Optional. In case of using EXTOLL add these lines to your ~/.bashrc

```shell
# Adjust next line to your actual path
export EXTOLL_HOME=/extoll2
if [ -d /extoll2 ]; then
       source $EXTOLL_HOME/extoll_env.bash
       # ensure this path is correct
       export MPI_PATH=$EXTOLL_HOME/mpi/openmpi-1.6.4
       export PATH=$MPI_PATH/bin:$PATH

       export LD_LIBRARY_PATH=$EXTOLL_HOME/lib:$MPI_PATH/lib:$LD_LIBRARY_PATH
       ulimit -l 4194304
       ulimit -s unlimited
fi
```

### Building
The code to compile is in the folder `cpu_2d/`. It is built using `Make`.

The (currently) available Makefiles are:


- Makefile.gcc
- Makefile.gcc.keeneland

The Makefiles in the list below are currently being developed:


- Makefile.Intel
- Makefile.cpu

The file `Makefile` is a symbolic link to the Makefile.___ being used. It may be created/ edited using:

```shell
$ cd bfs_multinode/cpu_2d
$ rm -f Makefile
$ ln -s Makefile.gcc Makefile
```

```shell
$ cd bfs_multinode/cpu_2d
$ make
$ cd ../eval
```

#### Makefile Configuration
The Makefiles in the `cpu_2d/` folder are configurable. Edit the Makefile file to access the documentation of the variables. The documentation is placed in the header of the file.

The default variables-values for Makefile.gcc.keeneland are:

```Makefile
nvidia_architecture                       :="fermi"
nvidia_ptxas_otimize                      :="no"
manual_profiler_cuda                      :="no"
manual_profiler_other_compilers           :="yes"
openmp_on_cuda                            :="no"
openmp_on_other_compilers                 :="no"
custom_openmpi                            :="yes"
custom_openmpi_basedir                    :=/home/jromera/openmpi
scorep_profiler_enable                    :="yes"
scorep_profiler_automatic_instrumentation :="no"
scorep_custom                             :="yes"
scorep_custom_basedir                     :=/home/jromera/scorep
use_avx_instructions                      :="yes"
enable_compression                        :="no"
enable_simd_compression                   :="no"
enable_simt_compression                   :="no"
enable_compression_verify                 :="no"
enable_compression_debug                  :="no"
use_cuda                                  :="yes"
debug                                     :="no"
debug_code                                :="no"
quiet_output                              :="yes"
code_optimization_level                   :="O4"
code_optimization_flags                   :=-funroll-loops -flto
```

### Test Scenarios
Test scenarios are in the folder `eval/`

Sbatch relevant Test-cases in the previous list are:


- o4p2n-roptim.rsh [SCALE_FACTOR]
- o4p2n-coptim.rsh [SCALE_FACTOR]
- o4p2n-noptim.rsh [SCALE_FACTOR]
- o9p8n.rsh [SCALE_FACTOR]
- o16p8n.rsh [SCALE_FACTOR]

The name in these files explains what they do. For example:

The part "4p2n" indicates 4 Processes will be run in 2 Nodes. These scripts are run through Slurm. "16p8n" would indicate 16 processes distributed in 8 Nodes.

## Runing tests
Ensure that the Slurm queue is empty

```shell
$ squeue
JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)
```

Use Sbatch to execute the .rsh file. Add a numeric argument at the end (Sf)

```shell
$ sbatch o4p2n_roptim.rsh 21
Submitted batch job 423
$ sbatch o4p2n_coptim.rsh 21vv
Submitted batch job 424
$ sbatch o4p2n_noptim.rsh 21
Submitted batch job 425
```

# Automation Scripts
## List of scripts
* r-ify.sh
* r-compare.sh
* check-all.sh
* external-apps-installer.sh


## Installing scripts
Scripts are located under the `scripts/` folder

To install

- r-ify.sh
- r-compare.sh
- check-all.sh

Run:

```shell
$ cd bfs_multinode
$ # Make a symlink from eval to the script file
$ cd eval
$ ln -s ../scripts/r-ify.sh r-ify.sh
$ ln -s ../scripts/r-compare.sh r-compare.sh
$ ln -s ../scripts/check-all.sh check-all.sh
$ chmod u+x *.sh
```

To install (Useful for Profiling and Tracing)

- external-apps-installer.sh

Run:

```shell
$ cp scripts/external-apps-installer.sh $HOME
$ cd $HOME
$ chmod u+x external-apps-installer.sh
...
```

## Script r-ify.sh
### Description
It uses the execution traces to generate R-code. This R-code (once run) shows the time measurements of the Phases of the BFS code. This script uses result files with same Scale Factor. The results are represented as a Barplot.

### Execution

```shell
$ ./r-ify.sh 423 424 425

-> File slurm-423.out. Validation passed (Tasks: 4, GPUS/Task: 1, Scale Factor: 21).
-> File slurm-424.out. Validation passed (Tasks: 4, GPUS/Task: 1, Scale Factor: 21).
-> File slurm-425.out. Validation passed (Tasks: 4, GPUS/Task: 1, Scale Factor: 21).
Enter new labels for the X-Axe? (y/n) [n] y
Enter a total 3 label(s) between quoutes. Separate them with spaces: "4p2n-Roptim" "4p2n-Coptim" "4p2n-Noptim"
-> Created file "file-423-424-425.r".
-> R-Code successfully generated.
```

Open the file `file-423-424-425.r` with your R editor and run the code.

## Script r-compare.sh
### Description
As the previous script, this also uses the execution traces to generate R-code. This differs from the previous one in that it can compare several files from several Scale Factors. Results are visualized as a Lineplot.

### Execution

```shell
$ ./r-ify.sh
(to be completed)
```

## Script check-all.sh
### Description
This script automatizes the execution of tests for different Scale Factors.

### Execution

```shell
$ check-all.sh 15 30
```

This will run the tests with format `o*.rsh` in the `eval/` folder for Scale Factors 15 to 30. Process is shown in ncurses-like format.

## Script external-apps-installer.sh
### Description
This script downloads, decompress, builds and installs Score-P, Scalasca and CUBE locally. Optionally it installs also a compatible version of OpenMPI.<br>This script is configurable. The configuration may be changed editing the script.

Root priviledges are not required to perform the installation.

### Execution

```shell
$ cd $HOME
$ ./external-apps-installer.sh
```

# Current compression results

## Data Volumes

- Test uses 16 proccesses, 8 Nodes, ScaleFactor 22
- HW: Gigabit Ethernet. 2xNVIDIA Fermi cards per node.


| Vertex broadcast                         |                                        |
| :--------------------------------------: | :------------------------------------- |
| Before                                   | CPU-SIMD threshold64 Codec s4-bp128-d4 |
| 8192 bytes                               | 8192 bytes                             |
| Difference: 0 bytes                      |                                        |
| 0% gain                                  |                                        |

| Column communication                     |                                        |
| :--------------------------------------: | :------------------------------------: |
| Before                                   | CPU-SIMD threshold64 Codec s4-bp128-d4 |
| 7160177440 bytes                         | 5314394624 bytes                       |
| Difference: 1845782816 bytes             |                                        |
| 26% gain                                 |                                        |


| Column communication                     |                                        |
| :--------------------------------------: | :------------------------------------: |
| Before                                   | CPU-SIMD threshold64 Codec s4-bp128-d4 |
| 4904056832 bytes                         | 806405024 bytes                        |
| Difference: 4097651808 bytes             |                                        |
| 84% gain                                 |                                        |

# Changelog and Improvements

## Current Changelog
* version 1.1 (tag v1.1)
    * Added Row compression (~7% improvement).
    * Compression packetsize threshold tuning. (~?% improvement).

* version 1.0 (tag v1.0)
    * Compile-time optimizations (-ON flag).
    * Column compression uses SIMD (on CPU). No row Compression (~20% improvement).


* initial version


## Further future improvements/ challenges

A) Implementation improvements

* Replace Thrust order() with CUB order().
    * Motivation for this: We order our integer FQ secuence on each BFS. We use Thrust library for this
    * Duane Merrill from NVIDIA states that due to the configurability/ tuning options of this library, designed for NVIDIA cards, a boost in performance may be expected. Futher reading in [Duane Merrill's Why CUB?](http://nvlabs.github.io/cub/#sec4)
* Optimize Bitwise operations. Further reading in [Sean Eron's Bit Twiddling Hacks page](https://graphics.stanford.edu/~seander/bithacks.html)
* Use memcpy instead of Indexed Loops in compression/ decompression calls.
    * Motivation. Several experiments on Internet, point a better memory usage with block memory copying. Furter reading in [David Nadeau's blog](http://nadeausoftware.com/articles/2012/05/c_c_tip_how_copy_memory_quickly)
    * Indexed Loops are now used instead, to "Keep it Simple".
* Move the generation of the Compression code and the generation of the Compression integration calls (both, by a software Factory pattern) to main.cpp (outside the BFSrun code)
    * Motivation. Passing the created object as a mem pointer instear of creating/deleting on each call, may slightly decrease the observed overhead.

B) Algorthmic improvements

* None yet. Research. Read about the State of Art of BFS, Titech

C) Implementation & Algorthmic improvements

* Select an optimum SIMD codec for the patterns of integer sequences used in this app.
* Porting codec implementation to GPU, some of the measured compression overhead will dissapear.
* Check and recode the column-reduction's data transfer.
    * Motivation. currently 4 MPI_Sendrecv() One-to-One, blocking calls are being used for data transfer. Study if the observed column's unarcheived performance increasement is due to an impairement on the transfer throught using this MPI_Sendrecv() calls.
* Study the possibility of using non-blocking data transfer MPI calls in column reduction.




# Other
## Profiling and Tracing
This BFS application allows the code to be instrumented in Zones using Score-P with very low overhead. This requires Score-P and Scalasca to be installed in the system. The results may be analyzed either visually (using CUBE) or through console using `scorep-score`.

These tools may be installed locally (no priviledged user is needed) using the external-apps-installer.sh aforementioned.

The names of the instrumentable Zones are listed below. Other Zones may be added if needed.

```c
BFSRUN_region_vertexBroadcast
BFSRUN_region_nodesTest
BFSRUN_region_localExpansion
BFSRUN_region_testSomethingHasBeenDone
BFSRUN_region_columnCommunication
BFSRUN_region_rowCommunication
```

The first step is to update the system variables. This may be done either on .bashrc or in a separate script.

Update the paths in the variables below

```shell
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

export LD_LIBRARY_PATH=$HOME/scalasca/lib:$LD_LIBRARY_PATH
export PATH=$HOME/scalasca/bin:$PATH

export MPI_PATH=/home/jromera/openmpi
export PATH=$MPI_PATH/bin:$CUDA_PATH/bin:$PATH

export LD_LIBRARY_PATH=$MPI_PATH/lib:$CUDA_PATHo/lib64:$CUDA_PATHo/lib64/stubs:$CUDA_PATHo/lib:$CUDA_PATHo/extras/CUPTI/lib64:$CUDA_PATHo/extras/CUPTI/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/scorep/lib:$LD_LIBRARY_PATH
export PATH=$HOME/scorep/bin:$PATH
EOF
```

The variable `G500_ENABLE_RUNTIME_SCALASCA` set to yes will enable the required runtime instrumentor of Scalasca.

Results will be stored on a folder with format `scorep-*` in the `/eval` folder.

To instrument graphically with CUBE run:

```shell
$ cd eval/scorep-____FOLDER_NAME____
$ cube profile.cubex
```

To instrument through the console run:

```shell
$ cd eval/scorep-____FOLDER_NAME____
$ scorep-score -r profile.cubex
```

## Current Limitations
### Out-Of-Memory errors and CUDA memory size limitations:
For some high Score-Factors (e.g: 22, as of the day of writing this guide), the resulting Slurm trace will be:

```
Using SCALE-FACTOR 22
Tue Jul 14 15:42:51 CEST 2015
 ========================   JOB MAP   ========================

 Data for node: creek01 Num procs: 2
        Process OMPI jobid: [1404,1] Process rank: 0
        Process OMPI jobid: [1404,1] Process rank: 2

 Data for node: creek02 Num procs: 2
        Process OMPI jobid: [1404,1] Process rank: 1
        Process OMPI jobid: [1404,1] Process rank: 3

 =============================================================
row slices: 2, column slices: 2
graph_generation:               7.749108 s
Input list of edges genereted.
6.710886e+07 edge(s) generated in 8.499692s (7.895447 Medges/s on 4 processor(s))
Adjacency Matrix setup.
2.956432e+06 edge(s) removed, because they are duplicates or self loops.
1.283049e+08 unique edge(s) processed in 18.308235s (7.008041 Medges/s on 4 processor(s))
[../b40c/graph/bfs/csr_problem_2d.cuh, 697] CsrProblem cudaMalloc frontier_queues.d_values failed (CUDA error 2: out of memory)
[cuda/cuda_bfs.cu, 486] Reset error. (CUDA error 2: out of memory)
MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD
with errorcode 1.
```

### Validation errors for the BFS runs
Looking at your resulting Sbatch trace you may see:

**"Validation: failed"** / or an output trace similar to the one below:

The issue seems to be related with compiler version errors. The version of the compiler used for the CUDA code may not be met. See the Requirements section above.

```
$ cat slurm-JOBID.out
...
BFS Iteration 24: Finished in 0.111173s
max. local exp.:     0.004383s(3.942371%)
max. queue handling: 0.056909s(51.189917%)
est. rest:           0.049881s(44.867712%)
max. row com.:       0.038369s(34.513278%)
max. col com.:       0.069516s(62.529783%)
max. pred. list. red:0.007413s(6.667911%)
(3:0) Edge [26849(2273):244(244)] with wrong levels: 3 -1
Validation of iteration 24 finished in 0.022564s
Result: Invalid 524279 Edge(s) processed, 4.715888 MTeps
(0:3) Edge [244(244):26849(2273)] with wrong levels: -1 3
(0:3) Edge [244(244):29960(5384)] with wrong levels: -1 4
(3:0) Edge [29960(5384):244(244)] with wrong levels: 4 -1
BFS Iteration 25: Finished in 0.096601s
max. local exp.:     0.000905s(0.937128%)
max. queue handling: 0.048780s(50.496083%)
est. rest:           0.046916s(48.566789%)
max. row com.:       0.032854s(34.009833%)
max. col com.:       0.054057s(55.958921%)
max. pred. list. red:0.009008s(9.324883%)
Validation of iteration 25 finished in 0.020493s
Result: Valid 524280 Edge(s) processed, 5.427272 MTeps
invalid_level
invalid_level
...
```

## Troubleshooting
- Problem: In the .out file of Slurm/ Sbatch execution I get the text:

  ```
  S=C=A=N: Abort: No SCOREP instrumentation found in target ../cpu_2d/g500
  ```

- Solution:

  The instrumentation is activated for the runtime execution (i.e: the binary is being run prefixed with scalasca).

  Disable it with:

  ```shell
  $ export G500_ENABLE_RUNTIME_SCALASCA=no
  ```

## About this document
- Version 1.0
- Last revision: 2nd November 2015

## Editing this code

- Compilable code is located on the `cpu_2d/` directory.
- The `INTEGRATION_README` files contain information about each directory.
- The command `astyle` (optional) may be used to easily format the code. Avoids time consumming changes, and makes the code structure uniform.

1- Installing `astyle`:
```
$ # On debian/ ubuntu:
$ sudo apt-get install astyle

$ # On Mac:
$ brew install astyle
```

2- run:
```shell
$ # Create a link to the script (Only once)
$ cd cpu_2d
$ ln -s ../scripts/astyle.sh ./

$ # at any time:
$ cd cpu_2d
$ ./astyle.sh

```

## License
- This code contains a subset of Duane Merrill's BC40 repository of GPU-related functions, including his BFS implementation used in the paper, Scalable Graph Traversals.
- SIMDcompressionAndIntersection is licenced under Apache Licence 2.0.
- All copyrights reserved to their original owners.

## External Resources
[https://www.rstudio.com/products/RStudio/](https://www.rstudio.com/products/RStudio/)
