[TOC]

# Requirements

- C compiler. C++ Compiler with c++11 support.
- To use CUDA-BFS or CUDA-compression: CUDA 6+ support.
- To use SIMD compression: SSE4 support.
- To use SIMD+ compression SSE2 support.


# Download

- Create an account in [https://bitbucket.org](https://bitbucket.org)


## Using git

- Use your Bitbucket username account name as  ___Your_Bitbucket___

- [https://___Your_Bitbucket___@bitbucket.org/jyoung3131/bfs_multinode.git](https://___Your_Bitbucket___@bitbucket.org/jyoung3131/bfs_multinode.git)

- install `git`

- Clone the branch `architectural_tuning`,

```
$ # REPLACE ___Your_Bitbucket___
$ git clone -b architectural_tuning https://___Your_Bitbucket___@bitbucket.org/jyoung3131/bfs_multinode.git
$ cd bfs_multinode
```

## Using gz

- Download from:
[latest .gz](https://bitbucket.org/jyoung3131/bfs_multinode/get/architectural_tuning.tar.gz)

- Decompress:

```
$ tar -xzvf architectural_tuning.tar.gz
$ cd architectural_tuning
```

# Build

The code to compile is in the folder `cpu_2d/`. to build the binary:

```
./configure
make
```

for options, run `./configure --help`

# Run

Change to folder `eval/`

The tests are (valid with SLURM)

- o4p2n-roptim.rsh [SCALE_FACTOR]
- o4p2n-coptim.rsh [SCALE_FACTOR]
- o4p2n-noptim.rsh [SCALE_FACTOR]
- o9p8n.rsh [SCALE_FACTOR]
- o16p8n.rsh [SCALE_FACTOR]

```
sbatch o16p8n.rsh 21
```

runs a test with 16 proccesses in 8 nodes, using Scale Factor 21


# Scripts

* r-ify.sh
* r-compare.sh


Scripts are located under the `scripts/` folder

To install:

change to the `eval/` directory

```
$ ln -s ../scripts/r-ify.sh r-ify.sh
$ ln -s ../scripts/r-compare.sh r-compare.sh
$ chmod u+x *.sh
```


## Script r-ify.sh

It uses the execution traces to generate R-code. This R-code (once run) shows the time measurements of the Phases of the BFS code. This script uses result files with same Scale Factor. The results are represented as a Barplot.


```
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

As the previous script, this also uses the execution traces to generate R-code. This differs from the previous one in that it can compare several files from several Scale Factors. Results are visualized as a Lineplot.


```
$ ./r-compre.sh JOBID1 JOBID2 ...
```

## Script check-all.sh (requires SLURM)

This script automatizes the execution of tests for different Scale Factors.

```
$ check-all.sh 15 30
```

This will run the tests with format `o*.rsh` in the `eval/` folder for Scale Factors 15 to 30. Process is shown in ncurses-like format.


# Profiling

This BFS application allows the code to be instrumented in Zones using Score-P with very low overhead. This requires Score-P and Scalasca to be installed in the system. The results may be analyzed either visually (using CUBE) or through console using `scorep-score`.

These tools may be installed locally (no priviledged user is needed) using the external-apps-installer.sh aforementioned.

The names of the instrumentable Zones are listed below. Other Zones may be added if needed.

```
BFSRUN_region_vertexBroadcast
BFSRUN_region_nodesTest
BFSRUN_region_localExpansion
BFSRUN_region_testSomethingHasBeenDone
BFSRUN_region_columnCommunication
BFSRUN_region_rowCommunication
```

The first step is to update the system variables. This may be done either on .bashrc or in a separate script.

Update the paths in the variables below

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

```
$ cd eval/scorep-____FOLDER_NAME____
$ cube profile.cubex
```

To instrument through the console run:

```shell
$ cd eval/scorep-____FOLDER_NAME____
$ scorep-score -r profile.cubex
```

# Current Limitations

## Out-Of-Memory errors and CUDA memory size limitations:

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


# Troubleshooting
- Problem: In the .out file of Slurm/ Sbatch execution I get the text:

```
S=C=A=N: Abort: No SCOREP instrumentation found in target ../cpu_2d/g500
```

- Solution:

The instrumentation is activated for the runtime execution (i.e: the binary is being run prefixed with scalasca).

Disable it with:

```
$ export G500_ENABLE_RUNTIME_SCALASCA=no
```

# License

- This code contains a subset of Duane Merrill's BC40 repository of GPU-related functions, including his BFS implementation used in the paper, Scalable Graph Traversals.
- SIMDcompressionAndIntersection is licenced under Apache Licence 2.0.
- All copyrights reserved to their original owners.

# External Resources
[https://www.rstudio.com/products/RStudio/](https://www.rstudio.com/products/RStudio/)
