#########################################################
# Makefile - GCC
#########################################################
#
#
# Configurable options:
#
# nvidia_architecture
#   Values: fermi | kepler | maxwell | auto
#   Default value: auto
#   Description: "use all for generic code: Fermi and Kepler.
#
# nvidia_ptxas_otimize
#   Values: yes | no
#   Default value: no
#   Description: "enables -O3 optimization on ptax cuda regiters"
#
# openmp_on_cuda
#   Values: yes | no
#   Default value: no
#   Description: "Hybrid execution mode. OpenMP threads are added to
#  				  BFSruns processed with cuda"
#
# openmp_on_other_compilers
#   Values: yes | no
#   Default value: no
#   Description: "Hybrid execution mode. OpenMP threads are added to
#  				  other high O() code"
#
# manual_profiler_cuda
#   Values: yes | no
#   Default value: no
#   Description: "prints out information on initialization"
#
# manual_profiler_other_compilers
#   Values: yes | no
#   Default value: no
#   Description: "prints out results/ statistics at the end"
#
# custom_openmpi
#   Values: yes | no
#   Default value: no
#   Description: "use user compiled openmpi instead of system's"
#
# custom_openmpi_basedir
#   Values: path
#   Default value: CHANGE_ME
#   Description: "if previous option is set, absolute path to dir"
#
# scorep_profiler_enable
#   Values: yes | no
#   Default value: no
#   Description: "is scorep has been installed, enable it on code"
#
# scorep_profiler_manual_instrument
#   Values: yes | no
#   Default value: no
#   Description: "setting this option to yes, manual instrumentation is enabled.
#  				  Thus, the zones to be instrumented will be the currently
#  				  added zones using code wrappers (i.e: globalbfs.hh file)"
# 				  https://silc.zih.tu-dresden.de/scorep-current/html/instrumentation.html#manual_instrumentation
#
# scorep_custom
#   Values: yes | no
#   Default value: yes
#   Description: "use custom compiled scorep instead of systems one"
#
# scorep_custom_basedir
#   Values: path
#   Default value: CHANGE_ME
#   Description: "see custom_openmpi_basedir option above"
#
# use_avx_instructions
#   Values: yes | no
#   Default value: yes
#   Description: "CPU SSExx and AMD compatible AVX will be used.
#   				Required for SIMDcompression librasy"
#
# enable_compression
#   Values: yes | no
#   Default value: yes
#   Description: "Enables any type of compression through the network. Switch this option
#   				to "no" to disable completely compression. This option requires that
#   				either simd_compression_enabled (CPU) or simd_compression_enabled (GPU)
#   				are enabled.
#
# enable_compression_debug
#   Values: yes | no
#   Default value: no
#   Description: "shows statistics of compression rate, time of compression
#   			  decompression fot the compression schema used (cpusimd, gpusimt, etc),
#   			  and for each codec (in the case id at Lemire's simdcompress) if available."
#
# enable_compression_verify
#   Values: yes | no
#   Default value: no
#   Description: "sends both compressed, and decompressed data through
#   			  the network. Checks decompression at peer node. This option
#   			  impacts severely on performance since the original Frontier Queue
#   			  has to be transmitted along with the compressed one."
#
# simd_compression_enabled
#   Values: yes | no
#   Default value: yes
#   Description: "MPI packets will be sent compressed by using
#   			  D. Lemire SIMDCompression library"
#   External links:
#   			- https://github.com/lemire/SIMDCompressionAndIntersection
#
# debug
#   Values: yes | no
#   Default value:
#   Description: "Provides extra data in runtime execution.
#  				recorded in slurm-xx.out file"
#
# quiet_output
#   Values: yes | no
#   Default value: no
#   Description: "Disable verbose on Make command execution"
#
# use_cuda
#   Values: yes | no
#   Default value: yes
#   Description: "use CUDA implementation for the BFS algorithm instead of CPU"
#
# code_optimization_level
#   Values: O4 | O3 | O2 | O1 | O0
#   Default value:
#   Description: "Compiler optimization level. O0=none, O4=maximum"
#
# code_optimization_flags
#    Values: -funroll-loops -flto -finline-functions -funswitch-loops -fpredictive-commoning
#            -fgcse-after-reload -ftree-vectorize -ftree-partial-pre -fipa-cp-clone ...
#
#    Default value: -funroll-loops -flto (For GNU's version of Makefile)
#    Format: value value value (separated by spaces)
#    Description: "Optimization flags added to compiler"
#    External links:
#    			- GNU compiler: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#Optimize-Options
#
# debug_code
#   Values: yes | no
#   Default value: no
#   Description: "Enables symbol inclusion in the binary (compiler option -g).
#   				Disables optimizations (compiler-O0) and removes code_optimization_flags.
#   				It is useful to run this option with Valgrind in the runtime Test-file
#   				for small SLURM grids scripts/debugger.sh (run from eval/) will wrap the
#   				executable in MPI and SLURM"
#


#CXX = `locate bin/g++- | grep "bin/g++-[0-9]" | tail -1`
#CC = $(shell locate bin/gcc- | grep "bin/gcc-[0-9]" | tail -1`
#NVCXX = `locate bin/nvcc | grep bin/nvcc$$ | tail -1`
#NVDIR = `echo $(NVCXX) | sed 's,/bin/nvcc$$,,'`


ACLOCAL_AMFLAGS = -I m4

#
# Code Optimization levels. Read debug_code option help above.
#

if OPTIMIZATION_O4
	o_level = -O3 -ffast-math
	o_level_nvcc = -O3
endif
if OPTIMIZATION_O3
	o_level = -O3
	o_level_nvcc = -O3
endif
if OPTIMIZATION_O2
	o_level = -O2
	o_level_nvcc = -O2
endif
if OPTIMIZATION_O1
	o_level = -O1
	o_level_nvcc = -O1
endif
if OPTIMIZATION_O0
	o_level = -O0
	o_level_nvcc = -O0
endif


if AVX
	arch = -mavx
else
if SIMD
	arch = -msse4
else
	arch = -march=native
endif
endif


if PTXAS_OTIMIZATION
	PTXAS = --ptxas-options="-O3"
endif


if NVIDIA_ARCHITECTURE_FERMI
	NV_ARCHITECTURE = $(PTXAS) -fmad=true -arch=sm_20 -ftz=true -prec-div=false -prec-sqrt=false
else
if NVIDIA_ARCHITECTURE_KEPLER
	NV_ARCHITECTURE = $(PTXAS) -arch=sm_35
else
	NV_ARCHITECTURE = $(PTXAS) -gencode=arch=compute_20,code=\"sm_20,compute_20\" -gencode=arch=compute_30,code=\"sm_30,compute_30\" -gencode=arch=compute_35,code=\"sm_35,compute_35\"
endif
endif

#
# Instrumentation with Score-P
#
#

if SCOREP
	threads = none
	if OPENMP_ON_CUDA
		threads = omp
	endif
	if OPENMP_ON_OTHERS
		threads = omp
	endif
	scorep_instrumentation_flag =--cuda --mpp=mpi --thread=$(threads) --nopdt
# if SCOREP_CUSTOM
# if SCOREP_PROFILER_MANUAL_INSTRUMENTATION
	scorep_instrumentation_flag =--user --cuda --mpp=mpi --thread=$(threads) --nopdt --online-access --instrument-filter="filter.scorep"
	SCOREP_INSTRUMENTATION = -D_SCOREP_USER_INSTRUMENTATION -DSCOREP_USER_ENABLE
	SCOREP_CPPFLAGS = `scorep-config --cppflags`
	SCOREP_LLIBS = `scorep-config --libs`
	SCOREP_LFLAGS = `scorep-config --ldflags`
# endif
#	export LD_LIBRARY_PATH = $(scorep_custom_basedir)/lib::$(LD_LIBRARY_PATH)
#	libcudart_static_path = -L"`locate libcudart_static.a | tail -1 | sed -e 's,/libcudart_static.a,,'`"
#	libcudart_static_flag = -lcudart_static
#	PROFILER = scorep $(scorep_instrumentation_flag)
#else
#if SCOREP_PROFILER_MANUAL_INSTRUMENTATION
#	scorep_instrumentation_flag =--pomp --cuda --mpp=mpi --thread=$(threads) --nopdt --nocompiler
#endif
#	PROFILER = scorep $(scorep_instrumentation_flag)
#endif
else
	libcudart_static_path = -L$(NVDIR)/lib64
	libcudart_static_flag = -lcudart
endif

if OPENMP_ON_CUDA
	CUDA_OPENMP = -fopenmp
endif

if OPENMP_ON_OTHERS
	OPENMP = -fopenmp
endif

if BASIC_PROFING_CUDA
	CUDA_INSTRUM = -DINSTRUMENTED
endif

if BASIC_PROFING_OTHERS
	INSTRUMENTED = -DINSTRUMENTED
endif

if DEBUG
	DEFINES +=-D_DEBUG
endif


if COMPRESSION
if !SIMD
if !SIMT
 @echo "Compression is enabled, but neither CPU nor GPU compression is set"
 @exit
endif
endif
	DEFINES += -D_COMPRESSION
if DEBUG_COMPRESSION
	DEFINES += -D_COMPRESSIONDEBUG
	LLIBC = -static-libstdc++
endif
if VERIFY_COMPRESSION
	DEFINES += -D_COMPRESSIONVERIFY
endif
#
#
# Compression Block
#
#

# 1. CPU/SIMD Compression.
#    CPU with SSE4 support is required. CPU with AVX support is recommended.
#
if SIMD
	DEFINES += -D_SIMDCOMPRESS
	INCPATH = -I. -Icompression/cpusimd/include
	HEADERS = `ls compression/cpusimd/include/*h`
else
	INCPATH = -I.
	HEADERS =
endif
# 2. XXX/XXXX Compression.
# ...
#
endif

#
# Miscellanous
#
#

if CUDA
	DEFINE_CUDA = -D_CUDA
	bfs_object = cuda_bfs_cuda.o
else
	bfs_object =  cpubfs_bin.o
	bfs_implementation = cpubfs_bin.cpp
endif

#if CUSTOM_OPENMPI
#	export OMPI_CC = $(CC)
#	export OMPI_CXX = $(CXX)
#	MPICC = $(custom_openmpi_basedir)/bin/mpicc
#	MPICXX = $(custom_openmpi_basedir)/bin/mpicxx
#else
#	MPICC = mpicc
#	MPICXX = mpicxx
#endif

if QUIET_OUTPUT
	Q = @
endif

if DEBUGGING
	o_level = -O0
	o_level_nvcc = -O0
	DEBUG_FLAG=-g
	NVCC_DEBUG_FLAGS=-DTHRUST_DEBUG
	code_optimization_flags =
endif

#
# Compiler/ Linker options: C, C++, MPI, CUDA Flags and Libs
#
#
#

MPICFLAGS = $(arch) -std=c99 -pedantic -m64 -mfpmath=sse $(code_optimization_flags) -pipe $(DEFINE_CUDA) $(DEFINES) $(OPENMP) $(DEBUG_FLAG) -Wall -Wextra
MPICXXFLAGS = $(arch) -std=c++11 -pedantic -m64 $(o_level) $(code_optimization_flags) -pipe $(DEFINE_CUDA) $(DEFINES) $(OPENMP) $(INSTRUMENTED) $(DEBUG_FLAG) -Wall -Wextra
NVCXXFLAGS = $(arch_nvcc),-m64,$(o_level_nvcc),-pipe,$(DEFINE_CUDA),$(CUDA_INSTRUM),$(CUDA_OPENMP),$(DEBUG_FLAG)
LLIBS = $(SCOREP_LLIBS) $(libcudart_static_path) -L$(NVDIR)/lib $(libcudart_static_flag) -lcuda $(LLIBC)

LDFLAGS = -m64 $(SCOREP_LFLAGS) $(OPENMP) -Wl,$(o_level)
CXXFLAGS = $(arch) -std=c++11  -Weffc++ -pedantic -m64 $(o_level) $(code_optimization_flags) $(DEBUG_FLAG) -Wall -Wextra
CFLAGS = $(arch) -std=c99 -pedantic -m64 $(o_level) $(code_optimization_flags) $(DEBUG_FLAG) -Wall -Wextra


#NVDIR = $(NVDIR)
#CXX = $(CXX)
#CC = $(CC)
#MPICC = $(MPICC)
#MPICXX = $(MPICXX)
#LINK = $(PROFILER) $(MPICXX)

NVCXX=$(NVCC_PATH)
AM_CC=$(MPICC)
AM_CXX=$(MPICXX)
AM_F77=$(MPIF77)
LINKER=$(PROFILER) $(MPICXX)

#
# List of sources to compile
#
#
#

OUTFILES = `ls *.o`
# SOURCES = main.cpp generator/utils.c generator/splittable_mrg.c generator/make_graph.c generator/graph_generator.c $(bfs_implementation) validate/onesided.c validate/onesided_emul.c bitlevelfunctions.cpp

if SIMD
	COMPRESSION_OBJECTS = bitpacking.o integratedbitpacking.o simdbitpacking.o usimdbitpacking.o simdintegratedbitpacking.o varintdecode.o streamvbyte.o simdpackedsearch.o simdpackedselect.o frameofreference.o for.o
else
	COMPRESSION_OBJECTS =
endif

MAIN_OBJECTS = $(bfs_object) main.o utils.o splittable_mrg.o make_graph.o graph_generator.o onesided.o onesided_emul.o bitlevelfunctions.o

if COMPRESSION
	g500_OBJECTS = $(COMPRESSION_OBJECTS) $(MAIN_OBJECTS)
else
	g500_OBJECTS = $(MAIN_OBJECTS)
endif

#
# Make/ Build rules
#
#
#

all: clean-local g500

main.o: main.cpp validate/validate.h distmatrix2d.hh generator/graph_generator.h generator/user_settings.h validate/mpi_workarounds.h generator/utils.h \
		generator/splittable_mrg.h validate/onesided.h generator/make_graph.h cuda/cuda_bfs.h cuda/cuda_support.hh \
		../b40c/util/basic_utils.cuh ../b40c/graph/bfs/csr_problem_2d.cuh ../b40c/util/cuda_properties.cuh ../b40c/util/memset_kernel.cuh ../b40c/util/cta_work_progress.cuh ../b40c/util/error_utils.cuh \
		../b40c/util/device_intrinsics.cuh ../b40c/util/reduction/warp_reduce.cuh ../b40c/util/operators.cuh ../b40c/util/io/modified_load.cuh ../b40c/util/vector_types.cuh ../b40c/util/io/modified_store.cuh \
		../b40c/util/multiple_buffering.cuh ../b40c/graph/bfs/problem_type.cuh ../b40c/partition/problem_type.cuh ../b40c/radix_sort/sort_utils.cuh \
		../b40c/graph/bfs/enactor_multi_gpu_2d.cuh ../b40c/util/spine.cuh ../b40c/util/kernel_runtime_stats.cuh ../b40c/graph/bfs/enactor_base.cuh \
		../b40c/graph/bfs/two_phase/contract_atomic/kernel.cuh ../b40c/util/cta_work_distribution.cuh ../b40c/graph/bfs/two_phase/contract_atomic/cta.cuh ../b40c/util/scan/cooperative_scan.cuh \
		../b40c/util/srts_grid.cuh ../b40c/util/numeric_traits.cuh ../b40c/util/reduction/cooperative_reduction.cuh ../b40c/util/reduction/serial_reduce.cuh \
		../b40c/util/scan/serial_scan.cuh ../b40c/util/scan/warp_scan.cuh ../b40c/util/io/initialize_tile.cuh ../b40c/util/io/load_tile.cuh \
		../b40c/util/io/store_tile.cuh ../b40c/util/io/scatter_tile.cuh ../b40c/util/reduction/tree_reduce.cuh ../b40c/graph/bfs/two_phase/contract_atomic/kernel_policy.cuh \
		../b40c/util/srts_details.cuh ../b40c/graph/bfs/two_phase/expand_atomic/kernel.cuh ../b40c/graph/bfs/two_phase/expand_atomic/cta.cuh ../b40c/util/soa_tuple.cuh \
		../b40c/util/scan/soa/cooperative_soa_scan.cuh ../b40c/util/reduction/soa/cooperative_soa_reduction.cuh ../b40c/util/reduction/soa/serial_soa_reduce.cuh ../b40c/util/reduction/soa/warp_soa_reduce.cuh \
		../b40c/util/scan/soa/warp_soa_scan.cuh ../b40c/util/scan/soa/serial_soa_scan.cuh ../b40c/graph/bfs/two_phase/expand_atomic/kernel_policy.cuh ../b40c/util/srts_soa_details.cuh \
		../b40c/graph/bfs/partition_contract/policy.cuh ../b40c/partition/policy.cuh ../b40c/partition/spine/kernel.cuh ../b40c/scan/spine/kernel.cuh \
		../b40c/scan/downsweep/cta.cuh ../b40c/scan/problem_type.cuh ../b40c/reduction/problem_type.cuh ../b40c/scan/kernel_policy.cuh \
		../b40c/partition/upsweep/tuning_policy.cuh ../b40c/partition/downsweep/tuning_policy.cuh ../b40c/graph/bfs/partition_contract/upsweep/kernel.cuh ../b40c/graph/bfs/partition_contract/upsweep/cta.cuh \
		../b40c/partition/upsweep/cta.cuh ../b40c/partition/upsweep/aggregate_counters.cuh ../b40c/partition/upsweep/composite_counters.cuh ../b40c/partition/upsweep/tile.cuh \
		../b40c/graph/bfs/partition_contract/upsweep/tile.cuh ../b40c/graph/bfs/partition_contract/upsweep/kernel_policy.cuh ../b40c/partition/upsweep/kernel_policy.cuh ../b40c/graph/bfs/partition_contract/downsweep/kernel.cuh \
		../b40c/graph/bfs/partition_contract/downsweep/cta.cuh ../b40c/partition/downsweep/cta.cuh ../b40c/graph/bfs/partition_contract/downsweep/tile.cuh \
		../b40c/partition/downsweep/tile.cuh ../b40c/graph/bfs/partition_contract/downsweep/kernel_policy.cuh ../b40c/partition/downsweep/kernel_policy.cuh \
		../b40c/graph/bfs/copy/kernel.cuh ../b40c/graph/bfs/copy/cta.cuh ../b40c/graph/bfs/copy/kernel_policy.cuh globalbfs.hh bitlevelfunctions.h cpubfs_bin.h \
		compression/cpusimd.hh compression/nocompression.hh
	@echo ""
	@echo "--- Compiling g500"
	$(Q)$(MPICXX) -c $(MPICXXFLAGS) $(SCOREP_INSTRUMENTATION) $(SCOREP_CPPFLAGS) $(INCPATH) -o main.o main.cpp
	@echo "    MPICC     $@"

bitpacking.o: compression/cpusimd/include/bitpacking.h compression/cpusimd/src/bitpacking.cpp
if SIMD
	@echo ""
	@echo "============================================================================"
	@echo "== Compiling"
	@echo "============================================================================"
endif
	@echo ""
	@echo "--- Compiling CPU/SIMD compression"
	$(Q)$(CXX) $(CXXFLAGS) -c compression/cpusimd/src/bitpacking.cpp -Icompression/cpusimd/include
	@echo "    CCX       $@"

integratedbitpacking.o: compression/cpusimd/include/integratedbitpacking.h compression/cpusimd/src/integratedbitpacking.cpp
	$(Q)$(CXX) $(CXXFLAGS) -c compression/cpusimd/src/integratedbitpacking.cpp -Icompression/cpusimd/include
	@echo "    CCX       $@"

simdpackedsearch.o:  compression/cpusimd/src/simdpackedsearch.c
	$(Q)$(CC) $(CCFLAGS) -c compression/cpusimd/src/simdpackedsearch.c
	@echo "    CC        $@"

simdpackedselect.o:  compression/cpusimd/src/simdpackedselect.c
	$(Q)$(CC) $(CCFLAGS) -c compression/cpusimd/src/simdpackedselect.c
	@echo "    CC        $@"

streamvbyte.o:  compression/cpusimd/src/streamvbyte.c
	$(Q)$(CC) $(CCFLAGS) -c compression/cpusimd/src/streamvbyte.c
	@echo "    CC        $@"

varintdecode.o:  compression/cpusimd/src/varintdecode.c
	$(Q)$(CC) $(CCFLAGS) -c compression/cpusimd/src/varintdecode.c
	@echo "    CC        $@"

simdbitpacking.o: compression/cpusimd/include/simdbitpacking.h compression/cpusimd/src/simdbitpacking.cpp
	$(Q)$(CXX) $(CXXFLAGS) -c compression/cpusimd/src/simdbitpacking.cpp -Icompression/cpusimd/include
	@echo "    CCX       $@"

usimdbitpacking.o: compression/cpusimd/include/usimdbitpacking.h compression/cpusimd/src/usimdbitpacking.cpp
	$(Q)$(CXX) $(CXXFLAGS) -c compression/cpusimd/src/usimdbitpacking.cpp -Icompression/cpusimd/include
	@echo "    CCX       $@"

frameofreference.o: compression/cpusimd/src/frameofreference.cpp $(HEADERS)
	$(Q)$(CXX) $(CXXFLAGS) -c compression/cpusimd/src/frameofreference.cpp -Icompression/cpusimd/include
	@echo "    CCX       $@"

for.o: compression/cpusimd/src/for.c $(HEADERS)
	$(Q)$(CC) $(CCFLAGS) -c compression/cpusimd/src/for.c -Icompression/cpusimd/include
	@echo "    CC        $@"

simdintegratedbitpacking.o: compression/cpusimd/include/simdintegratedbitpacking.h compression/cpusimd/include/delta.h compression/cpusimd/src/simdintegratedbitpacking.cpp
	$(Q)$(CXX) $(CXXFLAGS) -c compression/cpusimd/src/simdintegratedbitpacking.cpp -Icompression/cpusimd/include
	@echo "    CCX       $@"

utils.o: generator/utils.c generator/splittable_mrg.h generator/graph_generator.h generator/user_settings.h generator/utils.h
	$(Q)$(MPICC) -c $(MPICFLAGS) -o $@ generator/utils.c
	@echo "    MPICC     $@"

splittable_mrg.o: generator/splittable_mrg.c generator/mod_arith.h generator/user_settings.h generator/mod_arith_64bit.h generator/mod_arith_32bit.h \
		generator/splittable_mrg.h generator/mrg_transitions.c
	$(Q)$(MPICC) -c $(MPICFLAGS) -o $@ generator/splittable_mrg.c
	@echo "    MPICC     $@"

make_graph.o: generator/make_graph.c generator/graph_generator.h generator/user_settings.h generator/utils.h generator/splittable_mrg.h
	$(Q)$(MPICC) -c $(MPICFLAGS) -o $@ generator/make_graph.c
	@echo "    MPICC     $@"

graph_generator.o: generator/graph_generator.c generator/user_settings.h generator/splittable_mrg.h generator/graph_generator.h
	$(Q)$(MPICC) -c $(MPICFLAGS) -o $@ generator/graph_generator.c
	@echo "    MPICC     $@"

simplecpubfs.o: simplecpubfs.cpp simplecpubfs.h globalbfs.hh distmatrix2d.hh generator/graph_generator.h generator/user_settings.h
	$(Q)$(MPICXX) -c $(MPICXXFLAGS) -o $@ simplecpubfs.cpp
	@echo "    MPICXX    $@"

onesided.o: validate/onesided.c validate/mpi_workarounds.h validate/onesided.h generator/utils.h generator/splittable_mrg.h
	$(Q)$(MPICC) -c $(MPICFLAGS) -o $@ validate/onesided.c
	@echo "    MPICC     $@"

onesided_emul.o: validate/onesided_emul.c validate/mpi_workarounds.h validate/onesided.h generator/utils.h generator/splittable_mrg.h
	$(Q)$(MPICC) -c $(MPICFLAGS) -o $@ validate/onesided_emul.c
	@echo "    MPICC     $@"

bitlevelfunctions.o: bitlevelfunctions.h bitlevelfunctions.cpp
	$(Q)$(MPICXX) -c $(MPICXXFLAGS) -o $@ bitlevelfunctions.cpp
	@echo "    MPICXX    $@"

cpubfs_bin.o: cpubfs_bin.cpp cpubfs_bin.h globalbfs.hh distmatrix2d.hh generator/graph_generator.h generator/user_settings.h
	$(Q)$(MPICXX) -c $(MPICXXFLAGS) -o $@ cpubfs_bin.cpp
	@echo "    MPICXX    $@"

cuda_bfs_cuda.o: cuda/cuda_bfs.cu
	@echo ""
if SIMD
	@echo "============================================================================"
	@echo "== Compiling"
	@echo "============================================================================"
endif
	@echo ""
	@echo "--- Compiling CUDA/BFS"
	$(Q)$(NVCXX) $(NV_ARCHITECTURE) $(NVCC_DEBUG_FLAGS) -c -ccbin $(CXX) -Xcompiler $(NVCXXFLAGS),"`$(MPICXX) -compile_info --showme:compile | sed -e 's,-compile_info,,' -e 's,--showme:compile,,' -e 's,g++,,' -e 's,icpc,,' -e 's,-pthread,,'`" -std=c++11 --compiler-options -fno-strict-aliasing -m64 $(o_level_nvcc) -use_fast_math -I$(NVDIR)/include -I.. cuda/cuda_bfs.cu -o cuda_bfs_cuda.o
	@echo "    NVCC      $@"
	@echo ""

g500:  $(g500_OBJECTS)
	@echo ""
	@echo "============================================================================"
	@echo "== Linking"
	@echo "============================================================================"
	@echo ""
	@echo "--- Linking"
	$(Q)$(LINKER) $(LFLAGS) -o $@ $(g500_OBJECTS) $(OBJCOMP) $(LLIBS)
	echo $(LINKER) $(LFLAGS) -o $@ $(g500_OBJECTS) $(OBJCOMP) $(LLIBS)
	@echo "    MPICXX    $@"

clean-local:
	@echo ""
	@echo "============================================================================"
	@echo "== Cleaning"
	@echo "============================================================================"
	@echo ""
	$(Q)rm -rf $(OUTFILES) $(g500_OBJECTS) 2> /dev/null
	$(Q)rm -rf *~ core *.core g500 2> /dev/null
	@echo "--- Cleaning"

.PHONY: g500 clean-local
