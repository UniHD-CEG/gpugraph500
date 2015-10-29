#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

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
# scorep_profiler_enable
#   Values: yes | no
#   Default value: no
#   Description: "is scorep has been installed, enable it on code"
#
# scorep_profiler_automatic_instrument
#   Values: yes | no
#   Default value: yes
#   Description: "setting this option to no, manual instrumentation is enabled.
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
#   				either enable_simd_compression (CPU) or enable_simd_compression (GPU)
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
# enable_simd_compression
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



AC_PREREQ([2.68])
AC_INIT([graph500-bfs], [1.1])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([globalbfs.hh])
AC_CONFIG_HEADERS([config.h])


# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_CPP
AC_PROG_INSTALL
LX_FIND_MPI
AC_LANG_PUSH([C++])
LX_FIND_MPI
AC_LANG_POP([C++])
AX_CHECK_CUDA
AC_LANG_PUSH([C++])
FF_CHECK_CUDA([6.0.0])
AC_LANG_POP([C++])
AX_PTHREAD
AX_OPENMP
AX_EXT
AX_CXX_COMPILE_STDCXX_11(,[mandatory])


# LT_INIT([disable-shared], [disable-static], [no-pic], [C++])
# AC_CONFIG_MACRO_DIR([m4])
#[scorep_custom_basedir=/home/jromera/scorep]
#[scorep_profiler_manual_instrumentation=yes]
#[scorep_custom=yes]
# AM_CONDITIONAL([SCOREP_CUSTOM], [test x"$scorep_custom" = xyes])
# AM_CONDITIONAL([SCOREP_PROFILER_MANUAL_INSTRUMENTATION], [test x"$scorep_profiler_manual_instrumentation" = xyes])

# [compression_enabled=yes]
AC_ARG_ENABLE([compression],
    [AS_HELP_STRING([--enable-compression], [aa])],
    [enable_compression=$enableval],
    [enable_compression=yes])

# [nvidia_ptxas_otimize=no]
AC_ARG_ENABLE([ptxas_otimization],
    [AS_HELP_STRING([--enable-ptxas_otimization], [aa])],,
    [enable_ptxas_otimization=no])

# [manual_profiler_cuda=no]
AC_ARG_ENABLE([cuda-basic-profiling],
    [AS_HELP_STRING([--enable-cuda-basic-profiling], [aa])],,
    [enable_cuda_basic_profiling=no])

# [manual_profiler_other_compilers=yes]
AC_ARG_ENABLE([others-basic-profiling],
    [AS_HELP_STRING([--enable-others-basic-profiling], [aa])],,
    [enable_others_basic_profiling=yes])

# [openmp_on_cuda=no]
AC_ARG_ENABLE([openmp-on-cudafiles],
    [AS_HELP_STRING([--enable-openmp-on-cudafiles], [aa])],,
    [enable_openmp_on_cudafiles=no])

# [openmp_on_other_compilers=no]
AC_ARG_ENABLE([openmp-on-others],
    [AS_HELP_STRING([--enable-openmp-on-others], [aa])],,
    [enable_openmp_on_others=no])

# [scorep_profiler_enable=yes]
AC_ARG_ENABLE([scorep],
    [AS_HELP_STRING([--enable-scorep], [aa])],,
    [enable_scorep=no])

# [use_avx_instructions=yes]
AC_ARG_ENABLE([avx],
    [AS_HELP_STRING([--enable-avx], [aa])],,
    [enable_avx=yes])

# [simd_compression_enabled=yes]
AC_ARG_ENABLE([simd],
    [AS_HELP_STRING([--enable-simd], [aa])],,
    [enable_simd=yes])

# [simt_compression_enabled=no]
AC_ARG_ENABLE([simt],
    [AS_HELP_STRING([--enable-simt], [aa])],,
    [enable_simt=no])

# [enable_compression_verify=no]
AC_ARG_ENABLE([verify-compression],
    [AS_HELP_STRING([--enable-verify-compression], [aa])],,
    [enable_verify_compression=no])

# [enable_compression_debug=no]
AC_ARG_ENABLE([debug-compression],
    [AS_HELP_STRING([--enable-debug-compression], [aa])],,
    [enable_debug_compression=no])

# [use_cuda=yes]
AC_ARG_ENABLE([cuda],
    [AS_HELP_STRING([--enable-cuda], [aa])],,
    [enable_cuda=yes])

# [debug=no]
AC_ARG_ENABLE([debug],
    [AS_HELP_STRING([--enable-debug], [aa])],,
    [enable_debug=no])

# [debug_code=no]
AC_ARG_ENABLE([debugging],
    [AS_HELP_STRING([--enable-debugging], [aa])],,
    [enable_debugging=no])

# [quiet_output=no]
AC_ARG_ENABLE([quiet],
    [AS_HELP_STRING([--enable-quiet], [aa])],,
    [enable_quiet=no])

# [code_optimization_level=O4]
AC_ARG_ENABLE(code-optimization,
[AS_HELP_STRING([--enable-code-optimization], [aa])],
[case "${enableval}" in
  O0) enable_code_optimization=O0 ;;
  O1) enable_code_optimization=O1 ;;
  O2) enable_code_optimization=O2 ;;
  O3) enable_code_optimization=O3 ;;
  O4) enable_code_optimization=O4 ;;
  *) AC_MSG_ERROR(bad value ${enableval} for --enable-code-optimization) ;;
esac],[enable_code_optimization=O4])

# [nvidia_architecture=fermi]
AC_ARG_ENABLE(nvidia-architecture,
[AS_HELP_STRING([--enable-nvidia-architecture], [aa])],
[case "${enableval}" in
  fermi) enable_nvidia_architecture=fermi ;;
  kepler) enable_nvidia_architecture=kepler ;;
  auto) enable_nvidia_architecture=auto ;;
  *) AC_MSG_ERROR(bad value ${enableval} for --enable-nvidia-architecture) ;;
esac],[enable_nvidia_architecture=fermi])

# [code_optimization_flags=-funroll-loops -flto]
AC_ARG_VAR(OPTIMIZATION_FLAGS,[AS_HELP_STRING([--optimization-flags], [aa])])
AS_IF([test "x$OPTIMIZATION_FLAGS" = x], [OPTIMIZATION_FLAGS="-loops -flto"])


### declare "precious" environment variables
# (implicit AC_SUBST)

AC_ARG_VAR(MPI_INC,[MPI include flags])
AC_ARG_VAR(MPI_LIB,[MPI lib flags])
AC_ARG_VAR(ALLCFLAGS,Customized C source compilation flags)
AC_ARG_VAR(ALLCXXFLAGS,Customized C++ source compilation flags)
AC_ARG_VAR(CC,The C compiler)
AC_ARG_VAR(MPICC,The MPI C compiler)
AC_ARG_VAR(CXX,The C++ compiler)
AC_ARG_VAR(MPICXX,The MPI C++ compiler)

AM_CONDITIONAL([OPTIMIZATION_O0], [test x"$enable_code_optimization" = xO0])
AM_CONDITIONAL([OPTIMIZATION_O1], [test x"$enable_code_optimization" = xO1])
AM_CONDITIONAL([OPTIMIZATION_O2], [test x"$enable_code_optimization" = xO2])
AM_CONDITIONAL([OPTIMIZATION_O3], [test x"$enable_code_optimization" = xO3])
AM_CONDITIONAL([OPTIMIZATION_O4], [test x"$enable_code_optimization" = xO4])
AM_CONDITIONAL([AVX], [test x"$enable_avx" = xyes])
AM_CONDITIONAL([SIMD], [test x"$enable_simd" = xyes])
AM_CONDITIONAL([SIMT], [test x"$enable_simt" = xyes])
AM_CONDITIONAL([PTXAS_OTIMIZATION], [test x"$enable_ptxas_otimization" = xyes])
AM_CONDITIONAL([NVIDIA_ARCHITECTURE_FERMI], [test x"$enable_nvidia_architecture" = xfermi])
AM_CONDITIONAL([NVIDIA_ARCHITECTURE_KEPLER], [test x"$enable_nvidia_architecture" = xkepler])
AM_CONDITIONAL([SCOREP], [test x"$enable_scorep" = xyes])
AM_CONDITIONAL([OPENMP_ON_CUDA], [test x"$enable_openmp_on_cudafiles" = xyes])
AM_CONDITIONAL([OPENMP_ON_OTHERS], [test x"$enable_openmp_on_others" = xyes])
AM_CONDITIONAL([BASIC_PROFING_CUDA], [test x"$enable_cuda_basic_profiling" = xyes])
AM_CONDITIONAL([BASIC_PROFING_OTHERS], [test x"$enable_thers_basic_profiling" = xyes])
AM_CONDITIONAL([DEBUG], [test x"$enable_debug" = xyes])
AM_CONDITIONAL([DEBUGGING], [test x"$enable_debugging" = xyes])
AM_CONDITIONAL([COMPRESSION], [test x"$enable_compression" = xyes])
AM_CONDITIONAL([DEBUG_COMPRESSION], [test x"$enable_debug_compression" = xyes])
AM_CONDITIONAL([VERIFY_COMPRESSION], [test x"$enable_verify_compression" = xyes])
AM_CONDITIONAL([CUDA], [test x"$enable_cuda" = xyes])
AM_CONDITIONAL([QUIET_OUTPUT], [test x"$enable_quiet" = xyes])


# Checks for libraries.
# FIXME: Replace `main' with a function in `-lcuda':
# AC_CHECK_LIB([cuda], [main])
# FIXME: Replace `main' with a function in `-lcudart':
# AC_CHECK_LIB([cudart], [main])
# FIXME: Replace `main' with a function in `-lcudart_static':
# AC_CHECK_LIB([cudart_static], [main])
# FIXME: Replace `main' with a function in `-llikwid':
# AC_CHECK_LIB([likwid], [main])


# Checks for header files.
AC_CHECK_HEADERS([fcntl.h limits.h stddef.h sys/time.h])
# AC_CHECK_HEADERS([fcntl.h inttypes.h limits.h stddef.h stdint.h stdlib.h string.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_TYPE_INT8_T
AC_C_RESTRICT
AC_TYPE_SIZE_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T
AC_CHECK_TYPES([ptrdiff_t])

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_CHECK_FUNCS([memmove memset select])
AC_LANG_PUSH([C++])
AC_CHECK_FUNCS([pow sqrt])
AC_LANG_POP([C++])


SIMD_OBJECTS="bitpacking.o integratedbitpacking.o simdbitpacking.o usimdbitpacking.o simdintegratedbitpacking.o varintdecode.o streamvbyte.o simdpackedsearch.o simdpackedselect.o frameofreference.o for.o"
AC_SUBST([SIMD_OBJECTS])
g500_OBJECTS="main.o utils.o splittable_mrg.o make_graph.o graph_generator.o onesided.o onesided_emul.o bitlevelfunctions.o"
AC_SUBST([g500_OBJECTS])

SCOREP_CPPFLAG=
for item in `scorep-config --cppflags`; do
  SCOREP_CPPFLAG+=" $item"
done

AC_SUBST([SCOREP_CPPFLAG])
SCOREP_LLIBS=
for item in `scorep-config --libs`; do
  SCOREP_LLIBS+=" $item"
done
AC_SUBST([SCOREP_LLIBS])

SCOREP_LFLAG=
for item in `scorep-config --ldflags`; do
  SCOREP_LFLAG+=" $item"
done
AC_SUBST([SCOREP_LFLAG])

simd_H_FILES=
for item in `ls compression/cpusimd/include/*h`; do
  simd_H_FILES+=" $item"
done
AC_SUBST([simd_H_FILES])

# SCOREP_CPPFLAG=`scorep-config --cppflags`
# AC_SUBST([SCOREP_CPPFLAG])
#
# SCOREP_LLIBS=`scorep-config --libs`
# AC_SUBST([SCOREP_LLIBS])
#
# SCOREP_LFLAG=`scorep-config --ldflags`
# AC_SUBST([SCOREP_LFLAG])
#
# simd_H_FILES=`ls compression/cpusimd/include/*h`
# AC_SUBST([simd_H_FILES])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
