#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.


AC_PREREQ([2.68])
AC_INIT([g500], [1.1])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([globalbfs.hh])
AC_CONFIG_HEADERS([config.h])


## Checks for programs.
#
AC_PROG_CXX
AX_CXX_COMPILE_STDCXX_11(,[mandatory])
if test "x${ac_success}" = "xno"; then
    AC_MSG_FAILURE([No compiler with C++11 support was found.])
fi
AC_PROG_CC
AC_PROG_CPP
AC_PROG_INSTALL

AX_MPI_PRE

### Commandline arguments
# Use ./configure --help to view

# compression
#   Values: yes | no
#   Default value: yes
AC_ARG_ENABLE([compression],
    AS_HELP_STRING([--enable-compression],
      [Enabled data compression through the network.
      This option is available only when --enable-cuda is active (default).
      Enabled by default.]),
    [enable_compression=$enableval],
    [enable_compression=yes])

# ptxas_otimization
#   Values: yes | no
#   Default value: no
AC_ARG_ENABLE([ptxas_otimization],
    AS_HELP_STRING([--enable-ptxas_otimization],
      [Selects whether PTXAS will be optimized or not.
      This option will only be used if --enable-cuda is present (default).
      The default PTXAS optimization is -O3.
      Disabled by default.]),,
    [enable_ptxas_otimization=no])

# bfs-basic-profiling
#   Values: yes | no
#   Default value: no
AC_ARG_ENABLE([bfs-basic-profiling],
    AS_HELP_STRING([--enable-bfs-basic-profiling],
      [Displays statistics for each BFS run.
      Enabled by default.]),,
    [enable_bfs_basic_profiling=no])

# other-basic-profiling
#   Values: yes | no
#   Default value: no
AC_ARG_ENABLE([other-basic-profiling],
    AS_HELP_STRING([--enable-other-basic-profiling],
      [Displays graph500 default statistics.
      Enabled by default.]),,
    [enable_other_basic_profiling=yes])

# openmp-on-cudafiles
#   Values: yes | no
#   Default value: no
AC_ARG_ENABLE([openmp-on-cudafiles],
    AS_HELP_STRING([--enable-openmp-on-cudafiles],
      [Selects whether or not OpenMP will be enabled.
      This option applies only to CUDA files.
      Disabled by default.]),,
    [enable_openmp_on_cudafiles=no])

# openmp-on-others
#   Values: yes | no
#   Default value: no
AC_ARG_ENABLE([openmp-on-others],
    AS_HELP_STRING([--enable-openmp-on-others],
      [Selects whether or not OpenMP will be enabled.
      This option applies to any non-CUDA file.
      Disabled by default.]),,
    [enable_openmp_on_others=no])

# scorep
#   Values: yes | no
#   Default value: no
AC_ARG_ENABLE([scorep],
    AS_HELP_STRING([--enable-scorep],
      [Enable instrumentation with Scalasca/ScoreP.
      Disabled by default.]),,
    [enable_scorep=no])

# avx
#   Values: yes | no
#   Default value: yes
AC_ARG_ENABLE([avx],
    AS_HELP_STRING([--enable-avx],
      [Whether or not use AVX instruction set.
      SSE2 will be used instead if --enable-compression is enabled (default),
      and either this option is disabled (--disable-avx), or AVX instructions
      are not not supported by hardware.
      Enabled by default.]),,
    [enable_avx=yes])

# enable_simd
#   Values: yes | no
#   Default value: yes
#   External links:
#         - https://github.com/lemire/SIMDCompressionAndIntersection
AC_ARG_ENABLE([simd],
    AS_HELP_STRING([--enable-simd],
      [MPI packets will be sent compressed using the SIMDCompression library
      This option is only active if --enable-compression is selected.
      Enabled by default.]),,
    [enable_simd=yes])

# enable_simt
#   Values: yes | no
#   Default value: yes
AC_ARG_ENABLE([simt],
    AS_HELP_STRING([--enable-simt],
      [Use CUDA implementation of compression.
      Not implemented yet.
      Disabled by default.]),,
    [enable_simt=no])

# verify-compression
#   Values: yes | no
#   Default value: no
AC_ARG_ENABLE([verify-compression],
    AS_HELP_STRING([--enable-verify-compression],
      [Sends both compressed and decompressed data through
      the network. Checks decompression after transmission.
      Disabled by default.]),,
    [enable_verify_compression=no])

# debug-compression
#   Values: yes | no
#   Default value: no
AC_ARG_ENABLE([debug-compression],
    AS_HELP_STRING([--enable-debug-compression],
      [Shows statistics of compression rate, time of compression, codec, ETC.
      Disabled by default.]),,
    [enable_debug_compression=no])

# cuda
#   Values: yes | no
#   Default value: yes
AC_ARG_ENABLE([cuda],
    AS_HELP_STRING([--enable-cuda],
      [Use CUDA implementation of the BFS runs.
      Requires hardware support.
      Enabled by default.]),,
    [enable_cuda=yes])

# debug
#   Values: yes | no
#   Default value:
AC_ARG_ENABLE([debug],
    AS_HELP_STRING([--enable-debug],
      [Provides extra traces of runtime variables.
      Disabled by default.]),,
    [enable_debug=no])

# debugging
#   Values: yes | no
#   Default value: no
AC_ARG_ENABLE([debugging],
    AS_HELP_STRING([--enable-debugging],
      [Enables -g option on compiler (debugging).
      when enabled, --enable-code-optimization will be set to O0 and
      --optimization-flags will be set to '' (empty).
      Disabled by default.]),,
    [enable_debugging=no])

# quiet_output
#   Values: yes | no
#   Default value: no
#   Description: ""
AC_ARG_ENABLE([quiet],
    AS_HELP_STRING([--enable-quiet],
      [Disable compile mensages when running make.]),,
    [enable_quiet=no])

# code-optimization
#   Values: O4 | O3 | O2 | O1 | O0
#   Default value:
AC_ARG_ENABLE(code-optimization,
AS_HELP_STRING([--enable-code-optimization=O0|O1|O2|O3|O4],
  [Compiler optimization level. O0=none, O4=maximun
  Optimization O0 disables code optimization.
  Optimization O4 equals to -O3 -ffast-math.
  Default optimization is O4. Default optimization may be safe on most compilers.]),
[case "${enableval}" in
  O0) enable_code_optimization=O0 ;;
  O1) enable_code_optimization=O1 ;;
  O2) enable_code_optimization=O2 ;;
  O3) enable_code_optimization=O3 ;;
  O4) enable_code_optimization=O4 ;;
  *) AC_MSG_ERROR(bad value ${enableval} for --enable-code-optimization) ;;
esac],[enable_code_optimization=O4])

# nvidia_architecture
#   Values: fermi | kepler | auto
#   Default value: fermi
AC_ARG_ENABLE(nvidia-architecture,
AS_HELP_STRING([--enable-nvidia-architecture=fermi|kepler|auto],
  [Selects the NVIDIA target architecture. Requires --enable-cuda to be selected (default).
  Default option is 'fermi']),
[case "${enableval}" in
  fermi) enable_nvidia_architecture=fermi ;;
  kepler) enable_nvidia_architecture=kepler ;;
  auto) enable_nvidia_architecture=auto ;;
  *) AC_MSG_ERROR(bad value ${enableval} for --enable-nvidia-architecture) ;;
esac],[enable_nvidia_architecture=fermi])

# code_optimization_flags
#    Values: -funroll-loops -flto -finline-functions -funswitch-loops -fpredictive-commoning
#            -fgcse-after-reload -ftree-vectorize -ftree-partial-pre -fipa-cp-clone ...
#
#    Default value: -funroll-loops -flto (For GNU's version of Makefile)
#    Format: value value value (separated by spaces)
#    External links:
#         - GNU compiler: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#Optimize-Options
AC_ARG_VAR(OPTIMIZATION_FLAGS,AS_HELP_STRING([OPTIMIZATION_FLAGS="-flag1 -flag2 ETC" (separated by spaces).],
  [Optimization flags added to the compiler.
  Default values are safe on most compilers.
  When the option --enable-debugging is active, this flags are disabled.
  Default is -loops -flto.]))
AS_IF([test "x$OPTIMIZATION_FLAGS" = x], [OPTIMIZATION_FLAGS="-loops -flto"])
AS_IF([test "x$enable_debugging" = xyes], [OPTIMIZATION_FLAGS=""])


### declare "precious" environment variables
# (implicit AC_SUBST)

AC_ARG_VAR(MPI_INC,[MPI include flags])
AC_ARG_VAR(MPI_LIB,[MPI lib flags])
AC_ARG_VAR(ALLCFLAGS,Customized C source compilation flags)
AC_ARG_VAR(ALLCXXFLAGS,Customized C++ source compilation flags)
AC_ARG_VAR(CC,The C compiler)
# AC_ARG_VAR(MPICC,The MPI C compiler)
AC_ARG_VAR(CXX,The C++ compiler)
# AC_ARG_VAR(MPICXX,The MPI C++ compiler)

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
AM_CONDITIONAL([BASIC_PROFING_BFS], [test x"$enable_bfs_basic_profiling" = xyes])
AM_CONDITIONAL([BASIC_PROFING_OTHER], [test x"$enable_other_basic_profiling" = xyes])
AM_CONDITIONAL([DEBUG], [test x"$enable_debug" = xyes])
AM_CONDITIONAL([DEBUGGING], [test x"$enable_debugging" = xyes])
AM_CONDITIONAL([COMPRESSION], [test x"$enable_compression" = xyes])
AM_CONDITIONAL([DEBUG_COMPRESSION], [test x"$enable_debug_compression" = xyes])
AM_CONDITIONAL([VERIFY_COMPRESSION], [test x"$enable_verify_compression" = xyes])
AM_CONDITIONAL([CUDA], [test x"$enable_cuda" = xyes])
AM_CONDITIONAL([QUIET_OUTPUT], [test x"$enable_quiet" = xyes])


### Checks for header files.
#

AC_CHECK_HEADERS([fcntl.h limits.h stddef.h sys/time.h])

### Checks for typedefs, structures, and compiler characteristics.
#

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

### Checks for library functions.
#

AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_CHECK_FUNCS([memmove memset select])
AC_LANG_PUSH([C++])
AC_CHECK_FUNCS([pow sqrt])
AC_LANG_POP([C++])

### declare environment variables
# (éxplicit AC_SUBST)

SIMD_OBJECTS="bitpacking.o integratedbitpacking.o simdbitpacking.o usimdbitpacking.o simdintegratedbitpacking.o varintdecode.o streamvbyte.o simdpackedsearch.o simdpackedselect.o frameofreference.o for.o"
g500_OBJECTS="main.o utils.o splittable_mrg.o make_graph.o graph_generator.o onesided.o onesided_emul.o bitlevelfunctions.o"

SCOREP_CPPFLAG=
for item in `scorep-config --cppflags 2> /dev/null`; do
  SCOREP_CPPFLAG+=" $item"
done

SCOREP_LLIBS=
for item in `scorep-config --libs 2> /dev/null`; do
  SCOREP_LLIBS+=" $item"
done

SCOREP_LFLAG=
for item in `scorep-config --ldflags 2> /dev/null`; do
  SCOREP_LFLAG+=" $item"
done

simd_H_FILES=
for item in `ls compression/cpusimd/include/*h 2> /dev/null`; do
  simd_H_FILES+=" $item"
done

AC_SUBST([SIMD_OBJECTS])
AC_SUBST([g500_OBJECTS])
AC_SUBST([SCOREP_CPPFLAG])
AC_SUBST([SCOREP_LLIBS])
AC_SUBST([SCOREP_LFLAG])
AC_SUBST([simd_H_FILES])

### Check for required libraries.
#

# ------------------------------------------------------------------------------
# Setup MPI
# ------------------------------------------------------------------------------

LX_FIND_MPI
AC_LANG_PUSH([C++])
LX_FIND_MPI
AC_LANG_POP([C++])

AX_MPI_POST

#AC_ARG_WITH(mpi, AS_HELP_STRING([--with-mpi=<path>],
#    [compile with MPI (parallelization) support.
#    Use <path> to specify the route to MPI.
#    By Default PATH is used]),,
#    [with_mpi=auto])
#AX_PROG_CXX_MPI([test x"$with_mpi" != xno],[use_mpi=yes],[
#    use_mpi=no
#    AC_MSG_FAILURE([MPI compiler required, but could not use MPI.])
#])

# ------------------------------------------------------------------------------
# Setup CUDA paths
# ------------------------------------------------------------------------------

if test "x$enable_cuda" = xyes; then
  AX_CHECK_CUDA
  if test "x$NVCC" = xno; then
    AC_MSG_FAILURE([NVIDIA CUDA nvcc compiler not found or CUDA support disabled. Check the $PATH variable to CUDA or specify --with-cuda=/path/to/cuda])
  fi
  CIT_CUDA_CONFIG
  # AX_CUDA
fi

# ------------------------------------------------------------------------------
# Setup AVX
# ------------------------------------------------------------------------------

AM_CONDITIONAL(HAVE_AVX,false)
if test  "x$enable_avx" = xyes -o "x$enable_compression" = xyes; then
  AX_EXT
  AC_CHECK_HEADERS([cpuid.h])
  if test "x$ax_cv_have_avx_ext" = xyes; then
    AM_CONDITIONAL(HAVE_AVX,true)
  else
    if test  "x$enable_compression" = xyes -a "x$enable_simd" = xyes ; then
      AC_MSG_FAILURE(AVX support is needed for SIMD compression)
    fi
  fi
fi

# FIXME: if compression SSE4 req
# FIXME: if not --enable-avx select best ssex extension

# ------------------------------------------------------------------------------
# Setup OpenCL
# ------------------------------------------------------------------------------

AC_ARG_WITH(opencl,
    AS_HELP_STRING([--with-opencl=<path>],[prefix to location of OpenCL include directory @<:@default=auto@:>@]),,
    [with_opencl=auto])

OPENCL_CFLAGS=
OPENCL_LIBS=

if test "x$with_opencl" != "xno"
then
   OPENCL_CFLAGS+=" -D FW_OPENCL"
   OPENCL_LIBS+=" -lOpenCL"
   if test "x$with_opencl" != "xauto"
   then
      OPENCL_CFLAGS+=" -I$with_opencl/include"
   else
      case $host_os in
      *darwin*)
        AC_CHECK_FILE(/System/Library/Frameworks/OpenCL.framework,,[with_opencl="no"])
        ;;
      *)
        AC_CHECK_FILE(/usr/include/CL,[OPENCL_CFLAGS+=" -I/usr/include"],[with_opencl="no"])
      esac
   fi
fi

AM_CONDITIONAL(BUILDOPENCL, test ! x$with_opencl = xno)
AX_CHECK_CL

# ------------------------------------------------------------------------------
# Setup PTHREADS
# ------------------------------------------------------------------------------

AX_PTHREAD

# ------------------------------------------------------------------------------
# Setup OPENMP
# ------------------------------------------------------------------------------

if test x$GCC = xyes
then
  AX_GCC_VERSION()
fi

AS_IF([test "x$enable_openmp_on_cudafiles" = "xyes" -o "x$enable_openmp_on_others" = "xyes"], [
  if test x$GCC = xyes
  then
    AX_COMPARE_VERSION([$GCC_VERSION], [ge], [4.2.0], [
      AX_OPENMP([AM_CONDITIONAL(HAVE_OPENMP,true)],
      [AM_CONDITIONAL(HAVE_OPENMP,false)])
    ],[])
  else
    AX_OPENMP([AM_CONDITIONAL(HAVE_OPENMP,true)],
    [AM_CONDITIONAL(HAVE_OPENMP,false)])
  fi
],[
AM_CONDITIONAL(HAVE_OPENMP, test true = false)
])
AC_SUBST(OPENMP_CFLAGS)

dnl OpenMP checker only defines for C when compiling both C and C++
OPENMP_CXXFLAGS=$OPENMP_CFLAGS
AC_SUBST(OPENMP_CXXFLAGS)

# ------------------------------------------------------------------------------
# Scalasca / ScoreP support
# ------------------------------------------------------------------------------

AX_CHECK_SCOREP

# ------------------------------------------------------------------------------
# Doxygen support
# ------------------------------------------------------------------------------

DX_HTML_FEATURE(ON)
DX_CHM_FEATURE(OFF)
DX_CHI_FEATURE(OFF)
DX_MAN_FEATURE(OFF)
DX_RTF_FEATURE(OFF)
DX_XML_FEATURE(OFF)
DX_PDF_FEATURE(OFF)
DX_PS_FEATURE(OFF)
DX_INIT_DOXYGEN([$PACKAGE_NAME],[g500.dox],[doc])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
