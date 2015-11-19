dnl  Process this file with autoconf to produce a configure script.

define(G500_COPYRIGHT,[[

Copyright 2015, University of Heidelberg

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

]])

AC_COPYRIGHT(G500_COPYRIGHT)
AH_TOP(/*G500_COPYRIGHT*/)

# Process this file with autoconf to produce a configure script.


AC_PREREQ([2.68])
AC_INIT([g500], [1.1])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([globalbfs.hh])
AC_CONFIG_HEADERS([config.h])
AM_ACLOCAL_INCLUDE(m4)

## User Redefine compilers.
#

AC_ARG_VAR(CC, [The C compiler])
AC_ARG_VAR(CXX, [The C++ compiler])

## Checks for programs.
#

AC_PROG_CXX
AX_CXX_COMPILE_STDCXX_11(,[mandatory])
AC_MSG_CHECKING([required -std=c++11 support])
C11=
if test "x${ac_success}" = "xno"; then
    AC_MSG_RESULT([no])
    AC_MSG_FAILURE([No compiler with C++11 support was found.])
else
    C11="-std=c++11"
    CXX="$CXX $C11"
    AC_MSG_RESULT([yes])
fi
AC_SUBST(C11)
AC_PROG_CC
AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_CC_STDC
c99=`echo $CC | grep "99"`
C99=
if test "x${c99}" = x; then
    c99=no
else
    c99=yes
    C99="-std=c99"
fi
AC_SUBST(C99)
AX_MPI_PRE

### declare "precious" environment variables
#

#   Values: yes | no
#   Default value: no
AC_ARG_ENABLE([bfs-basic-profiling],
    AS_HELP_STRING([--enable-bfs-basic-profiling],
      [Instrumentation. Displays statistics for each BFS run.
      Enabled by default.]),,
    [enable_bfs_basic_profiling=no])

#   Values: yes | no
#   Default value: no
AC_ARG_ENABLE([other-basic-profiling],
    AS_HELP_STRING([--enable-other-basic-profiling],
      [Instrumentation. Displays graph500 default statistics.
      Enabled by default.]),,
    [enable_other_basic_profiling=yes])

#   Values: yes | no
#   Default value: no
AC_ARG_ENABLE([scorep],
    AS_HELP_STRING([--enable-scorep],
      [Instrumentation. Enable instrumentation with Scalasca/ScoreP.
      Disabled by default.]),,
    [enable_scorep=no])

#   Values: yes | no
#   Default value: yes
AC_ARG_ENABLE([compression],
    AS_HELP_STRING([--enable-compression],
      [Compression. Enabled data compression through the network.
      This option is available only when --enable-cuda is active (default).
      Enabled by default.]),
    [enable_compression=$enableval],
    [enable_compression=yes])

#   Values: yes | no
#   Default value: yes
#   External links:
#     https://github.com/lemire/SIMDCompressionAndIntersection
AC_ARG_ENABLE([simd],
    AS_HELP_STRING([--enable-simd],
      [Compression. MPI packets will be sent compressed using the PFOR-delta SIMDCompression library.
      This option is only active if --enable-compression is selected.
      This option will be enabled by default if --enable-compression is active
      and no compression method is selected.]),,
    [enable_simd=no])

#   Values: yes | no
#   Default value: yes
AC_ARG_ENABLE([simt],
    AS_HELP_STRING([--enable-simt],
      [Compression. Use CUDA implementation of compression.
      Not implemented yet.
      Disabled by default.]),,
    [enable_simt=no])

#   Values: yes | no
#   Default value: no
AC_ARG_ENABLE([debug-compression],
    AS_HELP_STRING([--enable-debug-compression],
      [Compression. Shows statistics of compression rate, time of compression, codec, ETC.
      Disabled by default.]),,
    [enable_debug_compression=no])

#   Values: yes | no
#   Default value: no
AC_ARG_ENABLE([verify-compression],
    AS_HELP_STRING([--enable-verify-compression],
      [Compression. Sends both compressed and decompressed data through
      the network. Checks decompression after transmission.
      Disabled by default.]),,
    [enable_verify_compression=no])

#   Values: yes | no
#   Default value: no
AC_ARG_ENABLE([cuda-openmp],
    AS_HELP_STRING([--enable-cuda-openmp],
      [Mode. Selects whether or not OpenMP will be enabled.
      This option applies only to CUDA files.
      Disabled by default.]),,
    [enable_cuda_openmp=no])

#   Values: yes | no
#   Default value: no
AC_ARG_ENABLE([general-openmp],
    AS_HELP_STRING([--enable-general-openmp],
      [Mode. Selects whether or not OpenMP will be enabled.
      This option applies to any non-CUDA file.
      Disabled by default.]),,
    [enable_general_openmp=no])

#   Values: yes | no
#   Default value: yes
AC_ARG_ENABLE([cuda],
    AS_HELP_STRING([--enable-cuda],
      [Mode. Use CUDA implementation of the BFS runs.
      Requires hardware support.
      Enabled by default.]),,
    [enable_cuda=yes])

#   Values: yes | no
#   Default value: no
AC_ARG_ENABLE([ptxas_otimization],
    AS_HELP_STRING([--enable-ptxas_otimization],
      [Mode. Selects whether PTXAS will be optimized or not.
      This option will only be used if --enable-cuda is present (default).
      The default PTXAS optimization is -O3.
      Disabled by default.]),,
    [enable_ptxas_otimization=no])

#   Values: fermi | kepler | auto
#   Default value: fermi
AC_ARG_ENABLE(nvidia-architecture,
AS_HELP_STRING([--enable-nvidia-architecture=<fermi|kepler|auto>],
      [Mode. Selects the NVIDIA target architecture. Requires --enable-cuda to be selected (default).
      Default option is 'fermi']),
[case "${enableval}" in
      fermi) enable_nvidia_architecture=fermi ;;
      kepler) enable_nvidia_architecture=kepler ;;
      auto) enable_nvidia_architecture=auto ;;
      *) AC_MSG_ERROR(bad value ${enableval} for --enable-nvidia-architecture) ;;
esac],[enable_nvidia_architecture=fermi])

#   Values: yes | no
#   Default value:
AC_ARG_ENABLE([debug],
    AS_HELP_STRING([--enable-debug],
      [Debug. Provides extra traces of runtime variables.
      Disabled by default.]),,
    [enable_debug=no])

#   Values: yes | no
#   Default value: no
AC_ARG_ENABLE([debugging],
    AS_HELP_STRING([--enable-debugging],
      [Debug. Enables -g option on compiler (debugging).
      when enabled, --enable-code-optimization will be set to O0 and
      --optimization-flags will be set to '' (empty).
      Disabled by default.]),,
    [enable_debugging=no])

#   Values: yes | no
#   Default value: no
AC_ARG_ENABLE([quiet],
    AS_HELP_STRING([--enable-quiet],
      [Debug. Disable compile mensages when running make.]),,
    [enable_quiet=no])

### preprocess options
#

if test x"$enable_compression" = xyes -a x"$enable_simd" = xno; then
  if test x"$enable_simt" = xno; then
    [enable_simd=yes]
  fi
  :
fi

### declare conditionals
#

AM_CONDITIONAL([SIMD], [test x"$enable_simd" = xyes])
AM_CONDITIONAL([SIMT], [test x"$enable_simt" = xyes])
AM_CONDITIONAL([PTXAS_OTIMIZATION], [test x"$enable_ptxas_otimization" = xyes])
AM_CONDITIONAL([NVIDIA_ARCHITECTURE_FERMI], [test x"$enable_nvidia_architecture" = xfermi])
AM_CONDITIONAL([NVIDIA_ARCHITECTURE_KEPLER], [test x"$enable_nvidia_architecture" = xkepler])
AM_CONDITIONAL([SCOREP], [test x"$enable_scorep" = xyes])
AM_CONDITIONAL([OMP_CUDA], [test x"$enable_cuda_openmp" = xyes])
AM_CONDITIONAL([OMP_GENERAL], [test x"$enable_general_openmp" = xyes])
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
#

SIMD_OBJECTS="bitpacking.o integratedbitpacking.o simdbitpacking.o usimdbitpacking.o simdintegratedbitpacking.o varintdecode.o streamvbyte.o simdpackedsearch.o simdpackedselect.o frameofreference.o for.o"
SIMT_OBJECTS="cudacompress.o"
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
AC_SUBST([SIMT_OBJECTS])
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

# ------------------------------------------------------------------------------
# Checks Extoll
# ------------------------------------------------------------------------------

EXTOLL=no
AC_MSG_CHECKING([whether mpi has Extoll support])
omp_extoll=`ompi_info |grep -i mca | grep extoll 2> /dev/null`
if test x"$omp_extoll" = x; then
    AC_MSG_RESULT([no])
else
    AC_MSG_RESULT([yes])
    EXTOLL=yes
fi
AC_SUBST([EXTOLL])

# ------------------------------------------------------------------------------
# Checks CUDA
# ------------------------------------------------------------------------------

if test "x$enable_cuda" = xyes; then
  AX_CHECK_CUDA
  if test "x$NVCC" = xno; then
    AC_MSG_FAILURE([NVIDIA CUDA nvcc compiler not found or CUDA support disabled. Check the $PATH variable to CUDA or specify --with-cuda=/path/to/cuda])
  fi
  CIT_CUDA_CONFIG
  AX_CUDA
  GENCODE_SM20="-gencode=arch=compute_20,code=\"sm_20,compute_20\""
  GENCODE_SM30="-gencode=arch=compute_30,code=\"sm_30,compute_30\""
  GENCODE_SM35="-gencode=arch=compute_35,code=\"sm_35,compute_35\""
  GENCODE_SM50="-gencode=arch=compute_50,code=\"sm_50,compute_50\""
  CUDA_AUTOFLAGS="$GENCODE_SM20 $GENCODE_SM30 $GENCODE_SM35 $GENCODE_SM50"
  AC_SUBST([CUDA_AUTOFLAGS])
fi

# ------------------------------------------------------------------------------
# Checks SSE4.1
# ------------------------------------------------------------------------------

if test "x$enable_compression" = xyes; then
  AX_EXT
  AC_CHECK_HEADERS([cpuid.h])
  AC_MSG_CHECKING([whether mandatory -msse4.1 flag is supported])
  if test x"$enable_simd" = xyes; then
      ax_have_minimun_sse4_ext=`echo $SIMD_FLAGS | grep -i "sse4"`
      if test "x$ax_have_minimun_sse4_ext" = x; then
          AC_MSG_RESULT([no])
          AC_MSG_FAILURE(SSE4.1 support or greater is required for SIMD compression)
      else
          AC_MSG_RESULT([yes])
      fi
  fi
fi

# ------------------------------------------------------------------------------
# Checks FLAGS SUPPORT
# ------------------------------------------------------------------------------

### Checks latest architecture flag
#

mflag=
AC_MSG_CHECKING([for the latest architecture flag supported])
mflag=`echo $SIMD_FLAGS | awk '{ printf $NF; }'`
AC_MSG_RESULT([$mflag])
CC_MFLAG=
AX_CHECK_COMPILE_FLAG($mflag,
  [CC_MFLAG=$mflag], [])
AC_SUBST([CC_MFLAG])
CXX_MFLAG=
AC_LANG_PUSH([C++])
AX_CHECK_COMPILE_FLAG($mflag,
  [CXX_MFLAG=$mflag], [])
AC_LANG_POP([C++])
AC_SUBST([CXX_MFLAG])

### Checks machine architecture
#

CC_MBITS=
AC_CHECK_SIZEOF([size_t])
mbits=$((ac_cv_sizeof_size_t * 8))
mbits_flag="-m${mbits}"
AX_CHECK_COMPILE_FLAG($mbits_flag,
  [CC_MBITS=$mbits_flag], [])
AC_SUBST([CC_MBITS])
CXX_MBITS=
AC_LANG_PUSH([C++])
AX_CHECK_COMPILE_FLAG($mbits_flag,
  [CXX_MBITS=$mbits_flag], [])
AC_LANG_POP([C++])
AC_SUBST([CXX_MBITS])

AC_C_FIND_ENDIAN
AX_CC_MAXOPT
AX_CXX_MAXOPT
AX_MPICXX_MAXOPT
AX_CC_WARNINGS([maximum], [-pedantic -Wextra])
AX_CXX_WARNINGS([maximum], [-pedantic -Wextra -Weffc++])
WARN_MPICXX="-pedantic -Wall -Wextra"
WARN_MPICC="-pedantic -Wall -Wextra"
AC_SUBST([WARN_MPICXX])
AC_SUBST([WARN_MPICC])

IPO_LDFLAG=
hasFLTO=`echo $CXXFLAGS | grep -i "\-flto"`
hasIPO=`echo $CXXFLAGS | grep -i "\-flto"`
if test x"$hasFLTO" != x ; then
  IPO_LDFLAG=-flto
elif test x"$hasIPO" != x ; then
  IPO_LDFLAG=-ipo
fi
AC_SUBST([IPO_LDFLAG])

# ------------------------------------------------------------------------------
# Checks OpenCL
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
# Checks PTHREADS
# ------------------------------------------------------------------------------

AX_PTHREAD

# ------------------------------------------------------------------------------
# Checks OPENMP
# ------------------------------------------------------------------------------

AX_OPENMP
if test x$GCC = xyes
then
  AX_GCC_VERSION()
fi
AS_IF([test "x$enable_cuda_openmp" = "xyes" -o "x$enable_general_openmp" = "xyes"], [
  if test x$c99 = xno; then
	    AC_MSG_WARN([-stc=c99 is recommended with openMP])
  fi
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
OPENMP_CXXFLAGS=$OPENMP_CFLAGS
AC_SUBST(OPENMP_CXXFLAGS)

# ------------------------------------------------------------------------------
# Checks Scalasca / ScoreP support
# ------------------------------------------------------------------------------

AX_CHECK_SCOREP
if test x"$enable_scorep" = xyes -a x"$ac_found_scorep" = xno; then
	AC_MSG_FAILURE([Scalasca/scorep requered but not found])
fi

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
