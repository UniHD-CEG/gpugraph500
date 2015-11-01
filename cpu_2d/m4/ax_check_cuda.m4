#####
#
# SYNOPSIS
#
# AX_CHECK_CUDA
#
# DESCRIPTION
#
# Figures out if CUDA Driver API/nvcc is available, i.e. existence of:
# 	cuda.h
#   libcuda.so
#   nvcc
#
# If something isn't found, fails straight away.
#
# Locations of these are included in
#   CUDA_CFLAGS and
#   CUDA_LDFLAGS.
# Path to nvcc is included as
#   NVCC_PATH
# in config.h
#
# The author is personally using CUDA such that the .cu code is generated
# at runtime, so don't expect any automake magic to exist for compile time
# compilation of .cu files.
#
# LICENCE
# Public domain
#
# AUTHOR
# wili
#
#####

dnl Modified 2015 by uni-heidelberg.de
dnl Sets as default the detected PATH's version

AC_DEFUN([AX_CHECK_CUDA], [


[DEFAULT_CUDA_PATH=`which nvcc | sed -e 's,/bin/nvcc$,,' 2> /dev/null`]
if test ! -d ${DEFAULT_CUDA_PATH}; then
	[DEFAULT_CUDA_PATH="/usr/local/cuda"]
fi

# Provide your CUDA path with this
AC_ARG_WITH(cuda, AS_HELP_STRING([--with-cuda=<path>],[
				Use CUDA library.
				If argument is <empty> that means the library is reachable with the standard
				search path "/usr" or "/usr/local"  (set as default).
				Otherwise you give the <path> to the directory which contain the library.
				]),
	[cuda_prefix=$withval],
	[cuda_prefix="${DEFAULT_CUDA_PATH}"])

# Setting the prefix to the default if only --with-cuda was given
if test "x$cuda_prefix" == "xyes"; then
	if test "x$withval" == "xyes"; then
		cuda_prefix="${DEFAULT_CUDA_PATH}"
	fi
fi

# Checking for nvcc
AC_MSG_CHECKING([nvcc in $cuda_prefix/bin])
if test -f "${cuda_prefix}/bin/nvcc" -a -x "${cuda_prefix}/bin/nvcc"; then
	AC_MSG_RESULT([found])
	AC_DEFINE_UNQUOTED([NVCC_PATH], ["$cuda_prefix/bin/nvcc"], [Path to nvcc binary])
	# We need to add the CUDA search directories for header and lib searches

	CUDA_CFLAGS=""

	# Saving the current flags
	ax_save_CFLAGS="${CFLAGS}"
	ax_save_LDFLAGS="${LDFLAGS}"

	# Announcing the new variables
	AC_SUBST([CUDA_CFLAGS])
	AC_SUBST([CUDA_LDFLAGS])
	AC_SUBST([NVCC],[$cuda_prefix/bin/nvcc])
	AC_CHECK_FILE([$cuda_prefix/lib64],[lib64_found=yes],[lib64_found=no])
	if test "x$lib64_found" = xno ; then
		AC_CHECK_FILE([$cuda_prefix/lib],[lib32_found=yes],[lib32_found=no])
		if test "x$lib32_found" = xyes ; then
			AC_SUBST([CUDA_LIBDIR],[$cuda_prefix/lib])
		else
			AC_MSG_WARN([Couldn't find cuda lib directory])
			VALID_CUDA=no
		fi
	else
		AC_CHECK_SIZEOF([long])
		if test "x$ac_cv_sizeof_long" = "x8" ; then
			AC_SUBST([CUDA_LIBDIR],[$cuda_prefix/lib64])
			CUDA_CFLAGS+=" -m64"
		elif test "x$ac_cv_sizeof_long" = "x4" ; then
			AC_CHECK_FILE([$cuda_prefix/lib32],[lib32_found=yes],[lib32_found=no])
			if test "x$lib32_found" = xyes ; then
				AC_SUBST([CUDA_LIBDIR],[$cuda_prefix/lib])
				CUDA_CFLAGS+=" -m32"
			else
				AC_MSG_WARN([Couldn't find cuda lib directory])
				VALID_CUDA=no
			fi
		else
			AC_MSG_ERROR([Could not determine size of long variable type])
		fi
	fi

	if test "x$VALID_CUDA" != xno ; then
		CUDA_CFLAGS+=" -I$cuda_prefix/include"
		CFLAGS="$CUDA_CFLAGS $CFLAGS"
		CUDA_LDFLAGS="-L$CUDA_LIBDIR"
		LDFLAGS="$CUDA_LDFLAGS $LDFLAGS"

		# And the header and the lib
		AC_CHECK_HEADER([cuda.h], [],
			AC_MSG_WARN([Couldn't find cuda.h])
			VALID_CUDA=no
			,[#include <cuda.h>])
		if test "x$VALID_CUDA" != "xno" ; then
			AC_CHECK_LIB([cuda], [cuInit], [VALID_CUDA=yes], AC_MSG_FAILURE([Couldn't find libcuda]
			VALID_CUDA=no))
		fi
	fi
	# Returning to the original flags
	CFLAGS=${ax_save_CFLAGS}
	LDFLAGS=${ax_save_LDFLAGS}
else
	AC_MSG_RESULT([not found!])
	AC_MSG_WARN([nvcc was not found in $cuda_prefix/bin])
	VALID_CUDA=no
fi

if test "x$enable_cuda" = xyes && test x$VALID_CUDA = xyes ; then
	AC_MSG_NOTICE([Building with CUDA bindings])
elif test "x$enable_cuda" = xyes && test x$VALID_CUDA = xno ; then
	AC_MSG_ERROR([Cannot build CUDA bindings. Check errors])
fi

unset DEFAULT_CUDA_PATH
])
