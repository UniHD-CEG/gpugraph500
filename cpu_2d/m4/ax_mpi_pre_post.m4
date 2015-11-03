AC_DEFUN([AX_MPI_PRE],
[
AC_PREREQ(2.59)

# MPI root directory
AC_ARG_WITH(mpi_root,
[AS_HELP_STRING([--with-mpi-root=MPIROOT],
        [absolute path to the MPI root directory])])

if test x"$with_mpi_root" != "x";
then
  MPIROOT="$with_mpi_root"
fi

AC_ARG_WITH(mpicc,
[AS_HELP_STRING([--with-mpicc=MPICC],
        [name of the MPI C++ compiler to use (default mpicc)])])

if test x"$with_mpicc" != "x";
then
  MPICC="$with_mpicc"
else
  MPICC="mpicc"
fi

if test x"$with_mpi_root" != "x";
then
  MPICC="$with_mpi_root/bin/$MPICC"
fi


AC_ARG_WITH(mpicxx,
[AS_HELP_STRING([--with-mpicxx=MPICXX],
        [name of the MPI C++ compiler to use (default mpicxx)])])

if test x"$with_mpicxx" != "x";
then
  MPICXX="$with_mpicxx"
else
  MPICXX="mpicxx"
fi

if test x"$with_mpi_root" != "x";
then
  MPICXX="$with_mpi_root/bin/$MPICXX"
fi

# saveCC="$CC"
# saveCXX="$CXX"
# AC_SUBST(saveCC)
# AC_SUBST(saveCXX)
# CC="$MPICC"
# CXX="$MPICXX"

])




AC_DEFUN([AX_MPI_POST],
[
AC_PREREQ(2.59)

saveCC="$CC"
saveCXX="$CXX"
# AC_SUBST(saveCC)
# AC_SUBST(saveCXX)

CC="$MPICC"
CXX="$MPICXX"

AC_MSG_CHECKING(whether to use 32-bit or 64-bit locations)
AC_ARG_ENABLE(ulong,[AS_HELP_STRING([--enable-ulong],
  [enable 64-bit locations, only available on 64-bit systems. Default is 32-bit])],
  [use_ulong=$enableval],[use_ulong=no])

if test x$use_ulong = xyes -a 0$ac_cv_sizeof_void_p -ge 8 ; then
  UINT_TYPE=uint64_t
  AC_MSG_RESULT(64-bit)
else
  UINT_TYPE=uint32_t
  AC_MSG_RESULT(32-bit)
fi
AC_SUBST(UINT_TYPE)

AC_MSG_CHECKING(whether to use single or double precision)
AC_ARG_ENABLE(double,[AS_HELP_STRING([--disable-double],
  [disable double precision arithmetic [untested, default=double is enabled]])],
  [use_double=$enableval],[use_double=yes])

if test x$use_double = xno ; then
  REAL_TYPE=float
  echo
  echo -n "using ${REAL_TYPE} is not well tested, please report bugs if you find any..."
else
  REAL_TYPE=double
fi
AC_MSG_RESULT($REAL_TYPE)
AC_SUBST(REAL_TYPE)

AC_LANG_PUSH([C])
AC_MSG_CHECKING([Linking of MPI C programs])
AC_LINK_IFELSE([AC_LANG_PROGRAM([#include <mpi.h>],
             [MPI_Init(0,0)])],
             [AC_MSG_RESULT([ok])],
             [AC_MSG_RESULT([no])
             AC_MSG_FAILURE([MPI C compiler is required by $PACKAGE])])
AC_LANG_POP([C])

AC_LANG_PUSH([C++])
AC_MSG_CHECKING([Linking of MPI C++ programs])
AC_LINK_IFELSE([AC_LANG_PROGRAM([#include <mpi.h>],
             [MPI_Init(0,0)])],
             [AC_MSG_RESULT([ok])],
             [AC_MSG_RESULT([no])
             AC_MSG_FAILURE([MPI C++ compiler is required by $PACKAGE])])
AC_LANG_POP([C++])

CC="$saveCC"
CXX="$saveCXX"

])
