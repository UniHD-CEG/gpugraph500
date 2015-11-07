AC_DEFUN([AX_MPI_PRE],
[
AC_PREREQ(2.59)

# MPI root directory
AC_ARG_WITH(mpi,
[AS_HELP_STRING([--with-mpi=<path>],
        [absolute path to the MPI root directory.
         It should contain bin/ and include/ subdirectories.])])

if test x"$with_mpi" != "x";
then
  MPIROOT="$with_mpi"
fi

AC_ARG_WITH(mpicc,
[AS_HELP_STRING([--with-mpicc=mpicc],
        [name of the MPI C++ compiler to use (default mpicc)])])

if test x"$with_mpicc" != "x";
then
  MPICC="$with_mpicc"
else
  MPICC="mpicc"
fi

if test x"$with_mpi" != "x";
then
  MPICC="$with_mpi/bin/$MPICC"
fi


AC_ARG_WITH(mpicxx,
[AS_HELP_STRING([--with-mpicxx=mpicxx],
        [name of the MPI C++ compiler to use (default mpicxx)])])

if test x"$with_mpicxx" != "x";
then
  MPICXX="$with_mpicxx"
else
  MPICXX="mpicxx"
fi

if test x"$with_mpi" != "x";
then
  MPICXX="$with_mpi/bin/$MPICXX"
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
