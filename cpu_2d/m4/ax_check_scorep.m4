dnl Check for SCALASCA/SCOREP
dnl Copyright(c) 2015 uni-heidelberg.de

dnl ========LICENCE========
dnl This is free and unemcumbered software released into the public domain.
dnl For more information, see http://unlicense.org/UNLICENSE
dnl
dnl If you're unfamiliar with public domain, that means it's perfectly
dnl fine to start with this skeleton and code away, later relicensing as you see fit.
dnl ========LICENCE========
dnl/

dnl Test for SCOREP

dnl Sets SCOREP_EXEC

ac_scorep_found=no
AC_DEFUN([AX_CHECK_SCOREP], [
		[DEFAULT_EXECUTABLE_PATH=`which scorep 2> /dev/null`]
		AC_MSG_CHECKING(for SCOREP )

		AC_ARG_WITH(scorep,
			AS_HELP_STRING([--with-scorep=<path>],[
				Use SCOREP profiler.
				If argument is <empty> that means the library is reachable with the standard
				search path (set as default).
				Otherwise you give the <path> to the directory which contain the library.
				]),
			[if test "x$withval" = xyes; then
				EXECUTABLE_PATH="${DEFAULT_EXECUTABLE_PATH}"
			elif test "x$withval" != xno; then
				EXECUTABLE_PATH="$withval"
			fi],
			[EXECUTABLE_PATH="${DEFAULT_EXECUTABLE_PATH}"])


if test -f ${EXECUTABLE_PATH} -a -x ${EXECUTABLE_PATH}; then
	SCOREP_EXEC="${EXECUTABLE_PATH}"
	AC_MSG_RESULT(yes)
	ac_found_scorep=yes
else
	SCOREP_EXEC=
	AC_MSG_RESULT(no)
fi

# SCOREP_EXEC=${DEFAULT_EXECUTABLE_PATH}
AC_SUBST([SCOREP_EXEC])
unset EXECUTABLE_PATH
unset DEFAULT_EXECUTABLE_PATH
])
