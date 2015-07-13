/* Copyright (C) 2011 The Trustees of Indiana University.                  */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

//#include "common.h"
#include "mpi_workarounds.h"
#include "onesided.h"
#include "../generator/utils.h"
#include <mpi.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>


#include "stdio.h"

/* One-sided wrapper to allow emulation; a good MPI should be able to handle
 * the version in this file. */

