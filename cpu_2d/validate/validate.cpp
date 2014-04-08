/* Copyright (C) 2010-2011 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */
/* modification for 2d partitioning: Matthias Hauck 2014 */
#define __STDC_LIMIT_MACROS
#include "../distmatrix2d.hh"
#include "validate.h"
#include "../generator/utils.h"
#include "onesided.h"
//#include "common.h"
#include <mpi.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* This code assumes signed shifts are arithmetic, which they are on
 * practically all modern systems but is not guaranteed by C. */

int64_t get_pred_from_pred_entry(int64_t val) {
  return (val << 16) >> 16;
}

uint16_t get_depth_from_pred_entry(int64_t val) {
  return (val >> 48) & 0xFFFF;
}




