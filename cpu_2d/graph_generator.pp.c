/* Copyright (C) 2009-2010 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#include <stdlib.h>
#include <stdint.h>
#include <assert.h>




#include <inttypes.h>

#include "user_settings.h"
#include "splittable_mrg.h"
#include "graph_generator.h"

/* Initiator settings: for faster random number generation, the initiator
 * probabilities are defined as fractions (a = INITIATOR_A_NUMERATOR /
 * INITIATOR_DENOMINATOR, b = c = INITIATOR_BC_NUMERATOR /
 * INITIATOR_DENOMINATOR, d = 1 - a - b - c. */




/* If this macro is defined to a non-zero value, use SPK_NOISE_LEVEL /
 * INITIATOR_DENOMINATOR as the noise parameter to use in introducing noise
 * into the graph parameters.  The approach used is from "A Hitchhiker's Guide
 * to Choosing Parameters of Stochastic Kronecker Graphs" by C. Seshadhri, Ali
 * Pinar, and Tamara G. Kolda (http://arxiv.org/abs/1102.5046v1), except that
 * the adjustment here is chosen based on the current level being processed
 * rather than being chosen randomly. */

/* #define SPK_NOISE_LEVEL 1000 -- in INITIATOR_DENOMINATOR units */

static int generate_4way_bernoulli(mrg_state* st, int level, int nlevels) {
  /* Generator a pseudorandom number in the range [0, INITIATOR_DENOMINATOR)
   * without modulo bias. */
  static  uint32_t limit = (0xFFFFFFFF ## U % 10000);
  uint32_t val = mrg_get_uint_orig(st);
  if (/* Unlikely */ val < limit) {
    do {
      val = mrg_get_uint_orig(st);
    } while (val < limit);
  }

  int spk_noise_factor = 0;



  int adjusted_bc_numerator = 1900 + spk_noise_factor;
  val %= 10000;
  if (val < adjusted_bc_numerator) return 1;
  val -= adjusted_bc_numerator;
  if (val < adjusted_bc_numerator) return 2;
  val -= adjusted_bc_numerator;

  if (val < 5700) return 0;



  return 3;
}

/* Reverse bits in a number; this should be optimized for performance
 * (including using bit- or byte-reverse intrinsics if your platform has them).
 * */
static inline uint64_t bitreverse(uint64_t x) {




  /* 64-bit code */

  x = __builtin_bswap64(x);





  x = ((x >>  4) & 0x0F0F0F0F0F0F0F0F ## UL) | ((x & 0x0F0F0F0F0F0F0F0F ## UL) <<  4);
  x = ((x >>  2) & 0x3333333333333333 ## UL) | ((x & 0x3333333333333333 ## UL) <<  2);
  x = ((x >>  1) & 0x5555555555555555 ## UL) | ((x & 0x5555555555555555 ## UL) <<  1);
  return x;

}

/* Apply a permutation to scramble vertex numbers; a randomly generated
 * permutation is not used because applying it at scale is too expensive. */
static inline int64_t scramble(int64_t v0, int lgN, uint64_t val0, uint64_t val1) {
  uint64_t v = (uint64_t)v0;
  v += val0 + val1;
  v *= (val0 | 0x4519840211493211 ## UL);
  v = (bitreverse(v) >> (64 - lgN));
  (((v >> lgN) == 0)								   ? (void) (0)						   : __assert_fail (#(v >> lgN) == 0, "generator/graph_generator.c.tau.tmp", 100, __PRETTY_FUNCTION__));
  v *= (val1 | 0x3050852102C843A5 ## UL);
  v = (bitreverse(v) >> (64 - lgN));
  (((v >> lgN) == 0)								   ? (void) (0)						   : __assert_fail (#(v >> lgN) == 0, "generator/graph_generator.c.tau.tmp", 103, __PRETTY_FUNCTION__));
  return (int64_t)v;
}

/* Make a single graph edge using a pre-set MRG state. */
static
void make_one_edge(int64_t nverts, int level, int lgN, mrg_state* st, packed_edge* result, uint64_t val0, uint64_t val1) {
  int64_t base_src = 0, base_tgt = 0;
  while (nverts > 1) {
    int square = generate_4way_bernoulli(st, level, lgN);
    int src_offset = square / 2;
    int tgt_offset = square % 2;
    ((base_src <= base_tgt)								   ? (void) (0)						   : __assert_fail (#base_src <= base_tgt, "generator/graph_generator.c.tau.tmp", 115, __PRETTY_FUNCTION__));
    if (base_src == base_tgt) {
      /* Clip-and-flip for undirected graph */
      if (src_offset > tgt_offset) {
        int temp = src_offset;
        src_offset = tgt_offset;
        tgt_offset = temp;
      }
    }
    nverts /= 2;
    ++level;
    base_src += nverts * src_offset;
    base_tgt += nverts * tgt_offset;
  }
  write_edge(result,
             scramble(base_src, lgN, val0, val1),
             scramble(base_tgt, lgN, val0, val1));
}

/* Generate a range of edges (from start_edge to end_edge of the total graph),
 * writing into elements [0, end_edge - start_edge) of the edges array.  This
 * code is parallel on OpenMP and XMT; it must be used with
 * separately-implemented SPMD parallelism for MPI. */
void generate_kronecker_range(
        uint_fast32_t seed[5] /* All values in [0, 2^31 - 1), not all zero */,
       int logN /* In base 2 */,
       int64_t start_edge, int64_t end_edge,
       packed_edge* edges) {
  mrg_state state;
  int64_t nverts = (int64_t)1 << logN;
  int64_t ei;

  mrg_seed(&state, seed);

  uint64_t val0, val1; /* Values for scrambling */
  {
    mrg_state new_state = state;
    mrg_skip(&new_state, 50, 7, 0);
    val0 = mrg_get_uint_orig(&new_state);
    val0 *= 0xFFFFFFFF ## UL;
    val0 += mrg_get_uint_orig(&new_state);
    val1 = mrg_get_uint_orig(&new_state);
    val1 *= 0xFFFFFFFF ## UL;
    val1 += mrg_get_uint_orig(&new_state);
  }

  for (ei = start_edge; ei < end_edge; ++ei) {
    mrg_state new_state = state;
    mrg_skip(&new_state, 0, ei, 0);
    make_one_edge(nverts, 0, logN, &new_state, edges + (ei - start_edge), val0, val1);
  }
}

