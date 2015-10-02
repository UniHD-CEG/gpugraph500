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

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif

#include <inttypes.h>

#include "user_settings.h"
#include "splittable_mrg.h"
#include "graph_generator.h"

/* Initiator settings: for faster random number generation, the initiator
 * probabilities are defined as fractions (a = INITIATOR_A_NUMERATOR /
 * INITIATOR_DENOMINATOR, b = c = INITIATOR_BC_NUMERATOR /
 * INITIATOR_DENOMINATOR, d = 1 - a - b - c. */

#define INITIATOR_A_NUMERATOR 5700
#define INITIATOR_BC_NUMERATOR 1900
#define INITIATOR_DENOMINATOR 10000

/* If this macro is defined to a non-zero value, use SPK_NOISE_LEVEL /
 * INITIATOR_DENOMINATOR as the noise parameter to use in introducing noise
 * into the graph parameters.  The approach used is from "A Hitchhiker's Guide
 * to Choosing Parameters of Stochastic Kronecker Graphs" by C. Seshadhri, Ali
 * Pinar, and Tamara G. Kolda (http://arxiv.org/abs/1102.5046v1), except that
 * the adjustment here is chosen based on the current level being processed
 * rather than being chosen randomly. */

#define SPK_NOISE_LEVEL 0

/* #define SPK_NOISE_LEVEL 1000 -- in INITIATOR_DENOMINATOR units */

static int generate_4way_bernoulli(mrg_state *st, int level, int nlevels)
{
    /* Generator a pseudorandom number in the range [0, INITIATOR_DENOMINATOR)
     * without modulo bias. */
    static const uint32_t limit = (UINT32_C(0xFFFFFFFF) % INITIATOR_DENOMINATOR);
    uint32_t val = mrg_get_uint_orig(st);
    uint32_t adjusted_bc_numerator;
    int spk_noise_factor;

    if (/* Unlikely */ val < limit)
    {
        do
        {
            val = mrg_get_uint_orig(st);
        }
        while (val < limit);
    }

    spk_noise_factor = 2 * SPK_NOISE_LEVEL * level / nlevels - SPK_NOISE_LEVEL;
#if SPK_NOISE_LEVEL == 0
    spk_noise_factor = 0;
#endif
    adjusted_bc_numerator = INITIATOR_BC_NUMERATOR + spk_noise_factor;
    val %= INITIATOR_DENOMINATOR;
    if (val < adjusted_bc_numerator) return 1;
    val -= adjusted_bc_numerator;
    if (val < adjusted_bc_numerator) return 2;
    val -= adjusted_bc_numerator;
#if SPK_NOISE_LEVEL == 0
    if (val < INITIATOR_A_NUMERATOR) return 0;
#else
    if (val < INITIATOR_A_NUMERATOR * (INITIATOR_DENOMINATOR - 2 * INITIATOR_BC_NUMERATOR) /
        (INITIATOR_DENOMINATOR - 2 * adjusted_bc_numerator)) return 0;
#endif
    return 3;
}

/* Reverse bits in a number; this should be optimized for performance
 * (including using bit- or byte-reverse intrinsic if your platform has them).
 * */
static inline uint64_t bitreverse(uint64_t x)
{
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3)
#define USE_GCC_BYTESWAP /* __builtin_bswap* are in 4.3 but not 4.2 */
#endif

    /* 64-bit code */
#ifdef USE_GCC_BYTESWAP
    x = __builtin_bswap64(x);
#else
    x = (x >> 32) | (x << 32);
    x = ((x >> 16) & UINT64_C(0x0000FFFF0000FFFF)) | ((x & UINT64_C(0x0000FFFF0000FFFF)) << 16);
    x = ((x >> 8) & UINT64_C(0x00FF00FF00FF00FF)) | ((x & UINT64_C(0x00FF00FF00FF00FF)) << 8);
#endif
    x = ((x >> 4) & UINT64_C(0x0F0F0F0F0F0F0F0F)) | ((x & UINT64_C(0x0F0F0F0F0F0F0F0F)) << 4);
    x = ((x >> 2) & UINT64_C(0x3333333333333333)) | ((x & UINT64_C(0x3333333333333333)) << 2);
    x = ((x >> 1) & UINT64_C(0x5555555555555555)) | ((x & UINT64_C(0x5555555555555555)) << 1);
    return x;

}

/* Apply a permutation to scramble vertex numbers; a randomly generated
 * permutation is not used because applying it at scale is too expensive. */
static inline int64_t scramble(int64_t v0, int lgN, uint64_t val0, uint64_t val1)
{
    uint64_t v = (uint64_t) v0;
    v += val0 + val1;
    v *= (val0 | UINT64_C(0x4519840211493211));
    v = (bitreverse(v) >> (64 - lgN));
    assert((v >> lgN) == 0);
    v *= (val1 | UINT64_C(0x3050852102C843A5));
    v = (bitreverse(v) >> (64 - lgN));
    assert((v >> lgN) == 0);
    return (int64_t) v;
}

/* Make a single graph edge using a pre-set MRG state. */
static
void make_one_edge(int64_t nverts, int level, int lgN, mrg_state *st, packed_edge *result, uint64_t val0,
                   uint64_t val1)
{
    int64_t base_src = 0, base_tgt = 0;
    while (nverts > 1)
    {
        int square = generate_4way_bernoulli(st, level, lgN);
        int src_offset = square / 2;
        int tgt_offset = square % 2;
        assert(base_src <= base_tgt);
        if (base_src == base_tgt)
        {
            /* Clip-and-flip for undirected graph */
            if (src_offset > tgt_offset)
            {
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
    const uint_fast32_t seed[5] /* All values in [0, 2^31 - 1), not all zero */,
    int logN /* In base 2 */,
    int64_t start_edge, int64_t end_edge,
    packed_edge *edges)
{
    mrg_state state;
    int64_t nverts = (int64_t) 1 << logN;
    int64_t ei;

    mrg_seed(&state, seed);

    uint64_t val0, val1; /* Values for scrambling */
    {
        mrg_state new_state = state;
        mrg_skip(&new_state, 50, 7, 0);
        val0 = mrg_get_uint_orig(&new_state);
        val0 *= UINT64_C(0xFFFFFFFF);
        val0 += mrg_get_uint_orig(&new_state);
        val1 = mrg_get_uint_orig(&new_state);
        val1 *= UINT64_C(0xFFFFFFFF);
        val1 += mrg_get_uint_orig(&new_state);
    }

    for (ei = start_edge; ei < end_edge; ++ei)
    {
        mrg_state new_state = state;
        mrg_skip(&new_state, 0, ei, 0);
        make_one_edge(nverts, 0, logN, &new_state, edges + (ei - start_edge), val0, val1);
    }
}

