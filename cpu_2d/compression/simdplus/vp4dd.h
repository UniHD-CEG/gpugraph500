/**
    Copyright (C) powturbo 2013-2015
    GPL v2 License
  
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    - homepage : https://sites.google.com/site/powturbo/
    - github   : https://github.com/powturbo
    - twitter  : https://twitter.com/powturbo
    - email    : powturbo [_AT_] gmail [_DOT_] com
**/
//     vp4dd.h - "Integer Compression" Turbo PforDelta 
#pragma once

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>

#if defined(__GNUC__)
#define ALIGNED(t,v,n)  t v __attribute__ ((aligned (n)))
#define ALWAYS_INLINE   inline __attribute__((always_inline))
#define _PACKED                 __attribute__ ((packed))
#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)

#define popcnt32(__x)   __builtin_popcount(__x)
#define popcnt64(__x)   __builtin_popcountll(__x)
#endif


#define P4DSIZE 128 //64 //
#define P4DN 	(P4DSIZE/64)

//---------------- Bulk decompress of TurboPFor compressed integer array -------------------------------------------------------
// decompress a previously (with p4denc32) 32 bits packed array. Return value = end of packed buffer in 
unsigned char *p4ddec32(   unsigned char *__restrict in, unsigned n, unsigned *__restrict out);
unsigned char *p4ddec64(   unsigned char *__restrict in, unsigned n, uint64_t *__restrict out);

unsigned char *p4dd32(     unsigned char *__restrict in, unsigned n, unsigned *__restrict out, unsigned b, unsigned bx);
unsigned char *p4dd64(     unsigned char *__restrict in, unsigned n, uint64_t *__restrict out, unsigned b, unsigned bx);

unsigned char *p4ddecv32(  unsigned char *__restrict in, unsigned n, unsigned *__restrict out);  // SIMD

//-- delta min = 0 
unsigned char *p4dddec32(  unsigned char *__restrict in, unsigned n, unsigned *__restrict out, unsigned start);
unsigned char *p4dddecv32( unsigned char *__restrict in, unsigned n, unsigned *__restrict out, unsigned start);
//-- delta min = 1
unsigned char *p4dd1dec32( unsigned char *__restrict in, unsigned n, unsigned *__restrict out, unsigned start);
unsigned char *p4dd1decv32(unsigned char *__restrict in, unsigned n, unsigned *__restrict out, unsigned start);

// same as abose, b and bx not stored within the compressed stream header (see idxcr.c/idxqry.c for an example)
unsigned char *p4ddv32(    unsigned char *__restrict in, unsigned n, unsigned *__restrict out,                 unsigned b, unsigned bx);

unsigned char *p4ddd32(    unsigned char *__restrict in, unsigned n, unsigned *__restrict out, unsigned start, unsigned b, unsigned bx);
unsigned char *p4dddv32(   unsigned char *__restrict in, unsigned n, unsigned *__restrict out, unsigned start, unsigned b, unsigned bx);

unsigned char *p4dd1d32(   unsigned char *__restrict in, unsigned n, unsigned *__restrict out, unsigned start, unsigned b, unsigned bx);
unsigned char *p4dd1dv32(  unsigned char *__restrict in, unsigned n, unsigned *__restrict out, unsigned start, unsigned b, unsigned bx);

//---------------- Direct Access functions to compressed TurboPFor array -------------------------------------------------------
#define P4D_PAD8(__x) 		( (((__x)+8-1)/8) )
#define P4D_B(__x)  		(((__x) >> 1) & 0x3f)
#define P4D_XB(__x) 		(((__x) & 1)?((__x) >> 8):0)
#define P4D_ININC(__in, __x) __in += 1+(__x & 1)

static inline unsigned vp4dbits(unsigned char *__restrict in, int *bx); 

struct p4d {
  unsigned long long *xmap;
  unsigned char *ex;
  unsigned i,bx,cum[P4DN+1];
  int oval,idx;
};

// prepare direct access usage
static inline void p4dini(struct p4d *p4d, unsigned char *__restrict *pin, unsigned n, unsigned *b);

// Get a single value with index "idx" from a p4denc32 packed array
static ALWAYS_INLINE unsigned vp4dget32(struct p4d *p4d, unsigned char *__restrict in, unsigned b, unsigned idx);

// like vp4dget32 but for 16 bits packed array (with p4denc16)	
static ALWAYS_INLINE unsigned vp4dget16(struct p4d *p4d, unsigned char *__restrict in, unsigned b, unsigned idx); 

// Get the next single value greater of equal to val
static ALWAYS_INLINE int vp4dgeq(struct p4d *p4d, unsigned char *__restrict in, unsigned b, int val);

/* like p4ddec32 but using direct access. This is only a demo showing direct access usage. Use p4ddec32 for instead for decompressing entire blocks */
unsigned char *p4ddecx32(  unsigned char *__restrict in, unsigned n, unsigned *__restrict out);
unsigned char *p4dfdecx32( unsigned char *__restrict in, unsigned n, unsigned *__restrict out, unsigned start);
unsigned char *p4df0decx32(unsigned char *__restrict in, unsigned n, unsigned *__restrict out, unsigned start);

#define P4DSIZE 128 //64 //
#define P4DN 	(P4DSIZE/64)
#ifdef __cplusplus
}
#endif
