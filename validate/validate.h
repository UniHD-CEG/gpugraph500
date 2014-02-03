/*
 * Everything I need for validation.
 *
 */
#ifndef VALIDATE_H
#define VALIDATE_H

#include "../distmatrix2d.h"
#include "mpi_workarounds.h"


int validate_bfs_result(const DistMatrix2d& store, packed_edge *edgelist, int64_t number_of_edges,
                        const int64_t nglobalverts, const int64_t root, int64_t* const pred, int64_t* const edge_visit_count_ptr, int *level);

static inline size_t size_min(size_t a, size_t b) {
  return a < b ? a : b;
}

static inline ptrdiff_t ptrdiff_min(ptrdiff_t a, ptrdiff_t b) {
  return a < b ? a : b;
}

/* Chunk size for blocks of one-sided operations; a fence is inserted after (at
 * most) each CHUNKSIZE one-sided operations. */
/* It seams as there is a limit of the number of MPI_Get in an epoche in OpenMPI.
 * An incresed Chunksize can in this way problematic. */
#define CHUNKSIZE (1 << 13)
#define HALF_CHUNKSIZE ((CHUNKSIZE) / 2)

#endif // VALIDATE_H
