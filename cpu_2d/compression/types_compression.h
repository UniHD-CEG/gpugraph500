#ifndef COMPRESSION__TYPES_COMPRESSION_H
#define COMPRESSION__TYPES_COMPRESSION_H

#include "mpi.h"

#ifdef _SIMD
#define MPI_noCompressed MPI_INT64_T;
#define MPI_compressed MPI_UINT32_T;
typedef uint32_t compressionType;

#elif defined(_SIMD_PLUS)
#define MPI_noCompressed MPI_INT64_T;
#define MPI_compressed MPI_INT64_T;
typedef unsigned char compressionType;

#elif defined(_SIMT)
#define MPI_noCompressed MPI_INT64_T;
#define MPI_compressed MPI_INT64_T;
typedef int64_t compressionType;

#else
#define MPI_noCompressed MPI_INT64_T;
#define MPI_compressed MPI_INT64_T;
typedef int64_t compressionType;
#endif


#endif // COMPRESSION__TYPES_COMPRESSION_H
