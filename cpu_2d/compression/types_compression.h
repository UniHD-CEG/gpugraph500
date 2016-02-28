#ifndef COMPRESSION__TYPES_COMPRESSION_H
#define COMPRESSION__TYPES_COMPRESSION_H

#include "mpi.h"

#ifdef _SIMD
#define MPInoCompressed MPI_INT64_T;
#define MPIcompressed MPI_UINT32_T;
typedef uint32_t compressionType;

#elif defined(_SIMD_PLUS)
#define MPInoCompressed MPI_INT64_T;
#define MPIcompressed MPI_INT64_T;
typedef unsigned char compressionType;

#elif defined(_SIMT)
#define MPInoCompressed MPI_INT64_T;
#define MPIcompressed MPI_INT64_T;
typedef int64_t compressionType;

#else
#define MPInoCompressed MPI_INT64_T;
#define MPIcompressed MPI_INT64_T;
typedef int64_t compressionType;
#endif


#endif // COMPRESSION__TYPES_COMPRESSION_H
