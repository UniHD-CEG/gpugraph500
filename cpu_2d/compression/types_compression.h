#ifndef COMPRESSION__TYPES_COMPRESSION_H
#define COMPRESSION__TYPES_COMPRESSION_H


#ifdef _SIMD
typedef long long compressionType;
typedef MPI_INT MPI_compressed;
#elif defined(_SIMD_PLUS)
typedef unsigned char compressionType;
typedef MPI_LONG MPI_compressed;
#elif defined(_SIMT)
typedef int64_t compressionType;
typedef MPI_LONG MPI_compressed;
#else
typedef int64_t compressionType;
typedef MPI_LONG MPI_compressed;
#endif


#endif // COMPRESSION__TYPES_COMPRESSION_H
