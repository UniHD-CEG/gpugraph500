#ifndef COMPRESSION__TYPES_COMPRESSION_H
#define COMPRESSION__TYPES_COMPRESSION_H


#ifdef _SIMD
typedef int64_t compressionType;
#elif defined(_SIMD_PLUS)
typedef unsigned char compressionType;
#elif defined(_SIMT)
typedef int64_t compressionType;
#else
typedef int64_t compressionType;
#endif


#endif // COMPRESSION__TYPES_COMPRESSION_H
