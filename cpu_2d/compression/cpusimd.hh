#ifndef BFS_MULTINODE_CPUSIMD_COMPRESSION_H
#define BFS_MULTINODE_CPUSIMD_COMPRESSION_H

#ifdef _SIMD

#include "compression.hh"
#include "codecfactory.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

using namespace std::chrono;
using namespace SIMDCompressionLib;
using std::string;
using std::vector;
using std::equal;

template <typename T, typename T_C>
class CpuSimd: public Compression<T, T_C>
{
private:
    uint32_t SIMDCOMPRESSION_THRESHOLD;
    string codecName;
    IntegerCODEC &codec = *CODECFactory::getFromName("s4-bp128-dm");
    inline bool isCompressible(int size) const { return (size > SIMDCOMPRESSION_THRESHOLD); };
public:
    CpuSimd();
    void debugCompression(T *fq, const size_t size) const;
    inline void compress(T *fq_64, const size_t &size, T_C **compressed_fq_64, size_t &compressedsize) const ;
    inline void decompress(T_C *compressed_fq_64, const int size,
                    /*Out*/ T **uncompressed_fq_64, /*In Out*/size_t &uncompressedsize) const;
    void verifyCompression(const T *fq, const T *uncompressed_fq_64, size_t uncompressedsize) const;
    inline bool isCompressed(const size_t originalsize, const size_t compressedsize) const;
    inline string name() const;
    void reconfigure(int compressionThreshold, string compressionCodec);
};

template <typename T, typename T_C>
CpuSimd<T, T_C>::CpuSimd()
{
    SIMDCOMPRESSION_THRESHOLD = 64;
    codecName = "s4-bp128-dm";
    codec = *CODECFactory::getFromName(codecName);
}

template <typename T, typename T_C>
void CpuSimd<T, T_C>::reconfigure(int compressionThreshold, string compressionCodec)
{
    assert(compressionThreshold > 0);
    assert(compressionCodec.length() > 0);
    codecName = compressionCodec;
    codec = *CODECFactory::getFromName(codecName);
    SIMDCOMPRESSION_THRESHOLD = static_cast<uint32_t>(compressionThreshold);
}

template <typename T, typename T_C>
void CpuSimd<T, T_C>::debugCompression(T *fq, const size_t size) const
{
        assert(fq != NULL);
        assert(size >= 0);
}

template <typename T, typename T_C>
inline void CpuSimd<T, T_C>::compress(T *fq_64, const size_t &size, T_C **compressed_fq_32,
                               size_t &compressedsize) const
{
    if (isCompressible(size))
    {
        compressedsize = size;
        int err1, err2;
        T_C *fq_32;

        err1 = posix_memalign((void **)&fq_32, 16, size * sizeof(T_C));
        err2 = posix_memalign((void **)compressed_fq_32, 16, size * sizeof(T_C));
        if (err1 || err2) {
          printf("Memory error.\n");
          throw "Memory error.";
      exit(1);
        }

        for (int i = 0; i < size; ++i)
        {
            fq_32[i] = static_cast<T_C>(fq_64[i]);
        }
        codec.encodeArray(fq_32, size, *compressed_fq_32, compressedsize);

        free(fq_32);
    }
    else
    {
        /**
         * Buffer will not be compressed (Small size. Not worthed)
         */

    int err;
    err = posix_memalign((void **)compressed_fq_32, 16, size * sizeof(T_C));
    if (err) {
        printf("Memory error.\n");
        throw "Memory error.";
        exit(1);
    }
        for (int i = 0; i < size; ++i)
        {
            (*compressed_fq_32)[i] = static_cast<T_C>(fq_64[i]);
        }
        compressedsize = size;
    }
}

template <typename T, typename T_C>
inline void CpuSimd<T, T_C>::decompress(T_C *compressed_fq_32, const int size,
                                 /*Out*/ T **uncompressed_fq_64, /*In Out*/size_t &uncompressedsize) const
{
    int err;
   if (isCompressed(uncompressedsize, size))
    {
        T_C *compressed_fq_32b = (T_C *) malloc(size * sizeof(T_C));
        memcpy(compressed_fq_32b, compressed_fq_32, size * sizeof(T_C));
        T_C *uncompressed_fq_32 = (T_C *) malloc(uncompressedsize * sizeof(T_C));

        codec.decodeArray(compressed_fq_32b, size, uncompressed_fq_32, uncompressedsize);

        err = posix_memalign((void **)uncompressed_fq_64, 16, uncompressedsize * sizeof(T));
        // *uncompressed_fq_64 = (T *)malloc(uncompressedsize * sizeof(T));
        if (err)
        {
            printf("\nERROR: Memory allocation error!");
            throw "Memory error.";
        }

        for (int i = 0; i < uncompressedsize; ++i)
        {
            (*uncompressed_fq_64)[i] = static_cast<T>(uncompressed_fq_32[i]);
        }

        free(uncompressed_fq_32);
    }
    else
    {
        uncompressedsize = size;
        err = posix_memalign((void **)uncompressed_fq_64, 16, uncompressedsize * sizeof(T));
        // *uncompressed_fq_64 = (T *)malloc(uncompressedsize * sizeof(T));
        if (err)
        {
            printf("\nERROR: Memory allocation error!");
            throw "Memory error.";
        }


        for (int i = 0; i < uncompressedsize; ++i)
        {
           (*uncompressed_fq_64)[i] = static_cast<T>(compressed_fq_32[i]);
        }
    }
}

template <typename T, typename T_C>
void CpuSimd<T, T_C>::verifyCompression(const T *fq, const T *uncompressed_fq_64,
                                        const size_t uncompressedsize) const
{
    if (isCompressible(uncompressedsize))
    {
        assert(memcmp(fq, uncompressed_fq_64, uncompressedsize * sizeof(T)) == 0);
    }
}

template <typename T, typename T_C>
inline bool CpuSimd<T, T_C>::isCompressed(const size_t originalsize, const size_t compressedsize) const
{
    return (isCompressible(originalsize) && originalsize != compressedsize);
}

template <typename T, typename T_C>
inline string CpuSimd<T, T_C>::name() const
{
    return "cpusimd";
}


#endif // _SIMD
#endif // BFS_MULTINODE_CPUSIMD_COMPRESSION_H
