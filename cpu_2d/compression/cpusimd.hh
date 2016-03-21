#ifndef BFS_MULTINODE_CPUSIMD_COMPRESSION_H
#define BFS_MULTINODE_CPUSIMD_COMPRESSION_H

#ifdef _SIMD

#include "compression.hh"
#include "codecfactory.h"
#include "../config.h"
#include <chrono>
#include "../constants.hh"


using namespace std::chrono;
using namespace SIMDCompressionLib;
using std::string;
using std::vector;
using std::equal;

template <typename T, typename T_C>
class CpuSimd: public Compression<T, T_C>
{
private:
    size_t SIMDCOMPRESSION_THRESHOLD;
    string codecName;
    IntegerCODEC &codec = *CODECFactory::getFromName("s4-bp128-dm");
    inline bool isCompressible(size_t size) const { return (size > SIMDCOMPRESSION_THRESHOLD); };
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
    void init() const;
};

template <typename T, typename T_C>
CpuSimd<T, T_C>::CpuSimd()
{
    SIMDCOMPRESSION_THRESHOLD = 64UL;
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
    SIMDCOMPRESSION_THRESHOLD = static_cast<size_t>(compressionThreshold);
}

template <typename T, typename T_C>
void CpuSimd<T, T_C>::debugCompression(T *fq, const size_t size) const
{
    /**
     *
     * statistics, checks,etc here
     */
    assert(fq != NULL);
    assert(size >= 0UL);
}

template <typename T, typename T_C>
inline void CpuSimd<T, T_C>::compress(T * restrict fq_64, const size_t &size, T_C ** restrict compressed_fq_32,
                               size_t &compressedsize) const
{
    if (isCompressible(size))
    {
        compressedsize = size;
        T_C * restrict fq_32 = NULL;

        const int err1 = posix_memalign((void **)&fq_32, ALIGNMENT, size * sizeof(T_C));
        const int err2 = posix_memalign((void **)compressed_fq_32, ALIGNMENT, size * sizeof(T_C));
        if (err1 || err2) {
            throw "Memory error.";
        }

#ifndef _COMPRESSIONDEBUG
        // test overflow
        const uint32_t LIMIT_UINT32 = (1L << 32) - 1;
        for (size_t i = 0U; i < size; ++i)
        {
            const T tested = fq_64[i];
            assert(tested <= LIMIT_UINT32);
            // assert(tested >= 0);
        }
#endif

        for (size_t i = 0U; i < size; ++i)
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

        const int err = posix_memalign((void **)compressed_fq_32, ALIGNMENT, size * sizeof(T_C));
        if (err) {
            throw "Memory error.";
        }

        for (size_t i = 0U; i < size; ++i)
        {
            (*compressed_fq_32)[i] = static_cast<T_C>(fq_64[i]);
        }

        compressedsize = size;
    }
}

template <typename T, typename T_C>
inline void CpuSimd<T, T_C>::decompress(T_C * restrict compressed_fq_32, const int size,
                                 /*Out*/ T ** restrict uncompressed_fq_64, /*In Out*/size_t &uncompressedsize) const
{
   if (isCompressed(uncompressedsize, size))
    {
        T_C * restrict compressed_fq_32_tmp = NULL;
        T_C * restrict uncompressed_fq_32 = NULL;

        const int err1 = posix_memalign((void **)&compressed_fq_32_tmp, ALIGNMENT, size * sizeof(T_C));
        const int err2 = posix_memalign((void **)&uncompressed_fq_32, ALIGNMENT, uncompressedsize * sizeof(T_C));
        if (err1 || err2)
        {
            throw "Memory error.";
        }
        memcpy(compressed_fq_32_tmp, compressed_fq_32, size * sizeof(T_C));

        codec.decodeArray(compressed_fq_32_tmp, size, uncompressed_fq_32, uncompressedsize);

        const int err3 = posix_memalign((void **)uncompressed_fq_64, ALIGNMENT, uncompressedsize * sizeof(T));
        if (err3)
        {
            throw "Memory error.";
        }

        for (size_t i = 0UL; i < uncompressedsize; ++i)
        {
            (*uncompressed_fq_64)[i] = static_cast<T>(uncompressed_fq_32[i]);
        }

        free(compressed_fq_32_tmp);
        free(uncompressed_fq_32);
    }
    else
    {
        uncompressedsize = size;
        const int err = posix_memalign((void **)uncompressed_fq_64, ALIGNMENT, uncompressedsize * sizeof(T));
        if (err)
        {
            throw "Memory error.";
        }

        for (size_t i = 0U; i < uncompressedsize; ++i)
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

template <typename T, typename T_C>
void CpuSimd<T, T_C>::init() const
{
}

#endif // _SIMD
#endif // BFS_MULTINODE_CPUSIMD_COMPRESSION_H
