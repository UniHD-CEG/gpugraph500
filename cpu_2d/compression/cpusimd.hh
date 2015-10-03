#ifndef BFS_MULTINODE_CPUSIMD_COMPRESSION_H
#define BFS_MULTINODE_CPUSIMD_COMPRESSION_H

#ifdef _SIMDCOMPRESS



#include "compression.hh"
#include "codecfactory.h"

using namespace SIMDCompressionLib;

using std::string;
using std::vector;
using std::equal;


template <typename T>
class CpuSimd: public Compression<T>
{
private:
    uint32_t SIMDCOMPRESSION_THRESHOLD;
    IntegerCODEC codec;
public:
    void benchmarkCompression(const T *fq, const int size) const;
    void compress(T *fq_64, const size_t &size, T **compressed_fq_64, size_t &compressedsize);
    void decompress(T *compressed_fq_64, const int size,
                    /*Out*/ T **uncompressed_fq_64, /*In Out*/size_t &uncompressedsize);
    void verifyCompression(const T *fq, const T *uncompressed_fq_64, size_t uncompressedsize) const;
    inline bool isCompressed(const size_t originalsize, const size_t compressedsize) const;
    inline string name() const;
    /**
     * Constructor
     */
    CpuSimd()
    {
        &codec = *CODECFactory::getFromName("s4-bp128-dm");
        SIMDCOMPRESSION_THRESHOLD = 512;
        // Use 0xffffffff (2^32) to transparently deactivate compression.
        // SIMDCOMPRESSION_THRESHOLD = 0xffffffff;
    }
    /**
     * Destructor
     */
    virtual ~CpuSimd()
    {
    }
};


template <typename T>
void CpuSimd<T>::benchmarkCompression(const T *fq, const int size) const
{
    if (size > SIMDCOMPRESSION_THRESHOLD)
    {
        char const *codec_name = "s4-bp128-dm";
        IntegerCODEC &codec =  *CODECFactory::getFromName(codec_name);
        high_resolution_clock::time_point time_0, time_1;
        vector<uint32_t>  fq_32(fq, fq + size);
        vector<uint32_t>  compressed_fq_32(size + 1024);
        vector<uint32_t>  uncompressed_fq_32(size);
        size_t compressedsize = compressed_fq_32.size();
        size_t uncompressedsize = uncompressed_fq_32.size();
        time_0 = high_resolution_clock::now();
        codec.encodeArray(fq_32.data(), fq_32.size(), compressed_fq_32.data(), compressedsize);
        time_1 = high_resolution_clock::now();
        auto encode_time = chrono::duration_cast<chrono::nanoseconds>(time_1 - time_0).count();
        compressed_fq_32.resize(compressedsize);
        compressed_fq_32.shrink_to_fit();
        vector<T> compressed_fq_64(compressed_fq_32.begin(), compressed_fq_32.end());
        time_0 = high_resolution_clock::now();
        codec.decodeArray(compressed_fq_32.data(), compressed_fq_32.size(), uncompressed_fq_32.data(), uncompressedsize);
        time_1 = high_resolution_clock::now();
        auto decode_time = chrono::duration_cast<chrono::nanoseconds>(time_1 - time_0).count();
        uncompressed_fq_32.resize(uncompressedsize);
        vector<T> uncompressed_fq_64(uncompressed_fq_32.begin(), uncompressed_fq_32.end());
        /**
         * Check validity of results
         */
        assert(size == uncompressedsize && equal(uncompressed_fq_64.begin(), uncompressed_fq_64.end(), fq));
        double compressedbits = 32.0 * static_cast<double>(compressed_fq_32.size()) / static_cast<double>(fq_32.size());
        double compressratio = (100.0 - 100.0 * compressedbits / 32.0);
        printf("SIMD.codec: %s, c/d: %04ld/%04ldus, %02.3f%% gained\n", codec_name, encode_time, decode_time,
               compressratio);
    }
}

template <typename T>
void CpuSimd<T>::compress(T *fq_64, const size_t &size, T **compressed_fq_64,
                          size_t &compressedsize)
{
    if (size > SIMDCOMPRESSION_THRESHOLD)
    {
        uint32_t *fq_32 = (uint32_t *)malloc(size * sizeof(uint32_t));
        uint32_t *compressed_fq_32 = (uint32_t *)malloc(size * sizeof(uint32_t));
        if (compressed_fq_32 == NULL || fq_32 == NULL)
        {
            printf("\nERROR: Memory allocation error!");
            abort();
        }

        compressedsize = size;
        for (int i = 0; i < size; ++i)
        {
            fq_32[i] = static_cast<uint32_t>(fq_64[i]);
        }
        codec.encodeArray(fq_32, size, compressed_fq_32, compressedsize);
        // if this condition is met it can not be known whether or not there has been a compression.
        // Todo: find solution
        assert(compressedsize < size);
        *compressed_fq_64 = NULL;
        *compressed_fq_64 = (T *)malloc(compressedsize * sizeof(T));
        if (*compressed_fq_64 == NULL)
        {
            printf("\nERROR: Memory allocation error!");
            abort();
        }
        for (auto i = 0; i < compressedsize; ++i)
        {
            (*compressed_fq_64)[i] = static_cast<T>(compressed_fq_32[i]);
        }
        free(fq_32);
        free(compressed_fq_32);
    }
    else
    {
        /**
         * Buffer will not be compressed (Small size. Not worthed)
         */
        compressedsize = size;
        *compressed_fq_64 = fq_64;
    }
}

template <typename T>
void CpuSimd<T>::decompress(T *compressed_fq_64, const int size,
                            /*Out*/ T **uncompressed_fq_64, /*In Out*/size_t &uncompressedsize)
{
    if (isCompressed(uncompressedsize, size))
    {
        uint32_t *uncompressed_fq_32 = (uint32_t *) malloc(uncompressedsize * sizeof(uint32_t));
        uint32_t *compressed_fq_32 = (uint32_t *) malloc(size * sizeof(uint32_t));
        if (compressed_fq_32 == NULL || uncompressed_fq_32 == NULL)
        {
            printf("\nERROR: Memory allocation error!");
            abort();
        }
        // memcpy((uint32_t *)compressed_fq_32, (uint32_t *)compressed_fq_64, size * sizeof(uint32_t));
        for (int i = 0; i < size; ++i)
        {
            compressed_fq_32[i] = static_cast<uint32_t>(compressed_fq_64[i]);
        }
        codec.decodeArray(compressed_fq_32, size, uncompressed_fq_32, uncompressedsize);
        *uncompressed_fq_64 = (T *)malloc(uncompressedsize * sizeof(T));
        if (*uncompressed_fq_64 == NULL)
        {
            printf("\nERROR: Memory allocation error!");
            abort();
        }
        // memcpy((T *)uncompressed_fq_64, (uint32_t *)uncompressed_fq_32, uncompressedsize * sizeof(uint32_t));
        for (auto i = 0; i < uncompressedsize; ++i)
        {
            (*uncompressed_fq_64)[i] = static_cast<T>(uncompressed_fq_32[i]);
        }
        free(compressed_fq_32);
        free(uncompressed_fq_32);
    }
    else
    {
        /**
         * Buffer was not compressed (Small size. Not worthed)
         */
        uncompressedsize = size;
        *uncompressed_fq_64 = compressed_fq_64;
    }
}

template <typename T>
void CpuSimd<T>::verifyCompression(const T *fq, const T *uncompressed_fq_64,
                                   const size_t uncompressedsize) const
{
    if (uncompressedsize > SIMDCOMPRESSION_THRESHOLD)
    {
        assert(memcmp(fq, uncompressed_fq_64, uncompressedsize * sizeof(T)) == 0);
    }
}

template <typename T>
inline bool CpuSimd<T>::isCompressed(const size_t originalsize, const size_t compressedsize) const
{
    return (originalsize > SIMDCOMPRESSION_THRESHOLD && originalsize != compressedsize);
}

template <typename T>
inline string CpuSimd<T>::name() const
{
    return "cpusimd";
}




#endif // _SIMDCOMPRESS
#endif // BFS_MULTINODE_CPUSIMD_COMPRESSION_H
