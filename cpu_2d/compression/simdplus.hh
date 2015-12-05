#ifndef BFS_MULTINODE_SIMDPLUS_COMPRESSION_H
#define BFS_MULTINODE_SIMDPLUS_COMPRESSION_H

#include "compression.hh"
#include "vp4dc.h"
#include "vp4dd.h"

#include <chrono>
using namespace std::chrono;


template <typename T, typename T_C>
class SimdPlus: public Compression<T, T_C>
{
private:
    uint32_t SIMDCOMPRESSION_THRESHOLD;
    inline bool isCompressible(int size) const { return (size > SIMDCOMPRESSION_THRESHOLD); };
public:
    SimdPlus();
    void debugCompression(T *fq, const int size) const;
    void compress(T *fq_64, const size_t &size, T_C **compressed_fq_64, size_t &compressedsize) const;
    void decompress(T_C *compressed_fq_64, const int size,
                    /*Out*/ T **uncompressed_fq_64, /*In Out*/size_t &uncompressedsize) const;
    void verifyCompression(const T *fq, const T *uncompressed_fq_64, size_t uncompressedsize) const;
    inline bool isCompressed(const size_t originalsize, const size_t compressedsize) const;
    inline string name() const;
    void reconfigure(int compressionThreshold, string compressionExtraArgument);
};

template <typename T, typename T_C>
SimdPlus<T, T_C>::SimdPlus()
{
    SIMDCOMPRESSION_THRESHOLD = 64; // use 0xffffff (2^32) to transparently disable
}

template<typename T, typename T_C>
void SimdPlus<T, T_C>::reconfigure(int compressionThreshold, string compressionExtraArgument)
{
    SIMDCOMPRESSION_THRESHOLD = compressionThreshold;
    assert(compressionThreshold >= 0);
    assert(compressionExtraArgument.length() >= 0);
}


template<typename T, typename T_C>
void SimdPlus<T, T_C>::debugCompression(T *fq, const int size) const
{
    /**
     * disabled:
     */
    size_t compressedsize, uncompressedsize = static_cast<size_t>(size);
    T_C *compressed_fq_64;
    T *uncompressed_fq_64;
    high_resolution_clock::time_point time_0, time_1;
    time_0 = high_resolution_clock::now();
    compress(reinterpret_cast<uint64_t *>(fq), uncompressedsize, &compressed_fq_64, compressedsize);
    time_1 = high_resolution_clock::now();
    long encode_time = duration_cast<nanoseconds>(time_1 - time_0).count();
    time_0 = high_resolution_clock::now();
    decompress(compressed_fq_64, compressedsize, &uncompressed_fq_64, uncompressedsize);
    time_1 = high_resolution_clock::now();
    long decode_time = duration_cast<nanoseconds>(time_1 - time_0).count();

    /**
     * Check validity of results
     */
    verifyCompression(reinterpret_cast<uint64_t *>(fq), uncompressed_fq_64, uncompressedsize);
    double compressratio = 0.0;
    printf("debug:: no-compression, data: %ldB c/d: %04ld/%04ldus, %02.3f%% gained\n", size * sizeof(int), encode_time,
           decode_time, compressratio);

}

template<typename T, typename T_C>
void SimdPlus<T, T_C>::compress(T *fq_64, const size_t &size, T_C **compressed_fq_64,
                                     size_t &compressedsize) const
{
    if (isCompressible(size))
    {
        // L948
        const unsigned char *ptr_to_endaddress = p4denc64(reinterpret_cast<uint64_t *>(fq_64), size, &compressed_fq_64);
        compressedsize = static_cast<std::size_t>(ptr_to_endaddress - *compressed_fq_64);
        //
        // unsigned pa[BLOCK_SIZE + 2048];
        // bitdelta32(fq_64 + 1, --n, pa, fq_64[0], mode);
        // vbput32(compressed_fq_64, fq_64[0]);
        // return n == 128 ? p4dencv32(pa, n, compressed_fq_64) : p4denc32(pa, n, compressed_fq_64);

    }
    else
    {
        /**
         * Buffer will not be compressed (Small size. Not worthed)
         */
        compressedsize = size;
        *compressed_fq_64 = reinterpret_cast<uint64_t *>(fq_64);
    }
}

template<typename T, typename T_C>
void SimdPlus<T, T_C>::decompress(T_C *compressed_fq_64, const int size,
                                       /*Out*/ T **uncompressed_fq_64, /*In Out*/size_t &uncompressedsize) const
{
    if (isCompressed(uncompressedsize, size))
    {
        //
	uncompressed_fq_64 = reinterpret_cast<uint64_t **>(uncompressed_fq_64);
        p4ddec64(compressed_fq_64, uncompressedsize, *uncompressed_fq_64);
	uncompressed_fq_64 = reinterpret_cast<T **>(uncompressed_fq_64);
    }
    else
    {
        /**
         * Buffer was not compressed (Small size. Not worthed)
         */
        uncompressedsize = size;
        *uncompressed_fq_64 = reinterpret_cast<T *>(compressed_fq_64);
    }
}

template<typename T, typename T_C>
void SimdPlus<T, T_C>::verifyCompression(const T *fq, const T *uncompressed_fq_64,
        const size_t uncompressedsize) const
{
    if (isCompressible(uncompressedsize))
    {
        assert(memcmp(fq, uncompressed_fq_64, uncompressedsize * sizeof(T)) == 0);
    }
}

template<typename T, typename T_C>
inline bool SimdPlus<T, T_C>::isCompressed(const size_t originalsize, const size_t compressedsize) const
{
    return (isCompressible(originalsize) && originalsize != compressedsize);
}

template<typename T, typename T_C>
inline string SimdPlus<T, T_C>::name() const
{
    return "simdplus";
}


#endif // BFS_MULTINODE_SIMDPLUS_COMPRESSION_H
