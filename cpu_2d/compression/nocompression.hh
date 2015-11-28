#ifndef BFS_MULTINODE_NOCOMPRESSION_COMPRESSION_H
#define BFS_MULTINODE_NOCOMPRESSION_COMPRESSION_H

#include "compression.hh"
#include <chrono>

using namespace std::chrono;

template <typename T, typename T_C>
class NoCompression: public Compression<T, T_C>
{
public:
    void debugCompression(T *fq, const int size) const;
    void compress(T *fq_64, const size_t &size, T_C **compressed_fq_64, size_t &compressedsize) const;
    void decompress(T_C *compressed_fq_64, const int size,
                    /*Out*/ T **uncompressed_fq_64, /*In Out*/size_t &uncompressedsize) const;
    void verifyCompression(const T *fq, const T *uncompressed_fq_64, size_t uncompressedsize) const;
    inline bool isCompressed(const size_t originalsize, const size_t compressedsize) const;
    inline string name() const;
    void reconfigure(int compressionThreshold, string compressionExtraArgument);
};

template<typename T, typename T_C>
void NoCompression<T, T_C>::reconfigure(int compressionThreshold, string compressionExtraArgument)
{
    assert(compressionThreshold >= 0);
    assert(compressionExtraArgument.length() >= 0);
}


template<typename T, typename T_C>
void NoCompression<T, T_C>::debugCompression(T *fq, const int size) const
{

    size_t compressedsize, uncompressedsize = static_cast<size_t>(size);
    T *compressed_fq_64, *uncompressed_fq_64;
    high_resolution_clock::time_point time_0, time_1;
    time_0 = high_resolution_clock::now();
    compress(fq, uncompressedsize, &compressed_fq_64, compressedsize);
    time_1 = high_resolution_clock::now();
    long encode_time = duration_cast<nanoseconds>(time_1 - time_0).count();
    time_0 = high_resolution_clock::now();
    decompress(compressed_fq_64, compressedsize, &uncompressed_fq_64, uncompressedsize);
    time_1 = high_resolution_clock::now();
    long decode_time = duration_cast<nanoseconds>(time_1 - time_0).count();

    /**
     * Check validity of results
     */
    verifyCompression(fq, uncompressed_fq_64, uncompressedsize);
    double compressratio = 0.0;
    printf("debug:: no-compression, data: %ldB c/d: %04ld/%04ldus, %02.3f%% gained\n", size * sizeof(int), encode_time,
           decode_time, compressratio);

}

template<typename T, typename T_C>
void NoCompression<T, T_C>::compress(T *fq_64, const size_t &size, T_C **compressed_fq_64,
                                     size_t &compressedsize) const
{
    compressedsize = size;
    *compressed_fq_64 = fq_64;
}

template<typename T, typename T_C>
void NoCompression<T, T_C>::decompress(T_C *compressed_fq_64, const int size,
                                       /*Out*/ T **uncompressed_fq_64, /*In Out*/size_t &uncompressedsize) const
{
    uncompressedsize = size;
    *uncompressed_fq_64 = compressed_fq_64;
}

template<typename T, typename T_C>
void NoCompression<T, T_C>::verifyCompression(const T *fq, const T *uncompressed_fq_64,
        const size_t uncompressedsize) const
{
    assert(memcmp(fq, uncompressed_fq_64, uncompressedsize * sizeof(T)) == 0);
}

template<typename T, typename T_C>
inline bool NoCompression<T, T_C>::isCompressed(const size_t originalsize, const size_t compressedsize) const
{
    return (false && originalsize != compressedsize);
}

template<typename T, typename T_C>
inline string NoCompression<T, T_C>::name() const
{
    return "nocompression";
}


#endif // BFS_MULTINODE_NOCOMPRESSION_COMPRESSION_H
