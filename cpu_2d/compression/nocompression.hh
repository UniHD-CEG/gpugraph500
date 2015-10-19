#ifndef BFS_MULTINODE_NOCOMPRESSION_COMPRESSION_H
#define BFS_MULTINODE_NOCOMPRESSION_COMPRESSION_H

#include "compression.hh"
#include <chrono>

using namespace std::chrono;

template <typename T>
class NoCompression: public Compression<T>
{
public:
    void debugCompression(T *fq, const int size);
    void compress(T *fq_64, const size_t &size, T **compressed_fq_64, size_t &compressedsize);
    void decompress(T *compressed_fq_64, const int size,
                    /*Out*/ T **uncompressed_fq_64, /*In Out*/size_t &uncompressedsize);
    void verifyCompression(const T *fq, const T *uncompressed_fq_64, size_t uncompressedsize) const;
    inline bool isCompressed(const size_t originalsize, const size_t compressedsize) const;
    inline string name() const;
    void reconfigure(int compressionThreshold, string compressionExtraArgument);
};

template <typename T>
void NoCompression<T>::reconfigure(int compressionThreshold, string compressionExtraArgument)
{
    assert(compressionThreshold >= 0);
    assert(compressionExtraArgument.length() >= 0);
}


template <typename T>
void NoCompression<T>::debugCompression(T *fq, const int size)
{

    size_t compressedsize, uncompressedsize = static_cast<size_t>(size);
    T *compressed_fq_64, *uncompressed_fq_64;
    high_resolution_clock::time_point time_0, time_1;
    time_0 = high_resolution_clock::now();
    compress(fq, uncompressedsize, &compressed_fq_64, compressedsize);
    time_1 = high_resolution_clock::now();
    auto encode_time = duration_cast<nanoseconds>(time_1 - time_0).count();
    time_0 = high_resolution_clock::now();
    decompress(compressed_fq_64, compressedsize, &uncompressed_fq_64, uncompressedsize);
    time_1 = high_resolution_clock::now();
    auto decode_time = duration_cast<nanoseconds>(time_1 - time_0).count();

    /**
     * Check validity of results
     */
    verifyCompression(fq, uncompressed_fq_64, uncompressedsize);
    double compressratio = 0.0;
    printf("debug:: no-compression, data: %ldB c/d: %04ld/%04ldus, %02.3f%% gained\n", size * sizeof(int), encode_time,
           decode_time, compressratio);

}

template <typename T>
void NoCompression<T>::compress(T *fq_64, const size_t &size, T **compressed_fq_64,
                                size_t &compressedsize)
{
    compressedsize = size;
    *compressed_fq_64 = fq_64;
}

template <typename T>
void NoCompression<T>::decompress(T *compressed_fq_64, const int size,
                                  /*Out*/ T **uncompressed_fq_64, /*In Out*/size_t &uncompressedsize)
{
    uncompressedsize = size;
    *uncompressed_fq_64 = compressed_fq_64;
}

template <typename T>
void NoCompression<T>::verifyCompression(const T *fq, const T *uncompressed_fq_64,
        const size_t uncompressedsize) const
{
    assert(memcmp(fq, uncompressed_fq_64, uncompressedsize * sizeof(T)) == 0);
}

template <typename T>
inline bool NoCompression<T>::isCompressed(const size_t originalsize, const size_t compressedsize) const
{
    return (false && originalsize != compressedsize);
}

template <typename T>
inline string NoCompression<T>::name() const
{
    return "nocompression";
}


#endif // BFS_MULTINODE_NOCOMPRESSION_COMPRESSION_H
