#ifndef BFS_MULTINODE_NOCOMPRESSION_COMPRESSION_H
#define BFS_MULTINODE_NOCOMPRESSION_COMPRESSION_H



#include "compression.hh"
#include <chrono>

using namespace std::chrono;

using std::string;
using std::vector;
using std::equal;


template <typename T>
class NoCompression: public Compression<T>
{
private:
public:
    void benchmarkCompression(const T *fq, const int size) const;
    void compress(T *fq_64, const size_t &size, T **compressed_fq_64, size_t &compressedsize);
    void decompress(T *compressed_fq_64, const int size,
                    /*Out*/ T **uncompressed_fq_64, /*In Out*/size_t &uncompressedsize);
    void verifyCompression(const T *fq, const T *uncompressed_fq_64, size_t uncompressedsize) const;
    inline bool isCompressed(const size_t originalsize, const size_t compressedsize) const;
    inline string name() const;
};

template <typename T>
void NoCompression<T>::benchmarkCompression(const T *fq, const int size) const
{
    double compressratio = 0.0;
    printf(" c/d ratio: %02.3f%% gained\n", compressratio);

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
}

template <typename T>
inline bool NoCompression<T>::isCompressed(const size_t originalsize, const size_t compressedsize) const
{
    return false;
}

template <typename T>
inline string NoCompression<T>::name() const
{
    return "nocompression";
}


#endif // BFS_MULTINODE_NOCOMPRESSION_COMPRESSION_H
