#ifndef BFS_MULTINODE_COMPRESSION_H
#define BFS_MULTINODE_COMPRESSION_H


#include <string>
#include <cstring>


using std::string;

template <typename T>
class Compression
{
public:
    virtual void benchmarkCompression(T *fq, const int size) = 0;
    virtual void compress(T *fq_64, const size_t &size, T **compressed_fq_64, size_t &compressedsize) = 0;
    virtual void decompress(T *compressed_fq_64, const int size,
                            /*Out*/ T **uncompressed_fq_64, /*In Out*/size_t &uncompressedsize) = 0;
    virtual void verifyCompression(const T *fq, const T *uncompressed_fq_64,
                                   const size_t uncompressedsize) const = 0;
    virtual bool isCompressed(const size_t originalsize, const size_t compressedsize) const = 0;
    virtual void reconfigure(int compressionThreshold, string compressionExtraArgument) = 0;
    virtual ~Compression()
    {
    };
    virtual string name() const = 0;
};

#endif // BFS_MULTINODE_COMPRESSION_H
