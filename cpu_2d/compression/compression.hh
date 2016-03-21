#ifndef BFS_MULTINODE_COMPRESSION_H
#define BFS_MULTINODE_COMPRESSION_H


#include <string>
#include <cstring>


using std::string;

template <typename T, typename T_C>
class Compression
{
public:
    virtual void debugCompression(T *fq, const size_t size) const = 0;
    virtual void compress(T * restrict fq_64, const size_t &size, T_C ** restrict compressed_fq_64, size_t &compressedsize) const = 0;
    virtual void decompress(T_C * restrict compressed_fq_64, const int size,
                            /*Out*/ T ** restrict uncompressed_fq_64, /*In Out*/size_t &uncompressedsize) const = 0;
    virtual void verifyCompression(const T *fq, const T *uncompressed_fq_64,
                                   const size_t uncompressedsize) const = 0;
    virtual bool isCompressed(const size_t originalsize, const size_t compressedsize) const = 0;
    virtual void reconfigure(int compressionThreshold, string compressionExtraArgument) = 0;
    virtual ~Compression()
    {
    };
    virtual string name() const = 0;
    virtual void init() const = 0;
};

#endif // BFS_MULTINODE_COMPRESSION_H
