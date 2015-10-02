#ifndef BFS_MULTINODE_COMPRESSION_H
#define BFS_MULTINODE_COMPRESSION_H



template <class T>
class Compression
{
protected:
public:
    virtual void benchmarkCompression(const T *fq, const int size) =0;
    virtual void compress(T *fq_64, const size_t &size, T **compressed_fq_64, size_t &compressedsize) =0;
    virtual void decompress(T *compressed_fq_64, const int size,
                               /*Out*/ T **uncompressed_fq_64, /*In Out*/size_t &uncompressedsize) =0;
    virtual void verifyCompression(const T *fq, const T *uncompressed_fq_64,
                                   compressedsizeonst size_t uncompressedsize) =0;
    virtual inline bool isCompressed(const size_t originalsize, const size_t compressedsize) =0;
};

#endif // BFS_MULTINODE_COMPRESSION_H
