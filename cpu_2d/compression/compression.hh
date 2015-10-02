#ifndef BFS_MULTINODE_COMPRESSION_H
#define BFS_MULTINODE_COMPRESSION_H



template <class T>
class Compression
{
protected:
public:
    virtual void benchmarkCompression(const T *fq, const int size) const;
    virtual void Compression(T *fq_64, const size_t &size, T **compressed_fq_64, size_t &compressedsize) const;
    virtual void Decompression(T *compressed_fq_64, const int size,
                               /*Out*/ T **uncompressed_fq_64, /*In Out*/size_t &uncompressedsize) const;
    virtual void verifyCompression(const T *fq, const T *uncompressed_fq_64,
                                   compressedsizeonst size_t uncompressedsize) const;
    virtual inline bool isCompressed(const size_t originalsize, const size_t compressedsize) const;
};

/**
 * Benchmark compression/decompression.
 *
 */
virtual void benchmarkCompression(const T *fq, const int size) const
{
}

/**
 * Compression.
 *
 */
virtual void Compression(T *fq_64, const size_t &size, T **compressed_fq_64,
                         size_t &compressedsize) const
{
}

/**
 * Decompression.
 *
 */
virtual void Decompression(T *compressed_fq_64, const int size,
                           /*Out*/ T **uncompressed_fq_64, /*In Out*/size_t &uncompressedsize) const
{
}

/**
 * Compression/decompression verification.
 *
 */
virtual void verifyCompression(const T *fq, const T *uncompressed_fq_64,
                               const size_t uncompressedsize) const
{
}

/**
 * isCompressed()
 *
 */
virtual inline bool isCompressed(const size_t originalsize, const size_t compressedsize) const
{
}

#endif // BFS_MULTINODE_COMPRESSION_H
