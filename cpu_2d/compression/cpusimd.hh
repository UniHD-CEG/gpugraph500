#ifndef BFS_MULTINODE_CPUSIMD_COMPRESSION_H
#define BFS_MULTINODE_CPUSIMD_COMPRESSION_H



template <class T>
class CpuSimd
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


template <class T>
virtual void CpuSimd<class T>::benchmarkCompression(const T *fq, const int size) const
{
}

template <class T>
virtual void CpuSimd<class T>::Compression(T *fq_64, const size_t &size, T **compressed_fq_64,
            size_t &compressedsize) const
{
}

template <class T>
virtual void CpuSimd<class T>::Decompression(T *compressed_fq_64, const int size,
        /*Out*/ T **uncompressed_fq_64, /*In Out*/size_t &uncompressedsize) const
{
}

template <class T>
virtual void CpuSimd<class T>::verifyCompression(const T *fq, const T *uncompressed_fq_64,
            const size_t uncompressedsize) const
{
}

template <class T>
virtual inline bool CpuSimd<class T>::isCompressed(const size_t originalsize, const size_t compressedsize) const
{
}

#endif // BFS_MULTINODE_CPUSIMD_COMPRESSION_H
