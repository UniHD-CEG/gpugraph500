#ifndef BFS_MULTINODE_CPUSIMD_COMPRESSION_H
#define BFS_MULTINODE_CPUSIMD_COMPRESSION_H

#ifdef _SIMD

#include "compression.hh"
#include "codecfactory.h"
#include <chrono>

using namespace std::chrono;
using namespace SIMDCompressionLib;

using std::string;
using std::vector;
using std::equal;

template <typename T, typename T_C>
class CpuSimd: public Compression<T, T_C>
{
private:
    uint32_t SIMDCOMPRESSION_THRESHOLD;
    string codecName;
    IntegerCODEC &codec = *CODECFactory::getFromName("s4-bp128-dm");
    inline bool isCompressible(int size) const { return (size > SIMDCOMPRESSION_THRESHOLD); };
public:
    CpuSimd();
    void debugCompression(T *fq, const size_t size) const;
    inline bool compress(T *fq_64, const size_t &size, T_C **compressed_fq_64, size_t &compressedsize) const ;
    inline bool decompress(T_C *compressed_fq_64, const int size,
                    /*Out*/ T **uncompressed_fq_64, /*In Out*/size_t &uncompressedsize) const;
    void verifyCompression(const T *fq, const T *uncompressed_fq_64, size_t uncompressedsize) const;
    inline bool isCompressed(const size_t originalsize, const size_t compressedsize) const;
    inline string name() const;
    void reconfigure(int compressionThreshold, string compressionCodec);
};

template <typename T, typename T_C>
CpuSimd<T, T_C>::CpuSimd()
{
    SIMDCOMPRESSION_THRESHOLD = 64; // use 0xffffff (2^32) to transparently disable
    codecName = "s4-bp128-dm";
    codec = *CODECFactory::getFromName(codecName);
}

template <typename T, typename T_C>
void CpuSimd<T, T_C>::reconfigure(int compressionThreshold, string compressionCodec)
{
    assert(compressionThreshold > 0);
    assert(compressionCodec.length() > 0);
    codecName = compressionCodec;
    codec = *CODECFactory::getFromName(codecName);
    SIMDCOMPRESSION_THRESHOLD = static_cast<uint32_t>(compressionThreshold);
}

template <typename T, typename T_C>
void CpuSimd<T, T_C>::debugCompression(T *fq, const size_t size) const
{
	assert(fq != NULL);
        assert(size >= 0);

/*
    if (size > 0)
    {
        size_t compressedsize, uncompressedsize = static_cast<size_t>(size);
        T_C *compressed_fq;
	T *uncompressed_fq;
        high_resolution_clock::time_point time_0, time_1;
        time_0 = high_resolution_clock::now();
        compress(fq, uncompressedsize, &compressed_fq, compressedsize);
        time_1 = high_resolution_clock::now();
        long encode_time = chrono::duration_cast<chrono::nanoseconds>(time_1 - time_0).count();
        time_0 = high_resolution_clock::now();
        decompress(compressed_fq, compressedsize, &uncompressed_fq, uncompressedsize);
        time_1 = high_resolution_clock::now();
        long decode_time = chrono::duration_cast<chrono::nanoseconds>(time_1 - time_0).count();
        verifyCompression(fq, uncompressed_fq, uncompressedsize);
        if (isCompressed(uncompressedsize, compressedsize))
        {
            free(compressed_fq);
            free(uncompressed_fq);
        }
*/
        /**
         * Check validity of results
         */
/*
        double compressionratio = (static_cast<double>(compressedsize) / static_cast<double>(uncompressedsize));
        double compresspercent = (100.0 - 100.0 * compressionratio);
        long dataSize = (size * sizeof(int));
        string dataUnit;
        if (dataSize < 1000)
        {
            dataUnit = "B";
        }
        else if (dataSize < 1000000)
        {
            dataUnit = "KB";
            dataSize /= 1000;
        }
        else if (dataSize < 1000000000)
        {
            dataUnit = "MB";
            dataSize /= 1000;
        }
        else if (dataSize < 1000000000000)
        {
            dataUnit = "GB";
            dataSize /= 1000;
        }
        printf("debug:: cpu-simd (%s), data: %ld%s, c/d: %04ld/%04ldus, %02.3f%% gained\n", codecName.c_str(), dataSize,
               dataUnit.c_str(), encode_time, decode_time,
               compresspercent);
    }
*/
}

template <typename T, typename T_C>
inline bool CpuSimd<T, T_C>::compress(T *fq_64, const size_t &size, T_C **compressed_fq_32,
                               size_t &compressedsize) const
{
    bool compressed;
    if (isCompressible(size))
    //if (size > 0)
    {
        /*
	int err1, err2;
        T_C *fq_32; //= (T_C *)malloc(size * sizeof(T_C));
	err1 = posix_memalign((void **)&fq_32, 16, size * sizeof(T_C));
        //*compressed_fq_32 = (T_C *)malloc(size * sizeof(T_C));
	err2 = posix_memalign((void **)compressed_fq_32, 16, size * sizeof(T_C));
        if (err1 || err2)
        {
            printf("\nERROR: Memory allocation error!");
            abort();
        }

        compressedsize = size;
        for (int i = 0; i < size; ++i)
        {
            fq_32[i] = static_cast<T_C>(fq_64[i]);
        }
        codec.encodeArray(fq_32, size, *compressed_fq_32, compressedsize);

	//std::cout << "-- compressed buffer (" << compressedsize << ")--" << std::endl;
        for (int i=0; i< compressedsize; ++i){
	//std::cout << (*compressed_fq_32)[i];
	//printf("%ui", fq_64[i]);
	}
	//std::cout << std::endl;

        // if this condition is met it can not be known whether or not there has been a compression.
        // Todo: find solution
        assert(compressedsize < size);
        /-*
        *compressed_fq_64 = NULL;
        *compressed_fq_64 = (T_C *)malloc(compressedsize * sizeof(T_C));
        if (*compressed_fq_64 == NULL)
        {
            printf("\nERROR: Memory allocation error!");
            abort();
        }
        for (size_t i = 0; i < compressedsize; ++i)
        {
            (*compressed_fq_64)[i] = static_cast<T_C>(compressed_fq_32[i]);
        }
	*-/
        free(fq_32);
        // free(compressed_fq_32);
       */
        ////std::cout << "[c]-->origsize: " << size << " compressedsize:" << compressedsize << std::endl;
        ///
        T_C *fq_32 = (T_C *)malloc(size * sizeof(T_C));
        *compressed_fq_32 = (T_C *)malloc(size * sizeof(T_C));
        for (int i = 0; i < size; ++i)
        {
            // (*compressed_fq_32)[i] = static_cast<T_C>(fq_64[i]);
            fq_32[i] = static_cast<T_C>(fq_64[i]);
        }
        compressedsize = size;
        codec.encodeArray(fq_32, size, *compressed_fq_32, compressedsize);
	   compressed = true;
        free(fq_32);
    }
    else
    {
        /**
         * Buffer will not be compressed (Small size. Not worthed)
         */
	/*int err;
	err = posix_memalign((void **)compressed_fq_32, 16, size * sizeof(T_C));
	if (err) {
		printf("memory error!\n");
		abort();
	}*/
	   *compressed_fq_32 = (T_C *)malloc(size * sizeof(T_C));
        for (int i = 0; i < size; ++i)
        {
            (*compressed_fq_32)[i] = static_cast<T_C>(fq_64[i]);
        }
        compressedsize = size;
        // *compressed_fq_64 = reinterpret_cast<T_C *>(fq_64);
	    compressed = false;
    }
    return compressed;
}

template <typename T, typename T_C>
inline bool CpuSimd<T, T_C>::decompress(T_C *compressed_fq_32, const int size,
                                 /*Out*/ T **uncompressed_fq_64, /*In Out*/size_t &uncompressedsize) const
{
    bool compressed;
    if (isCompressed(uncompressedsize, size))
    //if (size > 0)
    {
	/*int err;
        T_C *uncompressed_fq_32; // = (T_C *) malloc(uncompressedsize * sizeof(T_C));
	err = posix_memalign((void **)&uncompressed_fq_32, 16, uncompressedsize * sizeof(T_C));
	if (err) {
		printf("memory error!\n");
		abort();
	}
        /-*
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
        *-/
	//memcpy(compressed_fq_32, compressed_fq, size * sizeof(uint32_t));
        //*compressed_fq_32 = reinterpret_cast<uint32_t *>(compressed_fq);

        //std::cout << "-- compressed buffer, prior to decompress ("<< size <<")--" << std::endl;
        for (int i=0; i< size; ++i){
        //std::cout << compressed_fq_32[i];
        //printf("%ui", compressed_fq_32[i]);
        }
        //std::cout << std::endl;

        codec.decodeArray(compressed_fq_32, size, uncompressed_fq_32, uncompressedsize);
	err = posix_memalign((void **)uncompressed_fq_64, 16, uncompressedsize * sizeof(T));
        // *uncompressed_fq_64 = (T *)malloc(uncompressedsize * sizeof(T));
        if (err)
        {
            printf("\nERROR: Memory allocation error!");
            abort();
        }
        // memcpy((T *)uncompressed_fq_64, (uint32_t *)uncompressed_fq_32, uncompressedsize * sizeof(uint32_t));
        for (size_t i = 0; i < uncompressedsize; ++i)
        {
            (*uncompressed_fq_64)[i] = static_cast<T>(uncompressed_fq_32[i]);
        }
        //free(compressed_fq_32);
        free(uncompressed_fq_32);
        ////std::cout << "[d]-->origsize: " << uncompressedsize << " compressedsize:" << size << std::endl;
        */
        T_C *uncompressed_fq_32 = (T_C *) malloc(uncompressedsize * sizeof(T_C));
        *uncompressed_fq_64 = (T *)malloc(uncompressedsize * sizeof(T));
        codec.decodeArray(compressed_fq_32, size, uncompressed_fq_32, uncompressedsize);
        for (int i = 0; i < uncompressedsize; ++i)
        {
            // (*uncompressed_fq_64)[i] = static_cast<T>(compressed_fq_32[i]);
            (*uncompressed_fq_64)[i] = static_cast<T>(uncompressed_fq_32[i]);
        }

        free(uncompressed_fq_32);
	    compressed = true;
    }
    else
    {
        /**
         * Buffer was not compressed (Small size. Not worthed)
         */
	//int err;
        uncompressedsize = size;
	    compressed = false;
        //*uncompressed_fq_64 = reinterpret_cast<T *>(compressed_fq_64);
	/*err = posix_memalign((void **)uncompressed_fq_64, 16, uncompressedsize * sizeof(T));
	if (err) {
		printf("memory error!\n");
		abort();
	}*/
	    *uncompressed_fq_64 = (T *)malloc(uncompressedsize * sizeof(T));
        for (int i = 0; i < size; ++i)
        {
	       (*uncompressed_fq_64)[i] = static_cast<T>(compressed_fq_32[i]);
        }
    }
	return compressed;
}

template <typename T, typename T_C>
void CpuSimd<T, T_C>::verifyCompression(const T *fq, const T *uncompressed_fq_64,
                                        const size_t uncompressedsize) const
{
    if (isCompressible(uncompressedsize))
    {
        assert(memcmp(fq, uncompressed_fq_64, uncompressedsize * sizeof(T)) == 0);
    }
}


template <typename T, typename T_C>
inline bool CpuSimd<T, T_C>::isCompressed(const size_t originalsize, const size_t compressedsize) const
{

    return (isCompressible(originalsize) && originalsize != compressedsize);
}


template <typename T, typename T_C>
inline string CpuSimd<T, T_C>::name() const
{
    return "cpusimd";
}


#endif // _SIMD
#endif // BFS_MULTINODE_CPUSIMD_COMPRESSION_H
