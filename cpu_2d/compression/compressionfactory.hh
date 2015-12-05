#ifndef BFS_MULTINODE_COMPRESSIONFACTORY_H
#define BFS_MULTINODE_COMPRESSIONFACTORY_H

#include <iostream>
#include <memory>
#include <map>
#include <string>
#include "compression.hh"
#if defined(_SIMD)
#include "cpusimd.hh"
#elif defined (_SIMD_PLUS)
#include "simdplus.hh"
#else
#include "nocompression.hh"
#endif

using std::map;
using std::string;
using std::shared_ptr;
using std::cerr;
using std::endl;

template <typename T, typename T_C>
static map<string, shared_ptr<Compression<T, T_C>>> initializeCompressfactory()
{
    map <string, shared_ptr<Compression<T, T_C>>> schemes;

#if defined(_SIMD) // Lemire's
    schemes["cpusimd"] = shared_ptr<Compression<T, T_C>>(new CpuSimd<T, T_C>());
#elif defined(_SIMD_PLUS) // Turbo-PFOR
    schemes["simdplus"] = shared_ptr<Compression<T, T_C>>(new SimdPlus<T, T_C>());
#elif defined(_SIMT) // CUDA
    schemes["gpusimt"] = shared_ptr<Compression<T, T_C>>(new GpuSimt<T, T_C>());
#else
    schemes["nocompression"] = shared_ptr<Compression<T, T_C>>(new NoCompression<T, T_C>());
#endif
    return schemes;
}


template <typename T, typename T_C>
class CompressionFactory
{
public:
    static map<string, shared_ptr<Compression<T, T_C>>> compressionschemes;
    static shared_ptr<Compression<T, T_C>> defaultptr;


    static shared_ptr<Compression<T, T_C>> &getFromName(string name)
    {
        if (compressionschemes.find(name) == compressionschemes.end())
        {
            cerr << "name " << name << " does not refer to a Compression Scheme." << endl;
            return defaultptr;
        };
        return compressionschemes[name];
    }
};
template <typename T, typename T_C>
map<string, shared_ptr<Compression<T, T_C>>> CompressionFactory<T, T_C>::compressionschemes =
    initializeCompressfactory<T, T_C>();
template <typename T, typename T_C>
shared_ptr<Compression<T, T_C>> CompressionFactory<T, T_C>::defaultptr = shared_ptr<Compression<T, T_C>>(nullptr);


#endif // BFS_MULTINODE_COMPRESSIONFACTORY_H
