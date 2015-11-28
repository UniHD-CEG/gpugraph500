#ifndef BFS_MULTINODE_COMPRESSIONFACTORY_H
#define BFS_MULTINODE_COMPRESSIONFACTORY_H

#include <iostream>
#include <memory>
#include <map>
#include <string>
#include "compression.hh"
//#ifdef _SIMD
//#include "cpusimd.hh"
//#endif
#include "nocompression.hh"

using std::map;
using std::string;
using std::shared_ptr;
using std::cerr;
using std::endl;

template <typename T, typename T_C>
static map<string, shared_ptr<Compression<T, T_C>>> initializeCompressfactory()
{
    map <string, shared_ptr<Compression<T, T_C>>> schemes;

#ifdef _SIMD // CPU-SIMD
    schemes["cpusimd"] = shared_ptr<Compression<T, T_C>>(new CpuSimd<T, T_C>());
#else
#ifdef _SIMD_IMPROVED // CPU-SIMD_IMPROVED
#else
    schemes["improvedsimd"] = shared_ptr<Compression<T, T_C>>(new CpuImprovedSimd<T, T_C>());
#ifdef _SIMT // GPU-SIMT
    schemes["gpusimt"] = shared_ptr<Compression<T, T_C>>(new GpuSimt<T, T_C>());
#endif
#endif
#endif
    schemes["nocompression"] = shared_ptr<Compression<T, T_C>>(new NoCompression<T, T_C>());

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
