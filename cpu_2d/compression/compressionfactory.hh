#ifndef BFS_MULTINODE_COMPRESSIONFACTORY_H
#define BFS_MULTINODE_COMPRESSIONFACTORY_H

#include <iostream>
#include <memory>
#include <map>
#include <string>
#include "compression.hh"
#ifdef _SIMD
#include "cpusimd.hh"
#endif
#include "nocompression.hh"

using std::map;
using std::string;
using std::shared_ptr;
using std::cerr;
using std::endl;

template <typename T>
static map<string, shared_ptr<Compression<T>>> initializeCompressfactory()
{
    map <string, shared_ptr<Compression<T>>> schemes;

#ifdef _SIMD // CPU-SIMD
    schemes["cpusimd"] = shared_ptr<Compression<T>>(new CpuSimd<T>());
#else
#ifdef _SIMTCOMPRESS // GPU-SIMT
    schemes["gpusimt"] = shared_ptr<Compression<T>>(new GpuSimt<T>());
#endif
#endif
    schemes["nocompression"] = shared_ptr<Compression<T>>(new NoCompression<T>());

    return schemes;
}


template <typename T>
class CompressionFactory
{
public:
    static map<string, shared_ptr<Compression<T>>> compressionschemes;
    static shared_ptr<Compression<T>> defaultptr;


    static shared_ptr<Compression<T>> &getFromName(string name)
    {
        if (compressionschemes.find(name) == compressionschemes.end())
        {
            cerr << "name " << name << " does not refer to a Compression Scheme." << endl;
            return defaultptr;
        };
        return compressionschemes[name];
    }
};
template <typename T>
map<string, shared_ptr<Compression<T>>> CompressionFactory<T>::compressionschemes = initializeCompressfactory<T>();
template <typename T>
shared_ptr<Compression<T>> CompressionFactory<T>::defaultptr = shared_ptr<Compression<T>>(nullptr);


#endif // BFS_MULTINODE_COMPRESSIONFACTORY_H
