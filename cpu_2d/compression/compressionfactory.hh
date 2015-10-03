#ifndef BFS_MULTINODE_COMPRESSIONFACTORY_H
#define BFS_MULTINODE_COMPRESSIONFACTORY_H

#include <map>
#include "compression.hh"
#include "cpusimd.hh"

//namespace CompressionNamespace
//{


using std::map;
using std::string;
using std::shared_ptr;
using std::cerr;
using std::endl;

template <typename Tp>
static map<string, shared_ptr<Compression<Tp>>> initializefactory()
{
    map <string, shared_ptr<Compression<Tp>>> schemes;
    // SIMD compression algorthm performed on CPU
    schemes["cpusimd"] = shared_ptr<Compression<Tp>>(new CpuSimd<Tp>());
    // SIMT compression algorthm performed on GPU
    // schemes["gpusimt"] = shared_ptr<Compression<T>>(new GpuSimt<T>());

    return schemes;
}




template <typename T>
class CompressionFactory
{
public:
    static map<string, shared_ptr<Compression<T>>> compressionschemes;
    static shared_ptr<Compression<T>> defaultptr;

    static string getName(Compression<T> &compression)
    {
        for (auto i = compressionschemes.begin(); i != compressionschemes.end() ; ++i)
        {
            if (i->second.get() == &compression)
            {
                return i->first;
            }
        }
        return "unknown";
    }

    static bool valid(string name)
    {
        return (compressionschemes.find(name) != compressionschemes.end()) ;
    }

    static shared_ptr<Compression<T>> &getFromName(string name)
    {
        if (!valid(name))
        {
            cerr << "name " << name << " does not refer to a Compression Scheme." << endl;
            return defaultptr;
        }
        return compressionschemes[name];
    }
};
template <typename T>
map<string, shared_ptr<Compression<T>>> CompressionFactory<T>::compressionschemes = initializefactory();
template <typename T>
shared_ptr<Compression<T>> CompressionFactory<T>::defaultptr = shared_ptr<Compression<T>>(nullptr);

//} // CompresionNamespace

#endif // BFS_MULTINODE_COMPRESSIONFACTORY_H
