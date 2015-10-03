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
static map<string, shared_ptr<Compression<Tp>>> initializeCompressfactory<Tp>()
{
    map <string, shared_ptr<Compression<Tp>>> schemes;
    // SIMD compression algorthm performed on CPU
    schemes["cpusimd"] = shared_ptr<Compression<Tp>>(new CpuSimd<Tp>());
    // SIMT compression algorthm performed on GPU
    // schemes["gpusimt"] = shared_ptr<Compression<Tp>>(new GpuSimt<Tp>());

    return schemes;
}




template <typename Tc>
class CompressionFactory
{
public:
    static map<string, shared_ptr<Compression<Tc>>> compressionschemes;
    static shared_ptr<Compression<Tc>> defaultptr;

    static string getName(Compression<Tc> &compression)
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

    static shared_ptr<Compression<Tc>> &getFromName(string name)
    {
        if (!valid(name))
        {
            cerr << "name " << name << " does not refer to a Compression Scheme." << endl;
            return defaultptr;
        }
        return compressionschemes[name];
    }
};
template <typename Tc>
map<string, shared_ptr<Compression<Tc>>> CompressionFactory<Tc>::compressionschemes = initializeCompressfactory<Tc>();
template <typename Tc>
shared_ptr<Compression<Tc>> CompressionFactory<Tc>::defaultptr = shared_ptr<Compression<Tc>>(nullptr);

//} // CompresionNamespace

#endif // BFS_MULTINODE_COMPRESSIONFACTORY_H
