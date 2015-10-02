#ifndef BFS_MULTINODE_COMPRESSIONFACTORY_H
#define BFS_MULTINODE_COMPRESSIONFACTORY_H

#include <map>
#include "compression.hh"
#include "cpusimd.hh"

using std::map;
using std::string;
using std::shared_ptr;
using std::cerr;
using std::endl;
using std::nullptr;

static map<string, shared_ptr<Compression>> initializefactory()
{
    map <string, shared_ptr<Compression>> schemes;
    schemes["cpusimd"] = shared_ptr<Compression>(new CpuSimd<FQ_T>());
    return schemes;
}

class Factory
{
public:
    static map<string, shared_ptr<Compression>> compressionschemes;
    static shared_ptr<Compression> defaultptr;

    static string getName(Compression &compression)
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

    static shared_ptr<Compression> &getFromName(string name)
    {
        if (!valid(name))
        {
            cerr << "name " << name << " does not refer to a Compression Scheme." << endl;
            return defaultptr;
        }
        return compressionschemes[name];
    }
};
map<string, shared_ptr<Compression>> Factory::compressionschemes = initializefactory();
shared_ptr<Compression> Factory::defaultptr = shared_ptr<Compression>(nullptr);


#endif // BFS_MULTINODE_COMPRESSIONFACTORY_H
