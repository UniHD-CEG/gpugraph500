#include <vector>

#include "globalbfs.hh"

#ifndef SIMPLECPUBFS_H
#define SIMPLECPUBFS_H

typedef long long vtxtyp;
typedef int       rowtyp;

class SimpleCPUBFS : public GlobalBFS<SimpleCPUBFS,void,unsigned char,DistMatrix2d<vtxtyp,rowtyp,false, 1> >
{
    std::vector<bool> visited;
    std::vector<vtxtyp> fq_out;
    std::vector<vtxtyp> fq_in;
public:
    typedef DistMatrix2d<vtxtyp,rowtyp,false, 1> MatrixT;
    SimpleCPUBFS(MatrixT &_store, int64_t verbosity);
    ~SimpleCPUBFS();

    void reduce_fq_out(void* startaddr, long insize);    //Global Reducer of the local outgoing frontier queues.  Have to be implemented by the children.
    void getOutgoingFQ(void *&startaddr, long& outsize);
    void setModOutgoingFQ(void* startaddr, long insize); //startaddr: 0, self modification
    void getOutgoingFQ(vtxtyp globalstart, long size, void* &startaddr, long& outsize);
    void setIncommingFQ(vtxtyp globalstart, long size, void* startaddr, long& insize_max);
    bool istheresomethingnew();           //to detect if finished
    void setStartVertex(const vtxtyp start);
    void runLocalBFS();
};

#endif // SIMPLECPUBFS_H
