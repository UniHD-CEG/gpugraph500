#include <vector>

#include "globalbfs.hh"

#ifndef SIMPLECPUBFS_H
#define SIMPLECPUBFS_H

class SimpleCPUBFS : public GlobalBFS<false,1>
{
    std::vector<bool> visited;
    std::vector<vtxtype> fq_out;
    std::vector<vtxtype> fq_in;
public:
    typedef DistMatrix2d<false, 1> MatrixT;
    SimpleCPUBFS(MatrixT &_store);
    ~SimpleCPUBFS();

    void reduce_fq_out(void* startaddr, long insize);    //Global Reducer of the local outgoing frontier queues.  Have to be implemented by the children.
    void getOutgoingFQ(void *&startaddr, vtxtype& outsize);
    void setModOutgoingFQ(void* startaddr, long insize); //startaddr: 0, self modification
    void getOutgoingFQ(vtxtype globalstart, vtxtype size, void* &startaddr, vtxtype& outsize);
    void setIncommingFQ(vtxtype globalstart, vtxtype size, void* startaddr, vtxtype& insize_max);
    bool istheresomethingnew();           //to detect if finished
    void setStartVertex(vtxtype start);
    void runLocalBFS();
};

#endif // SIMPLECPUBFS_H
