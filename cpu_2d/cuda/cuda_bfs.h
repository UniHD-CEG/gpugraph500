#include "../globalbfs.hh"

#ifndef CUDA_BFS_H
#define CUDA_BFS_H

class CUDA_BFS : public GlobalBFS<false,1>
{
public:
    typedef DistMatrix2d<false, 1> MatrixT;
    CUDA_BFS(MatrixT &_store);

    virtual void reduce_fq_out(void* startaddr, long insize);    //Global Reducer of the local outgoing frontier queues.  Have to be implemented by the children.
    virtual void getOutgoingFQ(void* &startaddr, vtxtype& outsize);
    virtual void setModOutgoingFQ(void* startaddr, long insize); //startaddr: 0, self modification
    virtual void getOutgoingFQ(vtxtype globalstart, vtxtype size, void* &startaddr, vtxtype& outsize);
    virtual void setIncommingFQ(vtxtype globalstart, vtxtype size, void* startaddr, vtxtype& insize_max);
    virtual bool istheresomethingnew();           //to detect if finished
    virtual void setStartVertex(vtxtype start);
    virtual void runLocalBFS();
};

#endif // CUDA_BFS_H
