#include "globalbfs.hh"

#ifndef CPUBFS_BIN_H
#define CPUBFS_BIN_H

class CPUBFS_bin : public GlobalBFS<CPUBFS_bin,uint64_t,DistMatrix2d<true, 64> >
{
    const int64_t col64;
    const int64_t row64;

    uint64_t*  __restrict__ visited;
    uint64_t*  __restrict__ fq_out;
    uint64_t*  __restrict__ fq_in;

public:
    typedef DistMatrix2d<true, 64> MatrixT;
    CPUBFS_bin(MatrixT &_store);
    ~CPUBFS_bin();

    void reduce_fq_out(uint64_t* startaddr, long insize);    //Global Reducer of the local outgoing frontier queues.  Have to be implemented by the children.
    void getOutgoingFQ(uint64_t *&startaddr, vtxtype& outsize);
    void setModOutgoingFQ(uint64_t* startaddr, long insize); //startaddr: 0, self modification
    void getOutgoingFQ(vtxtype globalstart, vtxtype size, uint64_t* &startaddr, vtxtype& outsize);
    void setIncommingFQ(vtxtype globalstart, vtxtype size, uint64_t* startaddr, vtxtype& insize_max);
    bool istheresomethingnew();           //to detect if finished
    void setStartVertex(const vtxtype start);
    void runLocalBFS();
};

#endif // CPUBFS_BIN_H
