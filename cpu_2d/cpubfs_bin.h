#include "globalbfs.h"

#ifndef CPUBFS_BIN_H
#define CPUBFS_BIN_H

class CPUBFS_bin : public GlobalBFS
{
    uint8_t* visited;
    uint8_t* fq_out;
    uint8_t* fq_in;

public:
    CPUBFS_bin(DistMatrix2d &_store);
    ~CPUBFS_bin();

    void reduce_fq_out(void* startaddr, long insize);    //Global Reducer of the local outgoing frontier queues.  Have to be implemented by the children.
    void getOutgoingFQ(void *&startaddr, vtxtype& outsize);
    void setModOutgoingFQ(void* startaddr, long insize); //startaddr: 0, self modification
    void getOutgoingFQ(vtxtype globalstart, vtxtype size, void* &startaddr, vtxtype& outsize);
    void setIncommingFQ(vtxtype globalstart, vtxtype size, void* startaddr, vtxtype& insize_max);
    bool istheresomethingnew();           //to detect if finished
    void setStartVertex(vtxtype start);
    void runLocalBFS();
};

#endif // CPUBFS_BIN_H
