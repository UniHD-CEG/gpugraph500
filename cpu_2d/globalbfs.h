#include "distmatrix2d.h"
#include <vector>

#ifndef GLOBALBFS_H
#define GLOBALBFS_H

/*
 * This classs implements a distributed level synchronus BFS on global scale.
 */
class GlobalBFS
{
    MPI_Comm row_comm, col_comm;

    // sending node column slice, startvtx, size
    std::vector<DistMatrix2d::fold_prop> fold_fq_props;

protected:
    DistMatrix2d& store;
    vtxtype* predessor;

    MPI_Datatype fq_tp_type; //Frontier Queue Transport Type
    void*   recv_fq_buff;
    long    recv_fq_buff_length;
    virtual void reduce_fq_out(void* startaddr, long insize)=0;    //Global Reducer of the local outgoing frontier queues.  Have to be implemented by the children.
    virtual void getOutgoingFQ(void* &startaddr, vtxtype& outsize)=0;
    virtual void setModOutgoingFQ(void* startaddr, long insize)=0; //startaddr: 0, self modification
    virtual void getOutgoingFQ(vtxtype globalstart, vtxtype size, void* &startaddr, vtxtype& outsize)=0;
    virtual void setIncommingFQ(vtxtype globalstart, vtxtype size, void* startaddr, vtxtype& insize_max)=0;
    virtual bool istheresomethingnew()=0;           //to detect if finished
    virtual void setStartVertex(vtxtype start)=0;
    virtual void runLocalBFS()=0;


public:
    GlobalBFS(DistMatrix2d& _store);

    #ifdef INSTRUMENTED
    void runBFS(vtxtype startVertex, double& lexp, double &lqueue);
    #else
    void runBFS(vtxtype startVertex);
    #endif


    vtxtype* getPredessor();
};

#endif // GLOBALBFS_H
