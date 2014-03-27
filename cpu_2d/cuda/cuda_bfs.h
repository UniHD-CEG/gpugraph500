#include "../b40c/graph/bfs/csr_problem.cuh"
#include "../b40c/graph/bfs/enactor_multi_gpu.cuh"
#include "../globalbfs.hh"

#ifndef CUDA_BFS_H
#define CUDA_BFS_H

using namespace b40c::graph::bfs;

class CUDA_BFS : public GlobalBFS<CUDA_BFS,uint64_T, DistMatrix2d<false, 1> >
{
    CsrProblem csr_problem;
    EnactorMultiGpu<INSTRUMENT> bfsGPU;

public:
    typedef DistMatrix2d<false, 1> MatrixT;
    CUDA_BFS(MatrixT &_store);

    void reduce_fq_out(uint64_t* startaddr, long insize);    //Global Reducer of the local outgoing frontier queues.  Have to be implemented by the children.
    void getOutgoingFQ(uint64_t* &startaddr, vtxtype& outsize);
    void setModOutgoingFQ(uint64_t* startaddr, long insize); //startaddr: 0, self modification
    void getOutgoingFQ(vtxtype globalstart, vtxtype size, uint64_t* &startaddr, vtxtype& outsize);
    void setIncommingFQ(vtxtype globalstart, vtxtype size, uint64_t* startaddr, vtxtype& insize_max);
    bool istheresomethingnew();           //to detect if finished
    void setStartVertex(vtxtype start);
    void runLocalBFS();
};

#endif // CUDA_BFS_H
