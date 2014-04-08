#ifdef __CUDACC__
#include "cuda_support.hh" //for enactor_base.cuh

#include "b40c/graph/bfs/csr_problem_2d.cuh"
#include "b40c/graph/bfs/enactor_multi_gpu_2d.cuh"
#else
namespace b40c {
namespace graph {
namespace bfs {

template < typename _VertexId, typename _SizeT,bool MARK_PREDECESSORS>
struct CsrProblem;

template <bool INSTRUMENT>
class EnactorMultiGpu;
}
}
}
#endif

#include "../globalbfs.hh"

#ifndef CUDA_BFS_H
#define CUDA_BFS_H

using namespace b40c::graph::bfs;

//cuda types have to be chosen, what might be a problem
typedef long long vtxtyp;
typedef int       rowtyp;

class CUDA_BFS : public GlobalBFS<  CUDA_BFS,
                                    vtxtyp,
                                    DistMatrix2d<vtxtyp, rowtyp, true, 1>  // use local ids
                                  >
{
    typedef CsrProblem<vtxtyp,
                       rowtyp,
                       true> Csr;

    double queue_sizing;
    uint64_t qb_length, rb_length;
    vtxtyp* __restrict__ queuebuff;
    vtxtyp* __restrict__ redbuff;

    //Csr::VisitedMask** __restrict__ vmask;
    unsigned char** __restrict__ vmask;

    bool newElements;

    Csr* csr_problem;
#ifdef INSTRUMENTED
    EnactorMultiGpu<true>* bfsGPU;
#else
    EnactorMultiGpu<false>* bfsGPU;
#endif

public:
    typedef DistMatrix2d<vtxtyp, rowtyp, true, 1> MatrixT;
    CUDA_BFS(MatrixT &_store, int num_gpus, double _queue_sizing);
    ~CUDA_BFS();

    void getBackPredecessor();
    void getBackOutqueue();
    void setBackInqueue();

    void reduce_fq_out(vtxtyp* startaddr, long insize);    //Global Reducer of the local outgoing frontier queues.  Have to be implemented by the children.
    void getOutgoingFQ(vtxtyp* &startaddr, long& outsize);
    void setModOutgoingFQ(vtxtyp* startaddr, long insize); //startaddr: 0, self modification
    void getOutgoingFQ(vtxtyp globalstart, long size, vtxtyp* &startaddr, long& outsize);
    void setIncommingFQ(vtxtyp globalstart, long size, vtxtyp* startaddr, long& insize_max);
    bool istheresomethingnew();           //to detect if finished
    void setStartVertex(vtxtyp start);
    void runLocalBFS();
};

#endif // CUDA_BFS_H
