#ifndef CUDA_BFS_H
#define CUDA_BFS_H

#define __restrict__
#ifdef __CUDACC__
#include "cuda_support.hh" //for enactor_base.cuh
#include "b40c/graph/bfs/csr_problem_2d.cuh"
#include "b40c/graph/bfs/enactor_multi_gpu_2d.cuh"
#else
namespace b40c
{
namespace graph
{
namespace bfs
{
template<typename _VertexId, typename _SizeT, bool MARK_PREDECESSORS>
struct CsrProblem;
template<typename Csr, bool INSTRUMENT>
class EnactorMultiGpu;
}
}
}
#endif

#include "../globalbfs.hh"

#ifdef _DEBUG
#include "../validate/checkqueue.h"
#endif

using std::string;
using namespace b40c::graph::bfs;

//cuda types have to be chosen, what might be a problem
typedef long long vertexType;
typedef unsigned int rowtyp;

class CUDA_BFS : public GlobalBFS <CUDA_BFS, vertexType, unsigned char,
    DistMatrix2d<vertexType, rowtyp, true, 1, true>  // use local ids
    >
{
private:
    typedef unsigned char MType;
    typedef CsrProblem <vertexType, rowtyp, true> Csr;
    int64_t verbosity;
    double queue_sizing;
    uint64_t qb_length, rb_length;
    vertexType *__restrict__ queuebuff;
    vertexType *__restrict__ redbuff;
    //Csr::VisitedMask** __restrict__ vmask;
    MType *__restrict__ vmask;
    bool done;
    Csr *csr_problem;

#ifdef INSTRUMENTED
    EnactorMultiGpu<Csr, true> *bfsGPU;
#else
    EnactorMultiGpu<Csr, false> *bfsGPU;
#endif

#ifdef _DEBUG
    CheckQueue<vertexType> checkQueue;
#endif

protected:
public:
    typedef DistMatrix2d<vertexType, rowtyp, true, 1, true> MatrixT;
    CUDA_BFS(MatrixT &_store, int &num_gpus, double _queue_sizing, int64_t _verbosity);
    ~CUDA_BFS();
    void getBackPredecessor();
    void getBackOutqueue();
    void setBackInqueue();
    void generatOwenMask() { }
    void reduce_fq_out(vertexType globalstart, long size, vertexType *startaddr,
                       int insize);    //Global Reducer of the local outgoing frontier queues.  Have to be implemented by the children.
    void getOutgoingFQ(vertexType *&startaddr, int &outsize);
    void setModOutgoingFQ(vertexType *startaddr, int insize); //startaddr: 0, self modification
    void getOutgoingFQ(vertexType globalstart, long size, vertexType *&startaddr, int &outsize);
    void setIncommingFQ(vertexType globalstart, long size, vertexType *startaddr, int &insize_max);
    bool istheresomethingnew();           //to detect if finished
    void setStartVertex(vertexType start);
    void runLocalBFS();
    void bfsMemCpy(vertexType *&dst, vertexType *src, size_t size);
};

#endif // CUDA_BFS_H
