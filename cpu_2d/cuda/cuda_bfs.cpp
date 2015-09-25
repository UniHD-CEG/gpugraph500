#include "cuda\cuda_bfs.h"

CUDA_BFS::CUDA_BFS(MatrixT &_store):GlobalBFS<false,1>(_store)
{
}

void CUDA_BFS::reduce_fq_out(uint64_t *startaddr, long insize)
{
}

void CUDA_BFS::getOutgoingFQ(uint64_t *&startaddr, vtxtype &outsize)
{
}

void CUDA_BFS::setModOutgoingFQ(uint64_t *startaddr, long insize)
{
}

void CUDA_BFS::getOutgoingFQ(vtxtype globalstart, vtxtype size, uint64_t *&startaddr, vtxtype &outsize)
{
}

void CUDA_BFS::setIncommingFQ(vtxtype globalstart, vtxtype size, uint64_t *startaddr, vtxtype &insize_max)
{
}

bool CUDA_BFS::istheresomethingnew()
{
}

void CUDA_BFS::setStartVertex(vtxtype start)
{
}

void CUDA_BFS::runLocalBFS()
{
}

void bfsMemCpy(vtxtyp *&dst, vtxtyp *src, size_t size)
{
}
