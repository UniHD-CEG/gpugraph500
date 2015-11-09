#include "cuda\cuda_bfs.h"

CUDA_BFS::CUDA_BFS(MatrixT &_store): GlobalBFS<false, 1>(_store)
{
}

void CUDA_BFS::reduce_fq_out(uint64_t *startaddr, long insize)
{
}

void CUDA_BFS::getOutgoingFQ(uint64_t *&startaddr, vertexTypee &outsize)
{
}

void CUDA_BFS::setModOutgoingFQ(uint64_t *startaddr, long insize)
{
}

void CUDA_BFS::getOutgoingFQ(vertexTypee globalstart, vertexTypee size, uint64_t *&startaddr, vertexTypee &outsize)
{
}

void CUDA_BFS::setIncommingFQ(vertexTypee globalstart, vertexTypee size, uint64_t *startaddr, vertexTypee &insize_max)
{
}

bool CUDA_BFS::istheresomethingnew()
{
}

void CUDA_BFS::setStartVertex(vertexTypee start)
{
}

void CUDA_BFS::runLocalBFS()
{
}

void bfsMemCpy(vertexType *&dst, vertexType *src, size_t size)
{
}
