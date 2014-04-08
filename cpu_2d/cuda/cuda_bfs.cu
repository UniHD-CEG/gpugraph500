#include "cuda_bfs.h"

#include "b40c/util/error_utils.cuh"

#include <cstdlib>
#include <algorithm>
#include <functional>


CUDA_BFS::CUDA_BFS(MatrixT &_store,int num_gpus,double _queue_sizing):GlobalBFS< CUDA_BFS,vtxtyp,MatrixT>(_store), queue_sizing(_queue_sizing)
{
    //expect symetrie
    if(store.getNumRowSl() != store.getNumColumnSl()){
        fprintf(stderr,"Currently the partitioning has to be symetric.");
        exit(1);
    }
    predessor    = new vtxtyp[store.getLocColLength()];

    fq_tp_type = MPI_INT64_T;
    recv_fq_buff_length = std::max(store.getLocRowLength(), store.getLocColLength())+num_gpus;
    recv_fq_buff = new vtxtyp[recv_fq_buff_length];
    //multipurpose buffer
    qb_length    = 0;
    queuebuff    = new vtxtyp[recv_fq_buff_length];
    rb_length    = 0;
    redbuff      = new vtxtyp[recv_fq_buff_length];

    newElements = false;


    // Enable symmetric peer access between gpus
    // from test_bfs.cu
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        for (int other_gpu = (gpu + 1) % num_gpus;
             other_gpu != gpu;
             other_gpu = (other_gpu + 1) % num_gpus)
        {
                // Set device
                if (b40c::util::B40CPerror(cudaSetDevice(gpu),
                    "MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

                printf("Enabling peer access to GPU %d from GPU %d\n", other_gpu, gpu);

                cudaError_t error = cudaDeviceEnablePeerAccess(other_gpu, 0);
                if ((error != cudaSuccess) && (error != cudaErrorPeerAccessAlreadyEnabled)) {
                    fprintf(stderr, "MultiGpuBfsEnactor cudaDeviceEnablePeerAccess failed", __FILE__, __LINE__);
                }
        }
    }

   csr_problem = new Csr;
#ifdef INSTRUMENTED
   bfsGPU = new EnactorMultiGpu<true>;
#else
   bfsGPU = new EnactorMultiGpu<false>;
#endif


    csr_problem->FromHostProblem(
                false,                  //bool          stream_from_host,
                store.getLocRowLength(),//SizeT 		nodes,
                store.getEdgeCount(),   //SizeT 		edges,
                store.getColumnIndex(), //VertexId 	    *h_column_indices,
                store.getRowPointer(),  //SizeT 		*h_row_offsets,
                num_gpus                //int 		num_gpus,
                );

    //vmask = new Csr::VisitedMask*[csr_problem->num_gpus];
    vmask = new unsigned char*[csr_problem->num_gpus];

    for(int i=0; i < csr_problem->num_gpus; i++){
        typename Csr::GraphSlice* gs = csr_problem->graph_slices[i];

        //vmask[i]= new typename Csr::VisitedMask[gs->frontier_elements[1]];
        vmask[i]= new unsigned char[gs->frontier_elements[1]];
    }

}

CUDA_BFS::~CUDA_BFS(){

    for(int i=0; i < csr_problem->num_gpus; i++)
        delete[] vmask[i];
    delete[] vmask;

    delete bfsGPU;
    delete csr_problem;

    delete[] redbuff;
    delete[] queuebuff;
    delete[] recv_fq_buff;

    delete[] predessor;
}

void CUDA_BFS::reduce_fq_out(vtxtyp *startaddr, long insize)
{
    typename MatrixT::vtxtyp *sta_nxt = startaddr + csr_problem->num_gpus;
    typename MatrixT::vtxtyp *qb_nxt  = queuebuff + csr_problem->num_gpus;
    typename MatrixT::vtxtyp *rb_nxt  = redbuff + csr_problem->num_gpus;
    typename MatrixT::vtxtyp *endp;

    for(int i=0; i < csr_problem->num_gpus; i++){
        endp = std::set_union( qb_nxt,  qb_nxt+queuebuff[i],
                                     sta_nxt, sta_nxt+startaddr[i],
                                     rb_nxt );
        qb_nxt+=queuebuff[i];
        sta_nxt+=startaddr[i];
        redbuff[i]= endp - rb_nxt;
        rb_nxt = endp;

    }

    qb_length =  endp - redbuff;
    std::swap( queuebuff, redbuff);
}

void CUDA_BFS::getOutgoingFQ(vtxtyp *&startaddr, long &outsize)
{
    startaddr = queuebuff;
    outsize   = qb_length;
}

void CUDA_BFS::setModOutgoingFQ(vtxtyp *startaddr, long insize)
{
    #pragma omp parallel for
    for(int i=0; i < csr_problem->num_gpus; i++){
        typename Csr::GraphSlice* gs = csr_problem->graph_slices[i];
        cudaSetDevice(gs->gpu);
        cudaDeviceSynchronize();
    }

    if(startaddr!= 0){
        std::swap(recv_fq_buff, queuebuff);
        qb_length = insize;
    }
    //update visited
    #pragma omp parallel for
    for(int i=0; i < csr_problem->num_gpus; i++){
        typename MatrixT::vtxtyp *qb_nxt  = queuebuff + csr_problem->num_gpus;
        const int64_t end   = queuebuff[i];

        for(int j=0; j < i; j++){
            qb_nxt += queuebuff[j];
        }

        for(int j=0; j < end;j++){
            typename Csr::ProblemType::VertexId vtxID = qb_nxt[j] & Csr::ProblemType::VERTEX_ID_MASK;
            vmask[i][vtxID] = 1;
        }
        typename Csr::GraphSlice* gs = csr_problem->graph_slices[i];
        cudaSetDevice(gs->gpu);

        cudaMemcpyAsync( gs->d_filter_mask,
                         vmask[i],
                         gs->frontier_elements[1]*sizeof(typename Csr::VisitedMask),
                         cudaMemcpyHostToDevice,
                         gs->stream
            );
    }


}
/*
 *  Expect symetric partitioning, so all parameters are ignored.
 */
void CUDA_BFS::getOutgoingFQ(vtxtyp globalstart, long size, vtxtyp *&startaddr, long &outsize)
{
    startaddr = queuebuff;
    outsize   = queue_sizing;
}

/*  Sets the incoming FQ.
 *  Expect symetric partitioning, so all parameters are ignored.
 */
void CUDA_BFS::setIncommingFQ(vtxtyp globalstart, long size, vtxtyp *startaddr, long &insize_max)
{
    if(startaddr == recv_fq_buff)
        std::swap(recv_fq_buff, queuebuff);
    qb_length = insize_max;
}

bool CUDA_BFS::istheresomethingnew()
{
    return newElements;
}

void CUDA_BFS::getBackPredecessor(){
    bfsGPU->testOverflow(*csr_problem);
    csr_problem->ExtractResults(predessor);
    for(uint64_t i=0; i < store.getLocColLength(); i++){
        if(predessor[i]!=-1)
            predessor[i]=store.localtoglobalCol(predessor[i]);
    }
}

void CUDA_BFS::getBackOutqueue()
{
    //get length of next queues
    #pragma omp parallel for
    for(int i=0; i < csr_problem->num_gpus; i++){
        queuebuff[i] = bfsGPU->getQueueSize(i);
    }

    qb_length = 0;
    typename MatrixT::vtxtyp *qb_nxt  = queuebuff + csr_problem->num_gpus;
    // copy next queue to host
    for(int i=0; i < csr_problem->num_gpus; i++){
        typename Csr::GraphSlice* gs = csr_problem->graph_slices[i];
        cudaSetDevice(gs->gpu);

        cudaMemcpyAsync( qb_nxt,
                         gs->frontier_queues.d_keys[0],
                         queuebuff[i],
                         cudaMemcpyDeviceToHost,
                         gs->stream
            );
        qb_nxt += queuebuff[i];
        qb_length += queuebuff[i];
    }
    //
    #pragma omp parallel for
    for(int i=0; i < csr_problem->num_gpus; i++){
        Csr::GraphSlice* gs = csr_problem->graph_slices[i];
        cudaSetDevice(gs->gpu);
        cudaDeviceSynchronize();
    }

    for(int i=0; i < csr_problem->num_gpus; i++){
        typename Csr::GraphSlice* gs = csr_problem->graph_slices[i];
        cudaSetDevice(gs->gpu);

        cudaMemcpyAsync( vmask[i],
                         gs->d_filter_mask,
                         gs->frontier_elements[1]*sizeof(typename Csr::VisitedMask),
                         cudaMemcpyDeviceToHost,
                         gs->stream
            );
    }
}

void CUDA_BFS::setBackInqueue()
{
    typename MatrixT::vtxtyp *qb_nxt  = queuebuff + csr_problem->num_gpus;
    // copy next queue to device
    for(int i=0; i < csr_problem->num_gpus; i++){
        cudaSetDevice(i);
        typename Csr::GraphSlice* gs = csr_problem->graph_slices[i];

        cudaMemcpyAsync( gs->frontier_queues.d_keys[0],
                         qb_nxt,
                         queuebuff[i]*sizeof(vtxtyp),
                         cudaMemcpyHostToDevice,
                         gs->stream
            );
        qb_nxt += queuebuff[i];
    }

    //set length of current queue
    #pragma omp parallel for
    for(int i=0; i < csr_problem->num_gpus; i++){
        bfsGPU->setQueueSize(i,queuebuff[i]);
    }

}

void CUDA_BFS::setStartVertex(vtxtyp start)
{
    if(csr_problem->Reset(
                bfsGPU->GetFrontierType(),
                queue_sizing
                )!=cudaSuccess){
        fprintf(stderr,"Reset error.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if(bfsGPU->EnactSearch(
                *csr_problem,
                start
                )!=cudaSuccess){
        fprintf(stderr,"Start error.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void CUDA_BFS::runLocalBFS()
{
    if(bfsGPU->EnactIteration(
                *csr_problem,
                newElements
                )!=cudaSuccess){
        fprintf(stderr,"Iteration error.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
