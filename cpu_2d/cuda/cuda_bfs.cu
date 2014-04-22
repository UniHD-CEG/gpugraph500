#include "cuda_bfs.h"

#include "b40c/util/error_utils.cuh"

#include <cstdlib>
#include <algorithm>
#include <functional>


CUDA_BFS::CUDA_BFS(MatrixT &_store,int num_gpus,double _queue_sizing):
    GlobalBFS< CUDA_BFS,vtxtyp,MatrixT>(_store),
    queue_sizing(_queue_sizing),
    vmask(0)
{

    b40c::util::B40CPerror(cudaSetDeviceFlags(cudaDeviceMapHost),
                           "Enabling of the allocation of pinned host memory faild",__FILE__, __LINE__);

    if(num_gpus==0){
        b40c::util::B40CPerror(cudaGetDeviceCount(&num_gpus),
                    "Can't get number of devices!",__FILE__, __LINE__);
    }

    //expect symetrie
    if(store.getNumRowSl() != store.getNumColumnSl()){
        fprintf(stderr,"Currently the partitioning has to be symetric.");
        exit(1);
    }
    predecessor    = new vtxtyp[store.getLocColLength()];

    fq_tp_type = MPI_INT64_T;
    recv_fq_buff_length = static_cast<vtxtyp>
            (std::max(store.getLocRowLength(), store.getLocColLength())*_queue_sizing)+num_gpus;
    //recv_fq_buff = new vtxtyp[recv_fq_buff_length];
    cudaHostAlloc(&recv_fq_buff, recv_fq_buff_length*sizeof(vtxtyp) , NULL );
    //multipurpose buffer
    qb_length    = 0;
    cudaHostAlloc(&queuebuff, recv_fq_buff_length*sizeof(vtxtyp) , NULL );
    rb_length    = 0;
    cudaHostAlloc(&redbuff , recv_fq_buff_length*sizeof(vtxtyp) , NULL );

   csr_problem = new Csr;
#ifdef INSTRUMENTED
   bfsGPU = new EnactorMultiGpu<true>;
#else
   bfsGPU = new EnactorMultiGpu<false>;
#endif

     b40c::util::B40CPerror(csr_problem->FromHostProblem(
                false,                  //bool          stream_from_host,
                store.getLocRowLength(),//SizeT 		nodes,
                store.getEdgeCount(),   //SizeT 		edges,
                store.getColumnIndex(), //VertexId 	    *h_column_indices,
                store.getRowPointer(),  //SizeT 		*h_row_offsets,
                num_gpus,               //int 		num_gpus,
                0                       //verbosity
         ), "FromHostProblem failed!" ,__FILE__, __LINE__);

     //Test if peer comunication is possible
     bool peerPossible = true;
     for (int gpu = 0; gpu < num_gpus; gpu++) {
         for (int other_gpu = (gpu + 1) % num_gpus;
              other_gpu != gpu;
              other_gpu = (other_gpu + 1) % num_gpus)
         {
             int canAccessPeer;
             // Set device
             b40c::util::B40CPerror(cudaDeviceCanAccessPeer(&canAccessPeer,gpu,other_gpu)
                                    ,"Can not activate peer access!" ,__FILE__, __LINE__);
             if( !canAccessPeer  ){
                peerPossible = false;
                break;
             }
         }
     }
     // Enable symmetric peer access between gpus
     // from test_bfs.cu
     if(peerPossible)
     for (int gpu = 0; gpu < num_gpus; gpu++) {
         for (int other_gpu = (gpu + 1) % num_gpus;
              other_gpu != gpu;
              other_gpu = (other_gpu + 1) % num_gpus)
         {
                 // Set device
                 if (b40c::util::B40CPerror(cudaSetDevice(gpu),
                     "MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

                 //printf("Enabling peer access to GPU %d from GPU %d\n", other_gpu, gpu);

                 cudaError_t error = cudaDeviceEnablePeerAccess(other_gpu, 0);
                 if ((error != cudaSuccess) && (error != cudaErrorPeerAccessAlreadyEnabled)) {
                     b40c::util::B40CPerror(error,"MultiGpuBfsEnactor cudaDeviceEnablePeerAccess failed", __FILE__, __LINE__);
                     int canAccessPeer;
                     b40c::util::B40CPerror(cudaDeviceCanAccessPeer(&canAccessPeer,gpu,other_gpu)
                                            ,"Can not access device!" ,__FILE__, __LINE__);
                     if(canAccessPeer)
                         fprintf(stderr, "Can access peer %d from %d!\n",other_gpu,gpu);
                     else
                         fprintf(stderr, "Can't access peer %d from %d!\n",other_gpu,gpu);
                 }
         }
     }

}

CUDA_BFS::~CUDA_BFS(){

    if(vmask!=0) cudaFree(vmask);

    delete bfsGPU;
    delete csr_problem;

    cudaFree(redbuff);
    cudaFree(queuebuff);
    cudaFree(recv_fq_buff);

    delete[] predecessor;
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
        Csr::GraphSlice* gs = csr_problem->graph_slices[i];
        b40c::util::B40CPerror(cudaStreamSynchronize(gs->stream),
                    "Can't synchronize Stream." , __FILE__, __LINE__);
    }

    if(startaddr!= 0){
        std::swap(recv_fq_buff, queuebuff);
        qb_length = insize;
    }
    //update visited
    #pragma omp parallel for
    for(int i=csr_problem->num_gpus; i < qb_length; i++){
        typename Csr::ProblemType::VertexId vtxID = queuebuff[i] & Csr::ProblemType::VERTEX_ID_MASK;
        #pragma omp atomic
        vmask[vtxID>>3] |= 1<< (vtxID&0x7 );
    }
    for(int i=0; i < csr_problem->num_gpus; i++){
        typename Csr::GraphSlice* gs = csr_problem->graph_slices[i];
        int visited_mask_bytes 	 = ((gs->nodes * sizeof(typename Csr::VisitedMask)) + 8 - 1) / 8;
        b40c::util::B40CPerror(cudaMemcpyAsync( gs->d_visited_mask,
                     vmask,
                     visited_mask_bytes,
                     cudaMemcpyHostToDevice,
                     gs->stream
               ), "Copy of d_filer_mask from device failed" , __FILE__, __LINE__);
    }
}
/*
 *  Expect symetric partitioning, so all parameters are ignored.
 */
void CUDA_BFS::getOutgoingFQ(vtxtyp globalstart, long size, vtxtyp *&startaddr, long &outsize)
{
    startaddr = queuebuff;
    outsize   = qb_length;
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
    return !done;
}

void CUDA_BFS::getBackPredecessor(){
    //terminate all operations
    #pragma omp parallel for
    for(int i=0; i < csr_problem->num_gpus; i++){
        Csr::GraphSlice* gs = csr_problem->graph_slices[i];
        b40c::util::B40CPerror(cudaStreamSynchronize(gs->stream),
                    "Can't synchronize device." , __FILE__, __LINE__);
    }
    bfsGPU->testOverflow(*csr_problem);
    b40c::util::B40CPerror(csr_problem->ExtractResults(predecessor),
                            "Extraction of result failed" , __FILE__, __LINE__);
    for(uint64_t i=0; i < store.getLocColLength(); i++){
        if(predecessor[i]>-1 )
            predecessor[i]=store.localtoglobalRow(predecessor[i]);
        if(predecessor[i]==-2){
            predecessor[i]=store.localtoglobalCol(i);
        }
    }
}

void CUDA_BFS::getBackOutqueue()
{
    //get length of next queues
    #pragma omp parallel for
    for(int i=0; i < csr_problem->num_gpus; i++){
        queuebuff[i] = bfsGPU->getQueueSize<typename Csr::SizeT>(csr_problem->graph_slices[i]->gpu);
    }

    qb_length = csr_problem->num_gpus;
    typename MatrixT::vtxtyp *qb_nxt  = queuebuff + csr_problem->num_gpus;
    // copy next queue to host
    for(int i=0; i < csr_problem->num_gpus; i++){
        typename Csr::GraphSlice* gs = csr_problem->graph_slices[i];

        b40c::util::B40CPerror(cudaMemcpyAsync( qb_nxt,
                         gs->frontier_queues.d_keys[0],
                         queuebuff[i]*sizeof(typename Csr::VertexId),
                         cudaMemcpyDeviceToHost,
                         gs->stream
            ), "Copy of d_keys[0] failed" , __FILE__, __LINE__);
        qb_nxt += queuebuff[i];
        qb_length += queuebuff[i];
    }
    //
    #pragma omp parallel for
    for(int i=0; i < csr_problem->num_gpus; i++){
        Csr::GraphSlice* gs = csr_problem->graph_slices[i];
        b40c::util::B40CPerror(cudaStreamSynchronize(gs->stream),
                    "Can't synchronize Stream." , __FILE__, __LINE__);
    }

    // Queue preprocessing
    // Sorting
    #pragma omp parallel for
    for(int i=0; i < csr_problem->num_gpus; i++){
        typename MatrixT::vtxtyp *qb_nxt  = queuebuff + csr_problem->num_gpus;
        const int64_t end   = queuebuff[i];

        for(int j=0; j < i; j++){
            qb_nxt += queuebuff[j];
        }

        std::sort(qb_nxt, qb_nxt+end);
    }
    //Uniqueness
    typename MatrixT::vtxtyp *qb_nxt_in  = queuebuff + csr_problem->num_gpus;
    typename MatrixT::vtxtyp *qb_nxt_out = redbuff   + csr_problem->num_gpus;
    for(int i=0; i < csr_problem->num_gpus; i++){
        typename MatrixT::vtxtyp* start_in = std::upper_bound(qb_nxt_in, qb_nxt_in+queuebuff[i], -1);
        typename MatrixT::vtxtyp* end_out = std::unique_copy(start_in, qb_nxt_in+queuebuff[i], qb_nxt_out);
        qb_nxt_in+=queuebuff[i];
        redbuff[i] = end_out - qb_nxt_out;
        qb_nxt_out = end_out;
    }
    qb_length = qb_nxt_out - redbuff ;
    std::swap(queuebuff, redbuff);
}

void CUDA_BFS::setBackInqueue()
{
    typename MatrixT::vtxtyp *qb_nxt  = queuebuff + csr_problem->num_gpus;
    // copy next queue to device
    for(int i=0; i < csr_problem->num_gpus; i++){
        typename Csr::GraphSlice* gs = csr_problem->graph_slices[i];

        b40c::util::B40CPerror(cudaMemcpyAsync( gs->frontier_queues.d_keys[0],
                         qb_nxt,
                         queuebuff[i]*sizeof(typename Csr::VertexId),
                         cudaMemcpyHostToDevice,
                         gs->stream
            ), "Copy of d_keys[0] from device failed" , __FILE__, __LINE__);
        qb_nxt += queuebuff[i];
    }

    //set length of current queue
    #pragma omp parallel for
    for(int i=0; i < csr_problem->num_gpus; i++){
        bfsGPU->setQueueSize<typename Csr::SizeT>(i,static_cast<typename Csr::SizeT>(queuebuff[i]));
    }

}

void CUDA_BFS::setStartVertex(vtxtyp start)
{
    done = false;
    vtxtyp lstart = -1;

    if(b40c::util::B40CPerror(csr_problem->Reset(
                bfsGPU->GetFrontierType(),
                queue_sizing
       ), "Reset error.", __FILE__, __LINE__)!=cudaSuccess){
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Alloc and reset visited mask on host
    typename Csr::GraphSlice* gs = csr_problem->graph_slices[0];
    int visited_mask_bytes 	 = ((gs->nodes * sizeof(typename Csr::VisitedMask)) + 8 - 1) / 8;
    if(vmask == 0)
        cudaHostAlloc(&vmask, visited_mask_bytes, NULL);
    std::fill_n(vmask, visited_mask_bytes, 0);

    if(store.isLocalColumn(start)){
        lstart = store.globaltolocalCol(start);
        vmask[lstart >> 3] = 1 << (lstart & 0x7);
    }

    //new next queue
    std::fill_n(queuebuff, csr_problem->num_gpus, 0);
    qb_length =  csr_problem->num_gpus;

    if(store.isLocalRow(start)){
        vtxtyp rstart = store.globaltolocalRow(start);
        vtxtyp src_owner = csr_problem->GpuIndex(rstart);
        rstart |= (src_owner <<  Csr::ProblemType::GPU_MASK_SHIFT);

        queuebuff[src_owner] = 1;
        queuebuff[csr_problem->num_gpus] = rstart;
        qb_length = csr_problem->num_gpus+1;
    }

    if(b40c::util::B40CPerror(bfsGPU->EnactSearch(
                *csr_problem,
                lstart
      ), "Start error.", __FILE__, __LINE__)!=cudaSuccess){
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for(int i=0; i < csr_problem->num_gpus; i++){
        //set new visited map
        typename Csr::GraphSlice* gs = csr_problem->graph_slices[i];
        b40c::util::B40CPerror(cudaMemcpyAsync( gs->d_visited_mask,
                     vmask,
                     visited_mask_bytes,
                     cudaMemcpyHostToDevice,
                     gs->stream
               ), "Copy of d_filer_mask from device failed" , __FILE__, __LINE__);
        // set new current queue
        if(queuebuff[i]){
            b40c::util::B40CPerror(cudaMemcpyAsync( gs->frontier_queues.d_keys[0],
                             &queuebuff[csr_problem->num_gpus],
                             sizeof(typename Csr::VertexId),
                             cudaMemcpyHostToDevice,
                             gs->stream
                ), "Copy of d_keys[0] from device failed" , __FILE__, __LINE__);
        }
        bfsGPU->setQueueSize<typename Csr::SizeT>(i,static_cast<typename Csr::SizeT>(queuebuff[i]));
    }

}

void CUDA_BFS::runLocalBFS()
{
    if(b40c::util::B40CPerror(bfsGPU->EnactIteration(
                *csr_problem,
                done
      ), "Iteration error.", __FILE__, __LINE__)!=cudaSuccess){
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
