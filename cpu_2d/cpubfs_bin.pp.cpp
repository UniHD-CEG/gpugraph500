#include <cstdlib>
#include <algorithm>
#include <assert.h>
#include "comp_opt.h"
#include "cpubfs_bin.h"

CPUBFS_bin::CPUBFS_bin(MatrixT& _store, int64_t verbosity):GlobalBFS<CPUBFS_bin,uint64_t,uint64_t,MatrixT>(_store),col64(_store.getLocColLength()/64),row64(_store.getLocRowLength()/64)
{
    fq_tp_type = ((MPI_Datatype) ((void *) &( ompi_mpi_uint64_t))); //Frontier Queue Transport Type
    //if(posix_memalign((void**)&predecessor,64,sizeof(vtxtyp)*store.getLocColLength()))predecessor=0;// new vtxtyp[store.getLocColLength()];;
    MPI_Alloc_mem(sizeof(vtxtyp)*store.getLocColLength(), ((MPI_Info) ((void *) &( ompi_mpi_info_null))), (void*)&predecessor);
    //allocate recive buffer
    long recv_fq_buff_length_tmp = std::max(store.getLocRowLength(), store.getLocColLength());
    recv_fq_buff_length = recv_fq_buff_length_tmp/64 + ((recv_fq_buff_length_tmp%64 >0)? 1:0);
    //if(posix_memalign((void**)&recv_fq_buff,64,sizeof(uint64_t)*recv_fq_buff_length))recv_fq_buff=0;
    MPI_Alloc_mem(sizeof(uint64_t)*recv_fq_buff_length, ((MPI_Info) ((void *) &( ompi_mpi_info_null))), (void*)&recv_fq_buff);
    if(posix_memalign((void**)&visited,64,sizeof(uint64_t)*col64))visited=0;//new uint64_t[col64];
    //if(posix_memalign((void**)&fq_out,64,sizeof(uint64_t)*col64))fq_out=0; //new uint64_t[col64];
    MPI_Alloc_mem(sizeof(uint64_t)*col64, ((MPI_Info) ((void *) &( ompi_mpi_info_null))), (uint64_t*)&fq_out);
    if(posix_memalign((void**)&fq_in,64,sizeof(uint64_t)*row64)) fq_in =0; //new uint64_t[row64];

    if(predecessor  == 0 ||
       recv_fq_buff == 0 ||
       visited      == 0 ||
       fq_out       == 0 ||
       fq_in        == 0) {
      if(verbosity != 0) 
         fprintf(stderr,"Unable to allocate memory.\n");
       MPI_Abort((( MPI_Comm) ((void *) &( ompi_mpi_comm_world))), -1);

    }
}


CPUBFS_bin::~CPUBFS_bin()
{
    //free(predecessor);
    MPI_Free_mem(predecessor);
    predecessor=0;
    //free(recv_fq_buff);
    MPI_Free_mem(recv_fq_buff);
    recv_fq_buff = 0;
    free(visited);
    visited = 0;
    //free(fq_out);
    MPI_Free_mem(fq_out);
    fq_out = 0;
    free(fq_in);
    fq_in = 0;
}


void CPUBFS_bin::reduce_fq_out(uint64_t*  startaddr, long insize)
{
    //;
    //;

    ((insize == col64) ? static_cast<void> (0) : __assert_fail (#insize == col64, "cpubfs_bin.cpp.tau.tmp", 58, __PRETTY_FUNCTION__));    
    for(long i = 0; i < col64; i++){
        fq_out[i] |= startaddr[i];
    }
}

void CPUBFS_bin::getOutgoingFQ(uint64_t *&startaddr, long &outsize)
{
   startaddr = fq_out;
   outsize  = col64;

}

void CPUBFS_bin::setModOutgoingFQ(uint64_t*  startaddr, long insize)
{
   //;
   //;
   ;

   ((insize==col64) ? static_cast<void> (0) : __assert_fail (#insize==col64, "cpubfs_bin.cpp.tau.tmp", 77, __PRETTY_FUNCTION__));
   if(startaddr != 0)
       std::copy( startaddr, startaddr+col64, fq_out);
   for(long i = 0; i < col64; i++){
       visited[i] |=  fq_out[i];
   }
}

void CPUBFS_bin::getOutgoingFQ(vtxtyp globalstart, long size, uint64_t *&startaddr, long &outsize)
{
    startaddr = &fq_out[store.globaltolocalCol(globalstart)/64];
    outsize = size/64;
}

void CPUBFS_bin::setIncommingFQ(vtxtyp globalstart, long size, uint64_t*  startaddr, long &insize_max)
{
    //;
    ((insize_max >= size/64) ? static_cast<void> (0) : __assert_fail (#insize_max >= size/64, "cpubfs_bin.cpp.tau.tmp", 94, __PRETTY_FUNCTION__));
    std::copy(startaddr, startaddr+size/64, &fq_in[store.globaltolocalRow(globalstart)/64]);
}

bool CPUBFS_bin::istheresomethingnew()
{
    //;

    for(long i = 0; i < col64; i++){
        if(fq_out[i] > 0){
           return true;
        }
    }
    return false;
}

void CPUBFS_bin::setStartVertex(const vtxtyp start)
{
    ;
    ;
    std::fill_n( fq_in,   row64, 0);
    std::fill_n( visited, col64, 0);

    if(store.isLocalRow(start)){
        vtxtyp lstart = store.globaltolocalRow(start);
        fq_in[lstart/64] = 1ul << (lstart& 0x3F);
    }
     if(store.isLocalColumn(start)){
         vtxtyp lstart = store.globaltolocalCol(start);
         visited[lstart/64] = 1ul << (lstart&0x3F);
    }
    //reset predecessor list
   //;
   std::fill_n( predecessor,   store.getLocColLength(), -1);

   if(store.isLocalColumn(start)){
        predecessor[store.globaltolocalCol(start)] = start;
    }
}

void CPUBFS_bin::runLocalBFS()
{
    //;
    std::fill_n(fq_out, col64, 0);
    #pragma omp parallel for
    for(int64_t i = 0; i < row64 ; i++){
        if(fq_in[i] > 0){
        for(int ii = 0; ii < 64; ii++){
            if((fq_in[i]&1ul<<ii) > 0){
                const vtxtyp endrp = store.getRowPointer()[i*64+ii+1];
                for(vtxtyp j = store.getRowPointer()[i*64+ii]; j < endrp; j++){
                    vtxtyp visit_vtx_loc = store.getColumnIndex()[j];
                    if((visited[visit_vtx_loc>>6] & (1ul << (visit_vtx_loc &0x3F))) == 0 ){
                        if((fq_out[visit_vtx_loc>>6] & (1ul << (visit_vtx_loc &0x3F))) == 0 ){
                            uint64_t& fqn_ext = fq_out[visit_vtx_loc>>6];
                            uint64_t  setnext = 1ul << (visit_vtx_loc & 0x3F);
                            #pragma omp atomic
                            fqn_ext |= setnext;
                            predecessor[visit_vtx_loc] = store.localtoglobalRow(i*64+ii);
                        }
                    }
                }
            }
        }
        }
    }
}
