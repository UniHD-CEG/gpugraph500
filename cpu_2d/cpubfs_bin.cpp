#include <assert.h>
#include "cpubfs_bin.h"

CPUBFS_bin::CPUBFS_bin(DistMatrix2d &_store):GlobalBFS(_store)
{
    fq_tp_type = MPI_UINT8_T; //Frontier Queue Transport Type

    //allocate recive buffer
    long recv_fq_buff_length_tmp = std::max(store.getLocRowLength(), store.getLocColLength());
    recv_fq_buff_length = recv_fq_buff_length_tmp/8 + ((recv_fq_buff_length_tmp%8 >0)? 1:0);
    recv_fq_buff = static_cast<void*>( new uint8_t[recv_fq_buff_length]);

    visited = new uint8_t[store.getLocColLength()/8];
    fq_out  = new uint8_t[store.getLocColLength()/8];
    fq_in   = new uint8_t[store.getLocRowLength()/8];
}

CPUBFS_bin::~CPUBFS_bin()
{
    delete[] visited;
    delete[] fq_out;
    delete[] fq_in;
}


void CPUBFS_bin::reduce_fq_out(void *startaddr, long insize)
{
    assert(insize == store.getLocColLength()/8);
    for(long i = 0; i < insize; i++){
        fq_out[i] |= ((uint8_t*) startaddr)[i];
    }
}

void CPUBFS_bin::getOutgoingFQ(void *&startaddr, vtxtype &outsize)
{
   startaddr= fq_out;
   outsize  = store.getLocColLength()/8;

}

void CPUBFS_bin::setModOutgoingFQ(void *startaddr, long insize)
{
   if(startaddr != 0)
       memcpy (fq_out, startaddr, insize );
   for(long i = 0; i < store.getLocColLength()/64; i++){
       reinterpret_cast<uint64_t*>(visited)[i] |=  reinterpret_cast<uint64_t*>(fq_out)[i];
   }
   for(long i = 8*(store.getLocColLength()/64); i < store.getLocColLength()/8; i++){
       visited[i] |=  fq_out[i];
   }
}

void CPUBFS_bin::getOutgoingFQ(vtxtype globalstart, vtxtype size, void *&startaddr, vtxtype &outsize)
{
    startaddr = &fq_out[store.globaltolocalCol(globalstart)/8];
    outsize = size/8;
}

void CPUBFS_bin::setIncommingFQ(vtxtype globalstart, vtxtype size, void *startaddr, vtxtype &insize_max)
{
    memcpy(&fq_in[store.globaltolocalRow(globalstart)/8],  startaddr, size/8 );
}

bool CPUBFS_bin::istheresomethingnew()
{
    for(long i = 0; i < store.getLocColLength()/64; i++){
        if(reinterpret_cast<uint64_t*>(fq_out)[i] > 0){
           return true;
        }
    }
    for(long i = 8*(store.getLocColLength()/64); i < store.getLocColLength()/8; i++){
        if(fq_out[i] > 0){
           return true;
        }
    }
    return false;
}

void CPUBFS_bin::setStartVertex(vtxtype start)
{
    memset ( visited , 0, store.getLocColLength()/8 );
    memset ( fq_out  , 0, store.getLocColLength()/8 );
    memset ( fq_in   , 0, store.getLocRowLength()/8 );

    if(store.isLocalRow(start)){
        vtxtype lstart = store.globaltolocalRow(start);
        fq_in[lstart/8] = 1 << (lstart%8);
    }
     if(store.isLocalColumn(start)){
         vtxtype lstart = store.globaltolocalCol(start);
         visited[lstart/8] = 1 << (lstart%8);
    }
    //reset predessor list
    for(int i = 0; i < store.getLocColLength(); i++){
        predessor[i] = -1;
    }

   if(store.isLocalColumn(start)){
        predessor[store.globaltolocalCol(start)] = start;
    }
}

void CPUBFS_bin::runLocalBFS()
{
    memset ( fq_out  , 0, store.getLocColLength()/8 );
    long rj = 0;
    #pragma omp parallel for
    for(long i = 0; i < store.getLocRowLength()/8 ; i++){
        for(int ii = 0; ii < 8; ii++){
            if((fq_in[i]&1<<ii) > 0){
                for(vtxtype j = store.getRowPointer()[(i<<3)|ii]; j < store.getRowPointer()[(i<<3)+ii+1]; j++){
                    vtxtype visit_vtx     = store.getColumnIndex()[j];
                    vtxtype visit_vtx_loc = store.globaltolocalCol(visit_vtx);
                    if((visited[visit_vtx_loc>>3] & (1 << (visit_vtx_loc &0x7))) == 0 ){
                        if((fq_out[visit_vtx_loc>>3] & (1 << (visit_vtx_loc &0x7))) == 0 ){
                            #pragma omp atomic
                            fq_out[visit_vtx_loc>>3] |= 1 << (visit_vtx_loc &0x7);
                            predessor[visit_vtx_loc] = store.localtoglobalRow(i*8+ii);
                        }
                    }
                }
            }else{
                rj++;
            }
        }
    }
}
