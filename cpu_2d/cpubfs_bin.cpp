#include <assert.h>
#include "cpubfs_bin.h"

CPUBFS_bin::CPUBFS_bin(DistMatrix2d<true,64>& _store):GlobalBFS<true,64>(_store)
{
    fq_tp_type = MPI_UINT64_T; //Frontier Queue Transport Type

    //allocate recive buffer
    long recv_fq_buff_length_tmp = std::max(store.getLocRowLength(), store.getLocColLength());
    recv_fq_buff_length = recv_fq_buff_length_tmp/64 + ((recv_fq_buff_length_tmp%64 >0)? 1:0);
    recv_fq_buff = static_cast<void*>( new uint64_t[recv_fq_buff_length]);

    visited = new uint64_t[store.getLocColLength()/64];
    fq_out  = new uint64_t[store.getLocColLength()/64];
    fq_in   = new uint64_t[store.getLocRowLength()/64];
}

CPUBFS_bin::~CPUBFS_bin()
{
    delete[] visited;
    delete[] fq_out;
    delete[] fq_in;
}


void CPUBFS_bin::reduce_fq_out(void *startaddr, long insize)
{
    assert(insize == store.getLocColLength()/64);
    for(long i = 0; i < store.getLocColLength()/64; i++){
        fq_out[i] |= reinterpret_cast<uint64_t*>(startaddr)[i];
    }
}

void CPUBFS_bin::getOutgoingFQ(void *&startaddr, vtxtype &outsize)
{
   startaddr= fq_out;
   outsize  = store.getLocColLength()/64;

}

void CPUBFS_bin::setModOutgoingFQ(void *startaddr, long insize)
{
   assert(insize==store.getLocColLength()/64);
   if(startaddr != 0)
       memcpy (fq_out, startaddr, store.getLocColLength()/8 );
   for(long i = 0; i < store.getLocColLength()/64; i++){
       visited[i] |=  fq_out[i];
   }
}

void CPUBFS_bin::getOutgoingFQ(vtxtype globalstart, vtxtype size, void *&startaddr, vtxtype &outsize)
{
    startaddr = &fq_out[store.globaltolocalCol(globalstart)/64];
    outsize = size/64;
}

void CPUBFS_bin::setIncommingFQ(vtxtype globalstart, vtxtype size, void *startaddr, vtxtype &insize_max)
{
    assert(insize_max >= size/64);
    memcpy(&fq_in[store.globaltolocalRow(globalstart)/64],  startaddr, size/8 );
}

bool CPUBFS_bin::istheresomethingnew()
{
    for(long i = 0; i < store.getLocColLength()/64; i++){
        if(fq_out[i] > 0){
           return true;
        }
    }
    return false;
}

void CPUBFS_bin::setStartVertex(vtxtype start)
{
    memset ( visited , 0, store.getLocColLength()/8 );
    memset ( fq_in   , 0, store.getLocRowLength()/8 );

    if(store.isLocalRow(start)){
        vtxtype lstart = store.globaltolocalRow(start);
        fq_in[lstart/64] = 1ul << (lstart& 0x3F);
    }
     if(store.isLocalColumn(start)){
         vtxtype lstart = store.globaltolocalCol(start);
         visited[lstart/64] = 1ul << (lstart&0x3F);
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
    memset( fq_out, 0, store.getLocColLength()/8 );
    #pragma omp parallel for
    for(int64_t i = 0; i < store.getLocRowLength()/64 ; i++){
        for(int ii = 0; ii < 64; ii++){
            if((fq_in[i]&1ul<<ii) > 0){
                const vtxtype endrp = store.getRowPointer()[i*64+ii+1];
                for(vtxtype j = store.getRowPointer()[i*64+ii]; j < endrp; j++){
                    vtxtype visit_vtx_loc = store.getColumnIndex()[j];
                    if((visited[visit_vtx_loc>>6] & (1ul << (visit_vtx_loc &0x3F))) == 0 ){
                        if((fq_out[visit_vtx_loc>>6] & (1ul << (visit_vtx_loc &0x3F))) == 0 ){
                            #pragma omp atomic
                            fq_out[visit_vtx_loc>>6] |= 1ul << (visit_vtx_loc & 0x3F);
                            predessor[visit_vtx_loc] = store.localtoglobalRow(i*64+ii);
                        }
                    }
                }
            }
        }
    }
}
