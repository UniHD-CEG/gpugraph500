#include "globalbfs.h"
#include <cstdio>
#include <assert.h>

GlobalBFS::GlobalBFS(DistMatrix2d &_store): store(_store)
{
     // Split communicator into row and column communicator
     // Split by row, rank by column
     MPI_Comm_split(MPI_COMM_WORLD, store.getLocalRowID(), store.getLocalColumnID(), &row_comm);
     // Split by column, rank by row
     MPI_Comm_split(MPI_COMM_WORLD, store.getLocalColumnID(), store.getLocalRowID(), &col_comm);

     fold_fq_props = store.getFoldProperties();

     predessor = new vtxtype[store.getLocColLength()];
}

/*
 * BFS search:
 * 0) Node 0 sends start vertex to all nodes
 * 1) Nodes test, if they are responsible for this vertex and push it,
 *    if they are in there fq
 * 2) Local expansion
 * 3) Test if anything is done
 * 4) global expansion
 * 5) global fold
*/
void GlobalBFS::runBFS(vtxtype startVertex)
{
    //reset predessor list
    for(int i = 0; i < store.getLocColLength(); i++){
        predessor[i] = -1;
    }
// 0
    MPI_Bcast(&startVertex,1,MPI_LONG,0,MPI_COMM_WORLD);
// 1
    setStartVertex(startVertex);
// 2
    while(true){
    runLocalBFS();
// 3
    int anynewnodes, anynewnodes_global;
    anynewnodes = istheresomethingnew();
    MPI_Allreduce(&anynewnodes, &anynewnodes_global,1,MPI_INT,MPI_LOR,MPI_COMM_WORLD);
    if(!anynewnodes_global){
        return; //There is nothing too do. Finish iteration.
    }
// 4
    // tree based reduce operation with messages of variable size
    // root 0
    int rounds = 0;
    while((1 << rounds) < store.getNumRowSl() ){
        if((store.getLocalRowID() >> rounds)%2 == 1){
            void* startaddr_fq;
            long _outsize;
            //comute recv addr
            int recv_addr = (store.getLocalRowID() + store.getNumRowSl() - (1 << rounds)) % store.getNumRowSl();
            //get fq to send
            getOutgoingFQ(startaddr_fq, _outsize);
           //send fq
            MPI_Ssend(startaddr_fq,_outsize ,fq_tp_type,recv_addr,rounds,col_comm);
            break;
        } else if ( store.getLocalRowID() + (1 << rounds) < store.getNumRowSl() ){
            MPI_Status    status;
            int count;
            //compute send addr
            int sender_addr = (store.getLocalRowID() +  (1 << rounds)) % store.getNumRowSl();
            //recv fq
            MPI_Recv(recv_fq_buff, recv_fq_buff_length, fq_tp_type,sender_addr,rounds, col_comm, &status);
            MPI_Get_count(&status, fq_tp_type, &count);
             //do reduce
            reduce_fq_out(recv_fq_buff,static_cast<long>(count));
        }
        rounds++;
    }

    //distribute solution
    if(0 == store.getLocalRowID())
    {
        void* startaddr_fq;
        long _outsize;
        //get fq to send
        getOutgoingFQ(startaddr_fq, _outsize);
        MPI_Bcast(&_outsize,1,MPI_LONG,0 ,col_comm);
        MPI_Bcast(startaddr_fq,_outsize,fq_tp_type,0 ,col_comm);
        setModOutgoingFQ(0,_outsize);
    } else {
        long _outsize;
        MPI_Bcast(&_outsize,1,MPI_LONG,0,col_comm);
        assert(_outsize <= recv_fq_buff_length);
        MPI_Bcast(recv_fq_buff,_outsize,fq_tp_type,0,col_comm);
        setModOutgoingFQ(recv_fq_buff,_outsize);
    }
// 5
    for(std::vector<DistMatrix2d::fold_prop>::iterator it = fold_fq_props.begin(); it  != fold_fq_props.end(); it++){
        if(it->sendColSl == store.getLocalColumnID() ){
            void*   startaddr;
            long     outsize;
            getOutgoingFQ(it->startvtx, it->size, startaddr, outsize);
            MPI_Bcast(&outsize,1,MPI_LONG,it->sendColSl,row_comm);
            MPI_Bcast(startaddr,outsize,fq_tp_type,it->sendColSl,row_comm);
            setIncommingFQ(it->startvtx, it->size, startaddr, outsize);
        }else{
            long     outsize;
            MPI_Bcast(&outsize,1,MPI_LONG,it->sendColSl,row_comm);
            assert(outsize <= recv_fq_buff_length);
            MPI_Bcast(recv_fq_buff,outsize,fq_tp_type,it->sendColSl,row_comm);
            setIncommingFQ(it->startvtx, it->size, recv_fq_buff, outsize);
        }
    }
}
}


void GlobalBFS::validate()
{
    //curently manual validation ;)
    for(int i = 0; i < store.getLocColLength(); i++){
        printf("%d:[%ld]\n", store.localtoglobalCol(i),  predessor[i]);
    }

}
