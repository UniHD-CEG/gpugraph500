#include "distmatrix2d.hh"
#include <vector>
#include <cstdio>
#include <assert.h>

#ifndef GLOBALBFS_HH
#define GLOBALBFS_HH

/*
 * This classs implements a distributed level synchronus BFS on global scale.
 */
template<bool WOLO=false,int ALG=1>
class GlobalBFS
{
    MPI_Comm row_comm, col_comm;

    // sending node column slice, startvtx, size
    std::vector<typename DistMatrix2d<WOLO,ALG>::fold_prop> fold_fq_props;

protected:
    const DistMatrix2d<WOLO,ALG>& store;
    vtxtype* predessor;

    MPI_Datatype fq_tp_type; //Frontier Queue Transport Type
    void*   recv_fq_buff;
    long    recv_fq_buff_length;
    virtual void reduce_fq_out(void* startaddr, long insize)=0;    //Global Reducer of the local outgoing frontier queues.  Have to be implemented by the children.
    virtual void getOutgoingFQ(void* &startaddr, vtxtype& outsize)=0;
    virtual void setModOutgoingFQ(void* startaddr, long insize)=0; //startaddr: 0, self modification
    virtual void getOutgoingFQ(vtxtype globalstart, vtxtype size, void* &startaddr, vtxtype& outsize)=0;
    virtual void setIncommingFQ(vtxtype globalstart, vtxtype size, void* startaddr, vtxtype& insize_max)=0;
    virtual bool istheresomethingnew()=0;           //to detect if finished
    virtual void setStartVertex(vtxtype start)=0;
    virtual void runLocalBFS()=0;


public:
    GlobalBFS(DistMatrix2d<WOLO,ALG>& _store);

    #ifdef INSTRUMENTED
    void runBFS(vtxtype startVertex, double& lexp, double &lqueue);
    #else
    void runBFS(vtxtype startVertex);
    #endif


    vtxtype* getPredessor();
};

template<bool WOLO,int ALG>
GlobalBFS<WOLO,ALG>::GlobalBFS(DistMatrix2d<WOLO,ALG>&_store): store(_store)
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
#ifdef INSTRUMENTED
template<bool WOLO,int ALG>
void GlobalBFS<WOLO,ALG>::runBFS(vtxtype startVertex, double& lexp, double& lqueue)
#else
template<bool WOLO,int ALG>
void GlobalBFS<WOLO,ALG>::runBFS(vtxtype startVertex)
#endif
{
    #ifdef INSTRUMENTED
    double tstart, tend;
    lexp =0;
    lqueue =0;
    #endif

// 0
    MPI_Bcast(&startVertex,1,MPI_LONG,0,MPI_COMM_WORLD);
// 1
    #ifdef INSTRUMENTED
    tstart = MPI_Wtime();
    #endif
    setStartVertex(startVertex);
    #ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue +=tend-tstart;
    #endif
// 2
    while(true){
    #ifdef INSTRUMENTED
    tstart = MPI_Wtime();
    #endif
    runLocalBFS();
    #ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lexp +=tend-tstart;
    #endif
// 3
    int anynewnodes, anynewnodes_global;
    #ifdef INSTRUMENTED
    tstart = MPI_Wtime();
    #endif
    anynewnodes = istheresomethingnew();
    #ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue +=tend-tstart;
    #endif

    MPI_Allreduce(&anynewnodes, &anynewnodes_global,1,MPI_INT,MPI_LOR,MPI_COMM_WORLD);
    if(!anynewnodes_global){
        MPI_Allreduce(MPI_IN_PLACE, predessor ,store.getLocColLength(),MPI_LONG,MPI_MAX,col_comm);
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
            #ifdef INSTRUMENTED
            tstart = MPI_Wtime();
            #endif
            getOutgoingFQ(startaddr_fq, _outsize);
            #ifdef INSTRUMENTED
            tend = MPI_Wtime();
            lqueue +=tend-tstart;
            #endif
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
            #ifdef INSTRUMENTED
            tstart = MPI_Wtime();
            #endif
            reduce_fq_out(recv_fq_buff,static_cast<long>(count));
            #ifdef INSTRUMENTED
            tend = MPI_Wtime();
            lqueue +=tend-tstart;
            #endif
        }
        rounds++;
    }

    //distribute solution
    if(0 == store.getLocalRowID())
    {
        void* startaddr_fq;
        long _outsize;
        //get fq to send
        #ifdef INSTRUMENTED
        tstart = MPI_Wtime();
        #endif
        getOutgoingFQ(startaddr_fq, _outsize);
        #ifdef INSTRUMENTED
        tend = MPI_Wtime();
        lqueue +=tend-tstart;
        #endif
        MPI_Bcast(&_outsize,1,MPI_LONG,0 ,col_comm);
        MPI_Bcast(startaddr_fq,_outsize,fq_tp_type,0 ,col_comm);
        #ifdef INSTRUMENTED
        tstart = MPI_Wtime();
        #endif
        setModOutgoingFQ(0,_outsize);
        #ifdef INSTRUMENTED
        tend = MPI_Wtime();
        lqueue +=tend-tstart;
        #endif
    } else {
        long _outsize;
        MPI_Bcast(&_outsize,1,MPI_LONG,0,col_comm);
        assert(_outsize <= recv_fq_buff_length);
        MPI_Bcast(recv_fq_buff,_outsize,fq_tp_type,0,col_comm);
        #ifdef INSTRUMENTED
        tstart = MPI_Wtime();
        #endif
        setModOutgoingFQ(recv_fq_buff,_outsize);
        #ifdef INSTRUMENTED
        tend = MPI_Wtime();
        lqueue +=tend-tstart;
        #endif
    }
// 5
    for(typename std::vector<typename DistMatrix2d<WOLO,ALG>::fold_prop>::iterator it = fold_fq_props.begin(); it  != fold_fq_props.end(); it++){
        if(it->sendColSl == store.getLocalColumnID() ){
            void*   startaddr;
            long     outsize;
            #ifdef INSTRUMENTED
            tstart = MPI_Wtime();
            #endif
            getOutgoingFQ(it->startvtx, it->size, startaddr, outsize);
            #ifdef INSTRUMENTED
            tend = MPI_Wtime();
            lqueue +=tend-tstart;
            #endif
            MPI_Bcast(&outsize,1,MPI_LONG,it->sendColSl,row_comm);
            MPI_Bcast(startaddr,outsize,fq_tp_type,it->sendColSl,row_comm);
            #ifdef INSTRUMENTED
            tstart = MPI_Wtime();
            #endif
            setIncommingFQ(it->startvtx, it->size, startaddr, outsize);
            #ifdef INSTRUMENTED
            tend = MPI_Wtime();
            lqueue +=tend-tstart;
            #endif
        }else{
            long     outsize;
            MPI_Bcast(&outsize,1,MPI_LONG,it->sendColSl,row_comm);
            assert(outsize <= recv_fq_buff_length);
            MPI_Bcast(recv_fq_buff,outsize,fq_tp_type,it->sendColSl,row_comm);
            #ifdef INSTRUMENTED
            tstart = MPI_Wtime();
            #endif
            setIncommingFQ(it->startvtx, it->size, recv_fq_buff, outsize);
            #ifdef INSTRUMENTED
            tend = MPI_Wtime();
            lqueue +=tend-tstart;
            #endif
        }
    }
}
}

template<bool WOLO,int ALG>
vtxtype* GlobalBFS<WOLO,ALG>::getPredessor()
{
    return  predessor;
}

#endif // GLOBALBFS_HH
