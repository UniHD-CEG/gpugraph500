#include "distmatrix2d.hh"
#include "comp_opt.h"
#include <vector>
#include <cstdio>
#include <assert.h>

#ifndef GLOBALBFS_HH
#define GLOBALBFS_HH

/*
 * This classs implements a distributed level synchronus BFS on global scale.
 */
template<class Derived,
         class FQ_T,  // Queue Type
         class MType, // Bitmap mask
         class STORE > //Storage of Matrix
class GlobalBFS
{
    MPI_Comm row_comm, col_comm;

    // sending node column slice, startvtx, size
    std::vector<typename STORE::fold_prop> fold_fq_props;

    void allReduceBitCompressed(typename STORE::vtxtyp *&owen, typename STORE::vtxtyp *&tmp,
                                MType *&owenmap, MType *&tmpmap);

protected:
    const STORE& store;
    typename STORE::vtxtyp* predecessor;

    MPI_Datatype fq_tp_type; //Frontier Queue Transport Type
    MPI_Datatype bm_type;    // Bitmap Type
    //FQ_T*  __restrict__ recv_fq_buff; - conflicts with void* ref
    FQ_T*  recv_fq_buff;

    long    recv_fq_buff_length;

    MType* owenmask;
    MType* tmpmask;
    int64_t mask_size;
    // Functions that have to be implemented by the children
    //void reduce_fq_out(FQ_T* startaddr, long insize)=0;    //Global Reducer of the local outgoing frontier queues.  Have to be implemented by the children.
    //void getOutgoingFQ(FQ_T* &startaddr, vtxtype& outsize)=0;
    //void setModOutgoingFQ(FQ_T* startaddr, long insize)=0; //startaddr: 0, self modification
    //void getOutgoingFQ(vtxtype globalstart, vtxtype size, FQ_T* &startaddr, vtxtype& outsize)=0;
    //void setIncommingFQ(vtxtype globalstart, vtxtype size, FQ_T* startaddr, vtxtype& insize_max)=0;
    //bool istheresomethingnew()=0;           //to detect if finished
    //void setStartVertex(const vtxtype start)=0;
    //void runLocalBFS()=0;
    //For accelerators with owen memory
    void getBackPredecessor(); // expected to be used afet the application finished
    void getBackOutqueue();
    void setBackInqueue();

    void generatOwenMask();

public:
    GlobalBFS(STORE& _store);
    ~GlobalBFS();

    #ifdef INSTRUMENTED
    void runBFS(typename STORE::vtxtyp startVertex, double& lexp, double &lqueue, double& rowcom, double& colcom, double& predlistred);
    #else
    void runBFS(typename STORE::vtxtyp startVertex);
    #endif


    typename STORE::vtxtyp* getPredecessor();
};

/*
 * Bitmap compression on predecessor reduction
 *
 */
template<class Derived,class FQ_T,class MType,class STORE>
void GlobalBFS<Derived,FQ_T,MType,STORE>::allReduceBitCompressed(typename STORE::vtxtyp *&owen, typename STORE::vtxtyp *&tmp, MType *&owenmap, MType *&tmpmap)
{

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status    status;
    const int outsizebm = mask_size;
    // root 0
    int rounds = 0;
    while((1 << rounds) < store.getNumRowSl() ){
        //comute recv addr
        int recv_addr = (store.getLocalRowID() + store.getNumRowSl() - (1 << rounds)) % store.getNumRowSl();
        //compute send addr
        int sender_addr = (store.getLocalRowID() +  (1 << rounds)) % store.getNumRowSl();

        if((store.getLocalRowID() >> rounds)%2 == 1){
            MPI_Sendrecv(owenmap, outsizebm, bm_type, recv_addr, rounds<<1,
                         tmpmap,  outsizebm, bm_type, recv_addr, rounds<<1,
                         col_comm, &status );
            for(int i = 0; i < outsizebm; i++){
                tmpmap[i] = ~tmpmap[i] & owenmap[i];
            }
            int p= 0;
            for(int i = 0; i < outsizebm; i++){
                MType tmpm = tmpmap[i];
                while( tmpm != 0){
                     int last = __builtin_ctz(tmpm);
                     tmp[p] = owen[i*8*sizeof(MType)+last];
                     p++;
                     tmpm ^= (1 << last);
                }
            }
            //send fq
            MPI_Ssend(tmp, p ,fq_tp_type,recv_addr,(rounds<<1)+1,col_comm);
        } else if ( store.getLocalRowID() + (1 << rounds) < store.getNumRowSl() ){
            MPI_Sendrecv(owenmap, outsizebm, bm_type, sender_addr, rounds<<1,
                         tmpmap,  outsizebm, bm_type, sender_addr, rounds<<1,
                         col_comm, &status );
            for(int i = 0; i < outsizebm; i++){
                tmpmap[i]  = tmpmap[i]  & ~owenmap[i];
                owenmap[i] = owenmap[i] | tmpmap[i];
            }
            //recv fq
            MPI_Recv(tmp, store.getLocColLength() ,fq_tp_type,sender_addr,(rounds<<1)+1,col_comm,&status);
            int p= 0;
            for(int i = 0; i < outsizebm; i++){
                MType tmpm = tmpmap[i];
                while( tmpm != 0){
                     int last = __builtin_ctz(tmpm);
                     owen[i*8*sizeof(MType)+last]=tmp[p];
                     p++;
                     tmpm ^= (1 << last);
                }
            }

        }
        rounds++;
    }

    //distribute solution
    MPI_Bcast(owen,store.getLocColLength(),fq_tp_type,0,col_comm);
}

template<class Derived,class FQ_T,class MType,class STORE>
void GlobalBFS<Derived,FQ_T,MType,STORE>::getBackPredecessor(){}

template<class Derived,class FQ_T,class MType,class STORE>
void GlobalBFS<Derived,FQ_T,MType,STORE>::getBackOutqueue(){}

template<class Derived,class FQ_T,class MType,class STORE>
void GlobalBFS<Derived,FQ_T,MType,STORE>::setBackInqueue(){}

/*
 * Generates a map of the vertex with predecessor
 */
template<class Derived,class FQ_T,class MType,class STORE>
void GlobalBFS<Derived,FQ_T,MType,STORE>::generatOwenMask()
{
    #pragma omp parallel for
    for(long i=0; i < mask_size ; i++){
        MType tmp = 0;
        for(long j=0; j < 8*sizeof(MType); j++){
            if( predecessor[ i*8*sizeof(MType) + j] != -1 )
                tmp |= 1 << j;
        }
        owenmask[i] = tmp;
    }

}

template<class Derived,class FQ_T,class MType,class STORE>
GlobalBFS<Derived,FQ_T,MType,STORE>::GlobalBFS(STORE &_store): store(_store)
{
     // Split communicator into row and column communicator
     // Split by row, rank by column
     MPI_Comm_split(MPI_COMM_WORLD, store.getLocalRowID(), store.getLocalColumnID(), &row_comm);
     // Split by column, rank by row
     MPI_Comm_split(MPI_COMM_WORLD, store.getLocalColumnID(), store.getLocalRowID(), &col_comm);

     fold_fq_props = store.getFoldProperties();

     mask_size = (store.getLocColLength()/(8*sizeof(MType)))+((store.getLocColLength()%(8*sizeof(MType))>0)? 1 : 0);
     owenmask = new MType[mask_size];
     tmpmask = new MType[mask_size];
}
template<class Derived,class FQ_T,class MType,class STORE>
GlobalBFS<Derived,FQ_T,MType,STORE>::~GlobalBFS(){
    delete[] owenmask;
    delete[] tmpmask;
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
template<class Derived,class FQ_T,class MType,class STORE>
void GlobalBFS<Derived,FQ_T,MType,STORE>::runBFS(typename STORE::vtxtyp startVertex, double& lexp, double& lqueue, double& rowcom, double& colcom, double& predlistred)
#else
template<class Derived,class FQ_T,class MType,class STORE>
void GlobalBFS<Derived,FQ_T,MType,STORE>::runBFS(typename STORE::vtxtyp startVertex)
#endif
{
    #ifdef INSTRUMENTED
    double tstart, tend;
    lexp =0;
    lqueue =0;
    double comtstart, comtend;
    rowcom = 0;
    colcom = 0;
    #endif

// 0
    MPI_Bcast(&startVertex,1,MPI_LONG,0,MPI_COMM_WORLD);
// 1
    #ifdef INSTRUMENTED
    tstart = MPI_Wtime();
    #endif
    static_cast<Derived*>(this)->setStartVertex(startVertex);
    #ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue +=tend-tstart;
    #endif
// 2
    while(true){
    #ifdef INSTRUMENTED
    tstart = MPI_Wtime();
    #endif
    static_cast<Derived*>(this)->runLocalBFS();
    #ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lexp +=tend-tstart;
    #endif
// 3
    int anynewnodes, anynewnodes_global;
    #ifdef INSTRUMENTED
    tstart = MPI_Wtime();
    #endif
    anynewnodes = static_cast<Derived*>(this)->istheresomethingnew();
    #ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue +=tend-tstart;
    #endif

    MPI_Allreduce(&anynewnodes, &anynewnodes_global,1,MPI_INT,MPI_LOR,MPI_COMM_WORLD);
    if(!anynewnodes_global){
        #ifdef INSTRUMENTED
        tstart = MPI_Wtime();
        #endif
        static_cast<Derived*>(this)->getBackPredecessor();
        #ifdef INSTRUMENTED
        tend = MPI_Wtime();
        lqueue += tend-tstart;
        #endif
        //MPI_Allreduce(MPI_IN_PLACE, predecessor ,store.getLocColLength(),MPI_LONG,MPI_MAX,col_comm);
        generatOwenMask();
        allReduceBitCompressed(predecessor,
                               recv_fq_buff, // have to be changed for bitmap queue
                                        owenmask, tmpmask);
        #ifdef INSTRUMENTED
        tend = MPI_Wtime();
        predlistred = tend-tstart;
        #endif
        return; //There is nothing too do. Finish iteration.
    }
// 4
    #ifdef INSTRUMENTED
    comtstart = MPI_Wtime();
    #endif
    #ifdef INSTRUMENTED
    tstart = MPI_Wtime();
    #endif
    static_cast<Derived*>(this)->getBackOutqueue();
    #ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue +=tend-tstart;
    #endif
    // tree based reduce operation with messages of variable size
    // root 0
    int rounds = 0;
    while((1 << rounds) < store.getNumRowSl() ){
        if((store.getLocalRowID() >> rounds)%2 == 1){
            FQ_T* startaddr_fq;
            long _outsize;
            //comute recv addr
            int recv_addr = (store.getLocalRowID() + store.getNumRowSl() - (1 << rounds)) % store.getNumRowSl();
            //get fq to send
            #ifdef INSTRUMENTED
            tstart = MPI_Wtime();
            #endif
            static_cast<Derived*>(this)->getOutgoingFQ(startaddr_fq, _outsize);
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
            static_cast<Derived*>(this)->reduce_fq_out(recv_fq_buff,static_cast<long>(count));
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
        FQ_T* startaddr_fq;
        long _outsize;
        //get fq to send
        #ifdef INSTRUMENTED
        tstart = MPI_Wtime();
        #endif
        static_cast<Derived*>(this)->getOutgoingFQ(startaddr_fq, _outsize);
        #ifdef INSTRUMENTED
        tend = MPI_Wtime();
        lqueue +=tend-tstart;
        #endif
        MPI_Bcast(&_outsize,1,MPI_LONG,0 ,col_comm);
        MPI_Bcast(startaddr_fq,_outsize,fq_tp_type,0 ,col_comm);
        #ifdef INSTRUMENTED
        tstart = MPI_Wtime();
        #endif
        static_cast<Derived*>(this)->setModOutgoingFQ(0,_outsize);
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
        static_cast<Derived*>(this)->setModOutgoingFQ(recv_fq_buff,_outsize);
        #ifdef INSTRUMENTED
        tend = MPI_Wtime();
        lqueue +=tend-tstart;
        #endif
    }
    #ifdef INSTRUMENTED
    comtend = MPI_Wtime();
    colcom += comtend-comtstart;
    #endif

// 5
    #ifdef INSTRUMENTED
    comtstart = MPI_Wtime();
    #endif
    for(typename std::vector<typename STORE::fold_prop>::iterator it = fold_fq_props.begin(); it  != fold_fq_props.end(); it++){
        if(it->sendColSl == store.getLocalColumnID() ){
            FQ_T*   startaddr;
            long     outsize;
            #ifdef INSTRUMENTED
            tstart = MPI_Wtime();
            #endif
            static_cast<Derived*>(this)->getOutgoingFQ(it->startvtx, it->size, startaddr, outsize);
            #ifdef INSTRUMENTED
            tend = MPI_Wtime();
            lqueue +=tend-tstart;
            #endif
            MPI_Bcast(&outsize,1,MPI_LONG,it->sendColSl,row_comm);
            MPI_Bcast(startaddr,outsize,fq_tp_type,it->sendColSl,row_comm);
            #ifdef INSTRUMENTED
            tstart = MPI_Wtime();
            #endif
            static_cast<Derived*>(this)->setIncommingFQ(it->startvtx, it->size, startaddr, outsize);
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
            static_cast<Derived*>(this)->setIncommingFQ(it->startvtx, it->size, recv_fq_buff, outsize);
            #ifdef INSTRUMENTED
            tend = MPI_Wtime();
            lqueue +=tend-tstart;
            #endif
        }
    }
    #ifdef INSTRUMENTED
    tstart = MPI_Wtime();
    #endif
    static_cast<Derived*>(this)->setBackInqueue();
    #ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue +=tend-tstart;
    #endif

    #ifdef INSTRUMENTED
    comtend = MPI_Wtime();
    rowcom += comtend - comtstart;
    #endif
}
}

template<class Derived,class FQ_T,class MType,class STORE>
typename STORE::vtxtyp *GlobalBFS<Derived, FQ_T, MType, STORE>::getPredecessor()
{
    return  predecessor;
}

#endif // GLOBALBFS_HH
