
#include "globalbfs.hh"



/*
#ifdef _SIMDCOMPRESS
    #include "codecfactory.h"
    #include "intersection.h"
	using namespace SIMDCompressionLib;
#endif


#ifdef _SIMDCOMPRESS
    IntegerCODEC &codec =  * CODECFactory::getFromName("s4-bp128-dm");
#endif
*/

/**********************************************************************************
 * BFS search:
 * 0) Node 0 sends start vertex to all nodes
 * 1) Nodes test, if they are responsible for this vertex and push it,
 *    if they are in there fq
 * 2) Local expansion
 * 3) Test if anything is done
 * 4) global expansion
 * 5) global fold
 **********************************************************************************/
  // <Derived, FQ_T, MType, STORE>
#ifdef INSTRUMENTED
    template<class Derived,class FQ_T,class MType,class STORE>
    void GlobalBFS<Derived, FQ_T, MType, STORE>::runBFS(typename STORE::vtxtyp startVertex, double& lexp, double& lqueue, double& rowcom, double& colcom, double& predlistred)
#else
    template<class Derived, class FQ_T, class MType, class STORE>
    void GlobalBFS<Derived, FQ_T, MType, STORE>::runBFS(typename STORE::vtxtyp startVertex)
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

// 0) Node 0 sends start vertex to all nodes
    MPI_Bcast(&startVertex, 1, MPI_LONG, 0, MPI_COMM_WORLD);

// 1) Nodes test, if they are responsible for this vertex and push it, if they are in there fq
#ifdef INSTRUMENTED
    tstart = MPI_Wtime();
#endif

    static_cast<Derived *>(this)->setStartVertex(startVertex);

#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue +=tend-tstart;
#endif

// 2) Local expansion
    int iter = 0;
    while (true) {

#ifdef INSTRUMENTED
    tstart = MPI_Wtime();
#endif

        static_cast<Derived *>(this)->runLocalBFS();

#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lexp +=tend-tstart;
#endif

// 3) Test if anything is done
        int anynewnodes, anynewnodes_global;

#ifdef INSTRUMENTED
    tstart = MPI_Wtime();
#endif

        anynewnodes = static_cast<Derived *>(this)->istheresomethingnew();

#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue +=tend-tstart;
#endif

        MPI_Allreduce(&anynewnodes, &anynewnodes_global, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if (!anynewnodes_global) {

#ifdef INSTRUMENTED
    tstart = MPI_Wtime();
#endif

            static_cast<Derived *>(this)->getBackPredecessor();

#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue += tend-tstart;
#endif

            // MPI_Allreduce(MPI_IN_PLACE, predecessor ,store.getLocColLength(),MPI_LONG,MPI_MAX,col_comm);
            static_cast<Derived *>(this)->generatOwenMask();
            allReduceBitCompressed(predecessor,
                                   recv_fq_buff, // have to be changed for bitmap queue
                                   owenmask, tmpmask);

#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    predlistred = tend-tstart;
#endif

            return; // There is nothing too do. Finish iteration.
        }

// 4) global expansion
#ifdef INSTRUMENTED
    comtstart = MPI_Wtime();
    tstart = MPI_Wtime();
#endif

        static_cast<Derived *>(this)->getBackOutqueue();

#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue +=tend-tstart;
#endif

        int _outsize; //really int, because mpi supports no long message sizes :(
        using namespace std::placeholders;
        std::function <void(FQ_T, long, FQ_T *, int)> reduce =
                std::bind(static_cast<void (Derived::*)(FQ_T, long, FQ_T *, int)>(&Derived::reduce_fq_out),
                          static_cast<Derived *>(this), _1, _2, _3, _4);
        std::function <void(FQ_T, long, FQ_T *&, int &)> get =
                std::bind(static_cast<void (Derived::*)(FQ_T, long, FQ_T *&, int &)>(&Derived::getOutgoingFQ),
                          static_cast<Derived *>(this), _1, _2, _3, _4);

        vreduce(reduce, get,
                recv_fq_buff,
                _outsize,
                store.getLocColLength(),
                fq_tp_type,
                col_comm

#ifdef INSTRUMENTED
                 ,lqueue
#endif
        );

        static_cast<Derived *>(this)->setModOutgoingFQ(recv_fq_buff, _outsize);

#ifdef INSTRUMENTED
    comtend = MPI_Wtime();
    colcom += comtend-comtstart;
#endif

// 5) global fold
#ifdef INSTRUMENTED
    comtstart = MPI_Wtime();
#endif

        for (typename std::vector<typename STORE::fold_prop>::iterator it = fold_fq_props.begin();
             it != fold_fq_props.end(); ++it) {
            if (it->sendColSl == store.getLocalColumnID()) {
                FQ_T *startaddr;
                int outsize;

#ifdef INSTRUMENTED
    tstart = MPI_Wtime();
#endif

                static_cast<Derived *>(this)->getOutgoingFQ(it->startvtx, it->size, startaddr, outsize);

#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue +=tend-tstart;
#endif

                MPI_Bcast(&outsize, 1, MPI_LONG, it->sendColSl, row_comm);
                MPI_Bcast(startaddr, outsize, fq_tp_type, it->sendColSl, row_comm);

#ifdef INSTRUMENTED
    tstart = MPI_Wtime();
#endif

                static_cast<Derived *>(this)->setIncommingFQ(it->startvtx, it->size, startaddr, outsize);

#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue +=tend-tstart;
#endif

            } else {
                int outsize;
                MPI_Bcast(&outsize, 1, MPI_LONG, it->sendColSl, row_comm);
                assert(outsize <= recv_fq_buff_length);
                MPI_Bcast(recv_fq_buff, outsize, fq_tp_type, it->sendColSl, row_comm);

#ifdef INSTRUMENTED
    tstart = MPI_Wtime();
#endif

                static_cast<Derived *>(this)->setIncommingFQ(it->startvtx, it->size, recv_fq_buff, outsize);

#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue +=tend-tstart;
#endif
            }
        }

#ifdef INSTRUMENTED
    tstart = MPI_Wtime();
#endif

        static_cast<Derived *>(this)->setBackInqueue();

#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue +=tend-tstart;
#endif

#ifdef INSTRUMENTED
    comtend = MPI_Wtime();
    rowcom += comtend - comtstart;
#endif
        ++iter;
    }
}
