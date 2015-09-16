#ifndef GLOBALBFS_HH
#define GLOBALBFS_HH


#include "distmatrix2d.hh"
#include "comp_opt.h"
#include "bitlevelfunctions.h"
#include <vector>
#include <cstdio>
#include <assert.h>
#include "vreduce.hpp"
#include <ctgmath>
#include <string.h>
#include <functional>

#ifdef _SCOREP_USER_INSTRUMENTATION
    #include "scorep/SCOREP_User.h"
#endif

#ifdef _SIMDCOMPRESS
    #include "codecfactory.h"
    using namespace SIMDCompressionLib;
#endif

#ifdef INSTRUMENTED
    #include <unistd.h>
    #include <chrono>
    using namespace std::chrono;
#endif


/*
 * This classs implements a distributed level synchronus BFS on global scale.
 */
template<class Derived,
        class FQ_T,  // Queue Type
        class MType, // Bitmap mask
        class STORE> //Storage of Matrix
class GlobalBFS {
private:
    // Set to -1 to transparently disable SIMD(de)compression
    std::size_t SIMDCOMPRESSION_THRESHOLD = 20;
    MPI_Comm row_comm, col_comm;
    int rank;
    // sending node column slice, startvtx, size
    std::vector <typename STORE::fold_prop> fold_fq_props;
    void allReduceBitCompressed(typename STORE::vtxtyp *&owen, typename STORE::vtxtyp *&tmp,
                                MType *&owenmap, MType *&tmpmap);

// Todo: export-to-class
#ifdef _SIMDCOMPRESS

    /**
     * SIMD integration calls
     *
     *
     */
    void SIMDbenchmarkCompression(FQ_T *fq, int size, int rank) const;
    void SIMDverifyCompression(FQ_T *fq, FQ_T *uncompressed_fq_64, std::size_t uncompressedsize) const;
    void SIMDcompression(IntegerCODEC &codec, FQ_T *fq_64, std::size_t &size, FQ_T *&compressed_fq_64, std::size_t &compressedsize) const;
    void SIMDdecompression(IntegerCODEC &codec, FQ_T *compressed_fq_64, int size, FQ_T *&uncompressed_fq_64,
                                                                                                    std::size_t &uncompressedsize) const;
#endif

#ifdef INSTRUMENTED
    std::size_t getTotalSystemMemory();
#endif

protected:
    const STORE &store;
    typename STORE::vtxtyp *predecessor;
    MPI_Datatype fq_tp_type; //Frontier Queue Transport Type
    MPI_Datatype bm_type;    // Bitmap Type
    // FQ_T*  __restrict__ recv_fq_buff; - conflicts with void* ref
    FQ_T *recv_fq_buff;
    //, *compressed_fq_64; // uncompressed and compressed column-buffers
    long recv_fq_buff_length;
    MType *owenmask;
    MType *tmpmask;
    int64_t mask_size;

    /**
     * Inherited methods in children classes: cuda_bfs.cu (CUDA), cpubfs_bin.cpp (CPU improved) and simplecpubfs.cpp (CPU basic)
     *
     *
     */
    // Functions that have to be implemented by the children
    // void reduce_fq_out(FQ_T* startaddr, long insize)=0;  //Global Reducer of the local outgoing frontier queues. Have to be implemented by the children.
    // void getOutgoingFQ(FQ_T* &startaddr, vtxtype& outsize)=0;
    // void setModOutgoingFQ(FQ_T* startaddr, long insize)=0; //startaddr: 0, self modification
    // void getOutgoingFQ(vtxtype globalstart, vtxtype size, FQ_T* &startaddr, vtxtype& outsize)=0;
    // void setIncommingFQ(vtxtype globalstart, vtxtype size, FQ_T* startaddr, vtxtype& insize_max)=0;
    // bool istheresomethingnew()=0;           //to detect if finished
    // void setStartVertex(const vtxtype start)=0;
    // void runLocalBFS()=0;
    // For accelerators with own memory

    void getBackPredecessor(); // expected to be used afet the application finished
    void getBackOutqueue();
    void setBackInqueue();
    void generatOwenMask();

public:
    /**
     * Constructor & destructor declaration
     */
    GlobalBFS(STORE &_store, int _rank);
    ~GlobalBFS();

    typename STORE::vtxtyp *getPredecessor();

#ifdef INSTRUMENTED
    void runBFS(typename STORE::vtxtyp startVertex, double& lexp, double &lqueue, double& rowcom, double& colcom, double& predlistred);
#else
    void runBFS(typename STORE::vtxtyp startVertex);
#endif

};


/**
 * Constructor
 *
 */
template<class Derived, class FQ_T, class MType, class STORE>
GlobalBFS<Derived, FQ_T, MType, STORE>::GlobalBFS(STORE &_store, int _rank) : store(_store) {
    int mtypesize = 8 * sizeof(MType);
    // Split communicator into row and column communicator
    // Split by row, rank by column
    MPI_Comm_split(MPI_COMM_WORLD, store.getLocalRowID(), store.getLocalColumnID(), &row_comm);
    // Split by column, rank by row
    MPI_Comm_split(MPI_COMM_WORLD, store.getLocalColumnID(), store.getLocalRowID(), &col_comm);
    fold_fq_props = store.getFoldProperties();
    mask_size = (store.getLocColLength() / mtypesize) + ((store.getLocColLength() % mtypesize > 0) ? 1 : 0);
    owenmask = new MType[mask_size];
    tmpmask = new MType[mask_size];
    rank = _rank;
}

/**
 * Destructor
 *
 */
template<class Derived, class FQ_T, class MType, class STORE>
GlobalBFS<Derived, FQ_T, MType, STORE>::~GlobalBFS() {
    delete[] owenmask;
    delete[] tmpmask;
}

/**
 * getPredecessor()
 *
 */
template<class Derived, class FQ_T, class MType, class STORE>
typename STORE::vtxtyp *GlobalBFS<Derived, FQ_T, MType, STORE>::getPredecessor() {
    return predecessor;
}

/*
 * allReduceBitCompressed()
 * Bitmap compression on predecessor reduction
 *
 */
template<class Derived, class FQ_T, class MType, class STORE>
void GlobalBFS<Derived, FQ_T, MType, STORE>::allReduceBitCompressed(typename STORE::vtxtyp *&owen,
                                                                    typename STORE::vtxtyp *&tmp, MType *&owenmap,
                                                                    MType *&tmpmap) {
    MPI_Status status;
    int communicatorSize, communicatorRank, intLdSize, power2intLdSize, residuum;
    int psize = mask_size;
    int mtypesize = 8 * sizeof(MType);
    //step 1
    MPI_Comm_size(col_comm, &communicatorSize);
    MPI_Comm_rank(col_comm, &communicatorRank);

    intLdSize = ilogb(static_cast<double>(communicatorSize)); //integer log_2 of size
    power2intLdSize = 1 << intLdSize; // 2^n
    residuum = communicatorSize - (1 << intLdSize);

    //step 2
    if (communicatorRank < 2 * residuum) {
        if ((communicatorRank & 1) == 0) { // even
            MPI_Sendrecv(owenmap, psize, bm_type, communicatorRank + 1, 0, tmpmap, psize, bm_type, communicatorRank + 1, 0,
                         col_comm, &status);
            for (int i = 0; i < psize; ++i) {
                tmpmap[i] &= ~owenmap[i];
                owenmap[i] |= tmpmap[i];
            }

            MPI_Recv(tmp, store.getLocColLength(), fq_tp_type, communicatorRank + 1, 1, col_comm, &status);
            int p = 0;
            for (int i = 0; i < psize; ++i) {
                MType tmpm = tmpmap[i];
                int size = i * mtypesize;
                while (tmpm != 0) {
                    int last = ffsl(tmpm) - 1;
                    owen[size + last] = tmp[p];
                    ++p;
                    tmpm ^= (1 << last);
                }
            }

        } else { // odd
            MPI_Sendrecv(owenmap, psize, bm_type, communicatorRank - 1, 0, tmpmap, psize, bm_type, communicatorRank - 1, 0,
                                                                                                    col_comm, &status);
            for (int i = 0; i < psize; ++i) {
                tmpmap[i] = ~tmpmap[i] & owenmap[i];
            }
            int p = 0;
            for (int i = 0; i < psize; ++i) {
                MType tmpm = tmpmap[i];
                int size = i * mtypesize;
                while (tmpm != 0) {
                    int last = ffsl(tmpm) - 1;
                    tmp[p] = owen[size + last];
                    ++p;
                    tmpm ^= (1 << last);
                }
            }
            MPI_Send(tmp, p, fq_tp_type, communicatorRank - 1, 1, col_comm);
        }
    }
    const std::function <int(int)> newRank = [&residuum](int oldr) {
        return (oldr < 2 * residuum) ? oldr / 2 : oldr - residuum;
    };
    const std::function <int(int)> oldRank = [&residuum](int newr) {
        return (newr < residuum) ? newr * 2 : newr + residuum;
    };

    if ((((communicatorRank & 1) == 0) && (communicatorRank < 2 * residuum)) || (communicatorRank >= 2 * residuum)) {
        int ssize, vrank, offset, lowers, uppers, size, index, ioffset;

        ssize = psize;
        vrank = newRank(communicatorRank);
        offset = 0;

        for (int it = 0; it < intLdSize; ++it) {
            lowers = ssize / 2; //lower slice size
            uppers = ssize - lowers; //upper slice size
            size = lowers * mtypesize;

            if (((vrank >> it) & 1) == 0) {// even
                //Transmission of the the bitmap
                MPI_Sendrecv(owenmap + offset, ssize, bm_type, oldRank((vrank + (1 << it)) & (power2intLdSize - 1)), (it << 1) + 2,
                             tmpmap + offset, ssize, bm_type, oldRank((vrank + (1 << it)) & (power2intLdSize - 1)), (it << 1) + 2,
                             col_comm, &status);

                for (int i = 0; i < lowers; ++i) {
                    ioffset = i + offset;
                    tmpmap[ioffset] &= ~owenmap[ioffset];
                    owenmap[ioffset] |= tmpmap[ioffset];
                }
                for (int i = lowers; i < ssize; ++i) {
                    ioffset = i + offset;
                    tmpmap[ioffset] = (~tmpmap[ioffset]) & owenmap[ioffset];
                }
                //Generation of foreign updates
                int p = 0;
                for (int i = 0; i < uppers; ++i) {
                    MType tmpm = tmpmap[i + offset + lowers];
                    index = (i + offset + lowers) * mtypesize;
                    while (tmpm != 0) {
                        int last = ffsl(tmpm) - 1;
                        tmp[size + p] = owen[index + last];
                        ++p;
                        tmpm ^= (1 << last);
                    }
                }
                //Transmission of updates
                MPI_Sendrecv(tmp + size, p, fq_tp_type,
                             oldRank((vrank + (1 << it)) & (power2intLdSize - 1)), (it << 1) + 3,
                             tmp, size, fq_tp_type,
                             oldRank((vrank + (1 << it)) & (power2intLdSize - 1)), (it << 1) + 3,
                             col_comm, &status);
                //Updates for own data
                p = 0;
                for (int i = 0; i < lowers; ++i) {
                    MType tmpm = tmpmap[offset + i];
                    index = (i + offset) * mtypesize;
                    while (tmpm != 0) {
                        int last = ffsl(tmpm) - 1;
                        owen[index + last] = tmp[p];
                        ++p;
                        tmpm ^= (1 << last);
                    }
                }
                ssize = lowers;
            } else { // odd
                //Transmission of the the bitmap
                MPI_Sendrecv(owenmap + offset, ssize, bm_type,
                             oldRank((power2intLdSize + vrank - (1 << it)) & (power2intLdSize - 1)), (it << 1) + 2,
                             tmpmap + offset, ssize, bm_type,
                             oldRank((power2intLdSize + vrank - (1 << it)) & (power2intLdSize - 1)), (it << 1) + 2,
                             col_comm, &status);
                for (int i = 0; i < lowers; ++i) {
                    tmpmap[i + offset] = (~tmpmap[i + offset]) & owenmap[i + offset];
                }
                for (int i = lowers; i < ssize; ++i) {
                    tmpmap[i + offset] &= ~owenmap[i + offset];
                    owenmap[i + offset] |= tmpmap[i + offset];
                }
                //Generation of foreign updates
                int p = 0;
                for (int i = 0; i < lowers; ++i) {
                    MType tmpm = tmpmap[i + offset];
                    while (tmpm != 0) {
                        int last = ffsl(tmpm) - 1;
                        tmp[p] = owen[(i + offset) * mtypesize + last];
                        ++p;
                        tmpm ^= (1 << last);
                    }
                }
                //Transmission of updates
                MPI_Sendrecv(tmp, p, fq_tp_type,
                             oldRank((power2intLdSize + vrank - (1 << it)) & (power2intLdSize - 1)), (it << 1) + 3,
                             tmp + size, uppers * mtypesize, fq_tp_type,
                             oldRank((power2intLdSize + vrank - (1 << it)) & (power2intLdSize - 1)), (it << 1) + 3,
                             col_comm, &status);

                //Updates for own data
                p = 0;
                for (int i = 0; i < uppers; ++i) {
                    MType tmpm = tmpmap[offset + lowers + i];
                    int lindex = (i + offset + lowers) * mtypesize;
                    while (tmpm != 0) {
                        int last = ffsl(tmpm) - 1;
                        owen[lindex + last] = tmp[p + size];
                        ++p;
                        tmpm ^= (1 << last);
                    }
                }
                offset += lowers;
                ssize = uppers;
            }
        }
    }
    //Computation of displacements
    std::vector<int> sizes(communicatorSize);
    std::vector<int> disps(communicatorSize);

    unsigned int lastReversedSliceIDs = 0;
    unsigned int lastTargetNode = oldRank(lastReversedSliceIDs);

    sizes[lastTargetNode] = ((psize) >> intLdSize) * mtypesize;
    disps[lastTargetNode] = 0;

    for (unsigned int slice = 1; slice < power2intLdSize; ++slice) {
        unsigned int reversedSliceIDs = reverse(slice, intLdSize);
        unsigned int targetNode = oldRank(reversedSliceIDs);
        sizes[targetNode] = (psize >> intLdSize) * mtypesize;
        disps[targetNode] = ((slice * psize) >> intLdSize) * mtypesize;
        lastTargetNode = targetNode;
    }
    //nodes without a partial resulty
    int index;
    for (unsigned int node = 0; node < residuum; ++node) {
        index = 2 * node + 1;
        sizes[index] = 0;
        disps[index] = 0;
    }
    // Transmission of the final results
    MPI_Allgatherv(MPI_IN_PLACE, sizes[communicatorRank], fq_tp_type,
                   owen, &sizes[0], &disps[0], fq_tp_type, col_comm);
}

template<class Derived, class FQ_T, class MType, class STORE>
void GlobalBFS<Derived, FQ_T, MType, STORE>::getBackPredecessor() { }

template<class Derived, class FQ_T, class MType, class STORE>
void GlobalBFS<Derived, FQ_T, MType, STORE>::getBackOutqueue() { }

template<class Derived, class FQ_T, class MType, class STORE>
void GlobalBFS<Derived, FQ_T, MType, STORE>::setBackInqueue() { }

/*
 * Generates a map of the vertex with predecessor
 */
template<class Derived, class FQ_T, class MType, class STORE>
void GlobalBFS<Derived, FQ_T, MType, STORE>::generatOwenMask() {
    int mtypesize, store_col_length;

    mtypesize = 8 * sizeof(MType);
    store_col_length = store.getLocColLength();

#ifdef _OPENMP
    #pragma omp parallel for
#endif

    for (long i = 0; i < mask_size; ++i) {
        MType tmp = 0;
        int jindex, iindex = i * mtypesize;
        for (long j = 0; j < mtypesize; ++j) {
            jindex = iindex + j;
            if ((predecessor[jindex] != -1) && (jindex < store_col_length)) {
                tmp |= 1 << j;
            }
        }
        owenmask[i] = tmp;
    }
}



/**********************************************************************************
 * BFS search:
 * 0) Node 0 sends start vertex to all nodes
 * 1) Nodes test, if they are responsible for this vertex and push it,
 *    if they are in there fq
 * 2) Local expansion
 * 3) Test if anything is done
 * 4) global expansion: Column Communication
 * 5) global fold: Row Communication
 **********************************************************************************/
/**
 * runBFS
 *
 *
 */
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

#ifdef _SCOREP_USER_INSTRUMENTATION
    SCOREP_USER_REGION_DEFINE( BFSRUN_region_vertexBroadcast )
    SCOREP_USER_REGION_BEGIN( BFSRUN_region_vertexBroadcast, "BFSRUN_region_vertexBroadcast",SCOREP_USER_REGION_TYPE_COMMON )
#endif

// 0) Node 0 sends start vertex to all nodes
    MPI_Bcast(&startVertex, 1, MPI_LONG, 0, MPI_COMM_WORLD);

#ifdef _SCOREP_USER_INSTRUMENTATION
    SCOREP_USER_REGION_END( BFSRUN_region_vertexBroadcast )
#endif

// 1) Nodes test, if they are responsible for this vertex and push it, if they are in there fq
#ifdef INSTRUMENTED
    tstart = MPI_Wtime();
#endif

#ifdef _SCOREP_USER_INSTRUMENTATION
    SCOREP_USER_REGION_DEFINE( BFSRUN_region_nodesTest )
    SCOREP_USER_REGION_BEGIN( BFSRUN_region_nodesTest, "BFSRUN_region_nodesTest",SCOREP_USER_REGION_TYPE_COMMON )
#endif

    static_cast<Derived *>(this)->setStartVertex(startVertex);

#ifdef _SCOREP_USER_INSTRUMENTATION
    SCOREP_USER_REGION_END( BFSRUN_region_nodesTest )
#endif

#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue += tend - tstart;
#endif


// 2) Local expansion
    int iter = 0;

/**
 * Todo: refactor-extract
 *
 */
    while (true) {

#ifdef INSTRUMENTED
    tstart = MPI_Wtime();
#endif

#ifdef _SCOREP_USER_INSTRUMENTATION
    SCOREP_USER_REGION_DEFINE( BFSRUN_region_localExpansion )
    SCOREP_USER_REGION_BEGIN( BFSRUN_region_localExpansion, "BFSRUN_region_localExpansion",SCOREP_USER_REGION_TYPE_COMMON )
#endif

        static_cast<Derived *>(this)->runLocalBFS();

#ifdef _SCOREP_USER_INSTRUMENTATION
    SCOREP_USER_REGION_END( BFSRUN_region_localExpansion )
#endif

#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lexp += tend - tstart;
#endif

// 3) Test if anything is done
        int anynewnodes, anynewnodes_global;

#ifdef INSTRUMENTED
    tstart = MPI_Wtime();
#endif

#ifdef _SCOREP_USER_INSTRUMENTATION
    SCOREP_USER_REGION_DEFINE( BFSRUN_region_testSomethingHasBeenDone )
    SCOREP_USER_REGION_BEGIN( BFSRUN_region_testSomethingHasBeenDone, "BFSRUN_region_testSomethingHasBeenDone",SCOREP_USER_REGION_TYPE_COMMON )
#endif

        anynewnodes = static_cast<Derived *>(this)->istheresomethingnew();

#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue += tend - tstart;
#endif

        MPI_Allreduce(&anynewnodes, &anynewnodes_global, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);


        if (!anynewnodes_global) {


#ifdef INSTRUMENTED
    tstart = MPI_Wtime();
#endif

            static_cast<Derived *>(this)->getBackPredecessor();

#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue += tend - tstart;
#endif

            // MPI_Allreduce(MPI_IN_PLACE, predecessor ,store.getLocColLength(),MPI_LONG,MPI_MAX,col_comm);
            static_cast<Derived *>(this)->generatOwenMask();
            allReduceBitCompressed(predecessor,
                                   recv_fq_buff, // have to be changed for bitmap queue
                                   owenmask, tmpmask);


#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    predlistred = tend - tstart;
#endif

#ifdef _SCOREP_USER_INSTRUMENTATION
    SCOREP_USER_REGION_END( BFSRUN_region_testSomethingHasBeenDone )
#endif

            return; // There is nothing to do. Finish iteration.
        }


// 4) global expansion
#ifdef INSTRUMENTED
    comtstart = MPI_Wtime();
    tstart = MPI_Wtime();
#endif

#ifdef _SCOREP_USER_INSTRUMENTATION
    SCOREP_USER_REGION_DEFINE( BFSRUN_region_columnCommunication )
    SCOREP_USER_REGION_BEGIN( BFSRUN_region_columnCommunication, "BFSRUN_region_columnCommunication",SCOREP_USER_REGION_TYPE_COMMON )
#endif


        static_cast<Derived *>(this)->getBackOutqueue();

#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue += tend - tstart;
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


#ifdef _SCOREP_USER_INSTRUMENTATION
    SCOREP_USER_REGION_END( BFSRUN_region_columnCommunication )
#endif

#ifdef INSTRUMENTED
    comtend = MPI_Wtime();
    colcom += comtend-comtstart;
#endif

// 5) global fold
#ifdef INSTRUMENTED
    comtstart = MPI_Wtime();
#endif

#ifdef _SCOREP_USER_INSTRUMENTATION
    SCOREP_USER_REGION_DEFINE( BFSRUN_region_rowCommunication )
    SCOREP_USER_REGION_BEGIN( BFSRUN_region_rowCommunication, "BFSRUN_region_rowCommunication",SCOREP_USER_REGION_TYPE_COMMON )
#endif

#ifdef _SIMDCOMPRESS
        IntegerCODEC &codec = *CODECFactory::getFromName("s4-bp128-dm");
        std::size_t uncompressedsize, compressedsize;
#ifdef INSTRUMENTED
        std::size_t memory;
#endif
#endif

        int outsize;
        for (typename std::vector<typename STORE::fold_prop>::iterator it = fold_fq_props.begin();
                                                                            it != fold_fq_props.end(); ++it) {

            if (it->sendColSl == store.getLocalColumnID()) {

                FQ_T *startaddr;

#ifdef _SIMDCOMPRESS
                FQ_T *compressed_fq_64, *uncompressed_fq_64;
#endif

#ifdef INSTRUMENTED
    tstart = MPI_Wtime();
#endif

                static_cast<Derived *>(this)->getOutgoingFQ(it->startvtx, it->size, startaddr, outsize);

if (outsize > 20 && outsize < 40) {
    std::cout << std::endl << "POINT 0 - recv_fq_buff size: " << outsize << " rank: " << rank <<std::endl;
    for (int i=0; i <outsize; ++i) {
        std::cout << startaddr[i] << " ";
    }
    std::cout << std::endl << std::endl;
}

#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue += tend - tstart;
#endif

#ifdef _SIMDCOMPRESS
#ifdef _SIMDCOMPRESSBENCHMARK
                SIMDbenchmarkCompression(startaddr, outsize, rank);
#endif

                uncompressedsize = static_cast<std::size_t>(outsize);
                SIMDcompression(codec, startaddr, uncompressedsize, compressed_fq_64, compressedsize);



#ifdef INSTRUMENTED
                if (rank == 0) {
                    // debugs mem leaks
                    // memory=getTotalSystemMemory();
                    // printf("Memory: %lu, rank: %d\n", memory, rank);
                }
#endif
                /**
                 * Decompression test before broadcasted
                 */
                // uncompressedsize = static_cast<std::size_t>(outsize);
                // assert (uncompressedsize == outsize);
                // SIMDdecompression(codec, compressed_fq_64, compressedsize, uncompressed_fq_64, uncompressedsize);
                // SIMDverifyCompression(startaddr, uncompressed_fq_64, outsize);

#endif


#ifdef _SIMDCOMPRESS
                MPI_Bcast(&compressedsize, 1, MPI_LONG, it->sendColSl, row_comm);
                MPI_Bcast(&outsize, 1, MPI_LONG, it->sendColSl, row_comm);
                MPI_Bcast(compressed_fq_64, compressedsize, fq_tp_type, it->sendColSl, row_comm);
#else
                MPI_Bcast(&outsize, 1, MPI_LONG, it->sendColSl, row_comm);
                MPI_Bcast(startaddr, outsize, fq_tp_type, it->sendColSl, row_comm);
#endif

#ifdef _SIMDCOMPRESS
                uncompressedsize = static_cast<std::size_t>(outsize);
                assert (uncompressedsize == outsize);
                assert (compressedsize <= outsize);
                /*
                if (rank == 0){
                    uncompressed_fq_64 = startaddr;
                } else {
                    SIMDdecompression(codec, compressed_fq_64, compressedsize, uncompressed_fq_64, uncompressedsize);
                    assert (std::is_sorted(uncompressed_fq_64, uncompressed_fq_64+uncompressedsize));
                }*/

                // Save (G/C)PU cycles. decompression not needed for MPI rank 0 (root). The original array is available.
printf("--->1\n");
                if (rank != 0){
                    SIMDdecompression(codec, compressed_fq_64, compressedsize, startaddr, uncompressedsize);
                }
printf("--->1.1\n");

if (uncompressedsize > 20 && uncompressedsize < 40) {
    std::cout << std::endl << "POINT 1 - recv_fq_buff size: " << uncompressedsize << " rank: " << rank <<" rank2: "<< it->sendColSl <<std::endl;
    for (int i=0; i <uncompressedsize; ++i) {
        std::cout << uncompressed_fq_64[i] << " ";
    }
    std::cout << std::endl << std::endl;
}

#ifdef INSTRUMENTED
                if (rank == 0) {
                    // debugs mem leaks
                    // memory=getTotalSystemMemory();
                    // printf("Memory: %lu, rank: %d\n", memory, rank);
                }
#endif
#endif

#ifdef INSTRUMENTED
    tstart = MPI_Wtime();
#endif

/*
#ifdef _SIMDCOMPRESS
                static_cast<Derived *>(this)->setIncommingFQ(it->startvtx, it->size, startaddr, outsize);
#else
*/
                static_cast<Derived *>(this)->setIncommingFQ(it->startvtx, it->size, startaddr, outsize);
printf("--->1.2\n");
/*#endif*/

#ifdef _SIMDCOMPRESS
                if (outsize > SIMDCOMPRESSION_THRESHOLD && outsize != compressedsize && rank != 0) {
                    // there was compression and uncompression
                    delete[] uncompressed_fq_64;
                    delete[] compressed_fq_64;
                } else if (outsize > SIMDCOMPRESSION_THRESHOLD && outsize != compressedsize && rank == 0) {
                    delete[] compressed_fq_64;
                }
#endif
printf("--->1.3\n");

#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue += tend - tstart;
#endif

            } else {


#ifdef _SIMDCOMPRESS
                FQ_T *compressed_fq_64;
                int outsize, compressedsize;
                MPI_Bcast(&compressedsize, 1, MPI_LONG, it->sendColSl, row_comm);
                MPI_Bcast(&outsize, 1, MPI_LONG, it->sendColSl, row_comm);
                MPI_Bcast(compressed_fq_64, compressedsize, fq_tp_type, it->sendColSl, row_comm);

printf("--->2\n");
                /*
                uncompressedsize = static_cast<std::size_t>(outsize);
                SIMDdecompression(codec, recv_fq_buff, compressedsize, uncompressed_fq_64, uncompressedsize);
                assert (outsize == uncompressedsize);
                assert (std::is_sorted(uncompressed_fq_64, uncompressed_fq_64+uncompressedsize));
                */
                uncompressedsize = static_cast<std::size_t>(outsize);
                SIMDdecompression(codec, compressed_fq_64, compressedsize, recv_fq_buff, uncompressedsize);

printf("--->2.1\n");

if (uncompressedsize > 20 && uncompressedsize < 40) {
    std::cout << std::endl << "POINT 2 - recv_fq_buff size: " << uncompressedsize << " rank: " << rank <<std::endl;
    for (int i=0; i <uncompressedsize; ++i) {
        std::cout << recv_fq_buff[i] << " ";
    }
    std::cout << std::endl << std::endl;
}


#ifdef INSTRUMENTED
                if (rank == 0) {
                    // debugs mem leaks
                    // memory=getTotalSystemMemory();
                    // printf("Memory: %lu, rank: %d\n", memory, rank);
                }
#endif

#else
                int outsize;
                MPI_Bcast(&outsize, 1, MPI_LONG, it->sendColSl, row_comm);
                assert(outsize <= recv_fq_buff_length);
                MPI_Bcast(recv_fq_buff, outsize, fq_tp_type, it->sendColSl, row_comm);
#endif

#ifdef INSTRUMENTED
    tstart = MPI_Wtime();
#endif

printf("--->2.2\n");
                static_cast<Derived *>(this)->setIncommingFQ(it->startvtx, it->size, recv_fq_buff, outsize);
printf("--->2.3\n");
/*
#ifdef _SIMDCOMPRESS
                static_cast<Derived *>(this)->setIncommingFQ(it->startvtx, it->size, uncompressed_fq_64, outsize);
#else
                static_cast<Derived *>(this)->setIncommingFQ(it->startvtx, it->size, recv_fq_buff, outsize);
#endif
*/
#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue += tend - tstart;
#endif

/*
#ifdef _SIMDCOMPRESS
                if (outsize > SIMDCOMPRESSION_THRESHOLD && outsize != compressedsize) {
                    delete[] compressed_fq_64;
                }
#endif
*/
            }
        }


#ifdef INSTRUMENTED
    tstart = MPI_Wtime();
#endif

        static_cast<Derived *>(this)->setBackInqueue();

#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue += tend - tstart;
#endif

#ifdef INSTRUMENTED
    comtend = MPI_Wtime();
    rowcom += comtend - comtstart;
#endif

#ifdef _SCOREP_USER_INSTRUMENTATION
    SCOREP_USER_REGION_END( BFSRUN_region_rowCommunication )
#endif
        ++iter;
    }
}













#ifdef _SIMDCOMPRESS
/**
 *
 * TODO: extract to compression factory
 * simd-cpu-compression
 *
 *
 *
 */

/**
 * Benchmark compression/decompression.
 *
 */
template<class Derived, class FQ_T, class MType, class STORE>
void GlobalBFS<Derived, FQ_T, MType, STORE>::SIMDbenchmarkCompression(FQ_T *fq, int size, int _rank) const {
    if (size > SIMDCOMPRESSION_THRESHOLD) {
        char const *codec_name = "s4-bp128-dm";
        IntegerCODEC &codec =  *CODECFactory::getFromName(codec_name);
        high_resolution_clock::time_point time_0, time_1;
        std::vector<uint32_t>  fq_32(fq, fq + size);
        std::vector<uint32_t>  compressed_fq_32(size + 1024);
        std::vector<uint32_t>  uncompressed_fq_32(size);
        std::size_t compressedsize = compressed_fq_32.size();
        std::size_t uncompressedsize = uncompressed_fq_32.size();
        time_0 = high_resolution_clock::now();
        codec.encodeArray(fq_32.data(), fq_32.size(), compressed_fq_32.data(), compressedsize);
        time_1 = high_resolution_clock::now();
        auto encode_time = chrono::duration_cast<chrono::nanoseconds>(time_1-time_0).count();
        compressed_fq_32.resize(compressedsize);
        compressed_fq_32.shrink_to_fit();
        // TODO: Expensive Operation
        std::vector<FQ_T> compressed_fq_64(compressed_fq_32.begin(), compressed_fq_32.end());
        time_0 = high_resolution_clock::now();
        codec.decodeArray(compressed_fq_32.data(), compressed_fq_32.size(), uncompressed_fq_32.data(), uncompressedsize);
        time_1 = high_resolution_clock::now();
        auto decode_time = chrono::duration_cast<chrono::nanoseconds>(time_1-time_0).count();
        uncompressed_fq_32.resize(uncompressedsize);
        // TODO: Expensive Operation
        std::vector<FQ_T> uncompressed_fq_64(uncompressed_fq_32.begin(), uncompressed_fq_32.end());
        assert (size == uncompressedsize && std::equal(uncompressed_fq_64.begin(), uncompressed_fq_64.end(), fq));
        double compressedbits = 32.0 * static_cast<double>(compressed_fq_32.size()) / static_cast<double>(fq_32.size());
        double compressratio = (100.0 - 100.0 * compressedbits / 32.0);
        printf("SIMD.codec: %s, rank: %02d, c/d: %04ld/%04ldus, %02.3f%% gained\n", codec_name, _rank, encode_time, decode_time, compressratio);
    }
}

/**
 * SIMD compression.
 */
template<class Derived, class FQ_T, class MType, class STORE>
void GlobalBFS<Derived, FQ_T, MType, STORE>::SIMDcompression(IntegerCODEC &codec, FQ_T *fq_64, std::size_t &size, FQ_T *&compressed_fq_64,
                                                                                                        std::size_t &compressedsize) const {
     if (size > SIMDCOMPRESSION_THRESHOLD) {
        uint32_t *fq_32;
        uint32_t *compressed_fq_32 = new uint32_t[size+1024];
        fq_32 = new uint32_t[size];
        // std::copy((FQ_T *)fq_64, (FQ_T *)(fq_64+size), (uint32_t *)fq_32);
        std::copy(fq_64, fq_64+size, fq_32);
        compressedsize = size+1024;
        // char const *codec_name = "s4-bp128-dm";
        // IntegerCODEC &icodec =  *CODECFactory::getFromName(codec_name);
        codec.encodeArray(fq_32, size, compressed_fq_32, compressedsize);
        compressed_fq_64 = new FQ_T[compressedsize];
        // std::copy((uint32_t *)compressed_fq_32, (uint32_t *)(compressed_fq_32+compressedsize), (FQ_T *)compressed_fq_64);
        std::copy(compressed_fq_32, compressed_fq_32+compressedsize, compressed_fq_64);
        delete[] fq_32;
        delete[] compressed_fq_32;
     } else {
        /**
         * Buffer will not be compressed (Small size. Not worthed)
         */
        compressedsize = size;
        compressed_fq_64 = fq_64;
     }
}

/**
 * SIMD decompression.
 */
template<class Derived, class FQ_T, class MType, class STORE>
void GlobalBFS<Derived, FQ_T, MType, STORE>::SIMDdecompression(IntegerCODEC &codec, FQ_T *compressed_fq_64, int size,
                                                                FQ_T *&uncompressed_fq_64, std::size_t &uncompressedsize) const {
    if (uncompressedsize > SIMDCOMPRESSION_THRESHOLD && size != uncompressedsize) {
        uint32_t *uncompressed_fq_32 = new uint32_t[uncompressedsize];
        uint32_t *compressed_fq_32 = new uint32_t[size];
        std::copy(compressed_fq_64, compressed_fq_64+size, compressed_fq_32);
        // char const *codec_name = "s4-bp128-dm";
        // IntegerCODEC &icodec =  *CODECFactory::getFromName(codec_name);
        codec.decodeArray(compressed_fq_32, size, uncompressed_fq_32, uncompressedsize);
        uncompressed_fq_64 = new FQ_T[uncompressedsize];
        std::copy(uncompressed_fq_32, uncompressed_fq_32+uncompressedsize, uncompressed_fq_64);
        delete[] compressed_fq_32;
        delete[] uncompressed_fq_32;
    } else {
        uncompressedsize = size;
        uncompressed_fq_64 = compressed_fq_64;
    }
}

/**
 * SIMD compression/decompression verification.
 */
template<class Derived, class FQ_T, class MType, class STORE>
void GlobalBFS<Derived, FQ_T, MType, STORE>::SIMDverifyCompression(FQ_T *fq, FQ_T *uncompressed_fq_64, std::size_t uncompressedsize) const {
    if (uncompressedsize > SIMDCOMPRESSION_THRESHOLD) {
        if (std::equal(uncompressed_fq_64, uncompressed_fq_64 + uncompressedsize, fq)) {
            std::cout << "verification: compression-decompression OK." << std::endl;
        } else {
            std::cout << "verification: compression-decompression ERROR." << std::endl;
        }
    }
}
#endif

#ifdef INSTRUMENTED









/**
 * getTotalSystemMemory()
 * Used to manually search for Memory Leaks
 *
 *
 *
 */
    template<class Derived, class FQ_T, class MType, class STORE>
    std::size_t GlobalBFS<Derived, FQ_T, MType, STORE>::getTotalSystemMemory() {
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        return pages * page_size;
    }
#endif
#endif // GLOBALBFS_HH
