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
#include <stdlib.h>
#include <algorithm>
#ifdef _SCOREP_USER_INSTRUMENTATION
#include "scorep/SCOREP_User.h"
#endif

#ifdef INSTRUMENTED
#include <unistd.h>
#include <chrono>
using namespace std::chrono;
#endif

#ifdef _COMPRESSION
#include "compression/compression.hh"
#include "compression/types_compression.h"
#endif


using std::function;
using std::min;
using std::bind;
using std::vector;
using std::string;
using std::is_sorted;
using namespace std::placeholders;

/*
 * This classs implements a distributed level synchronus BFS on global scale.
 */
template <typename Derived,
          typename FQ_T,  // Queue Type
          typename MType, // Bitmap mask
          typename STORE> // Storage of Matrix
class GlobalBFS
{
private:
    MPI_Comm row_comm, col_comm;
    // sending node column slice, startvtx, size
    vector <typename STORE::fold_prop> fold_fq_props;
    void allReduceBitCompressed(typename STORE::vertexType *&owen, typename STORE::vertexType *&tmp,
                                MType *&owenmap, MType *&tmpmap);

protected:
    const STORE &store;
    typename STORE::vertexType *predecessor;
    MPI_Datatype fq_tp_type; //Frontier Queue Type
#ifdef _COMPRESSION
    MPI_Datatype fq_tp_typeC; //Compressed FQ
#endif
    MPI_Datatype bm_type;    // Bitmap Type
    // FQ_T*  __restrict__ fq_64; - conflicts with void* ref
    FQ_T *fq_64;
    // FQ_T *fq_64_slice;
    // compressionType *c_fq; // uncompressed and compressed column-buffers
    long fq_64_length;
    MType *owenmask;
    MType *tmpmask;
    int64_t mask_size;

    // Functions that have to be implemented by the children
    // void reduce_fq_out(FQ_T* startaddr, long insize)=0;  //Global Reducer of the local outgoing frontier queues. Have to be implemented by the children.
    // void getOutgoingFQ(FQ_T* &startaddr, vertexTypee& outsize)=0;
    // void setModOutgoingFQ(FQ_T* startaddr, long insize)=0; //startaddr: 0, self modification
    // void getOutgoingFQ(vertexTypee globalstart, vertexTypee size, FQ_T* &startaddr, vertexTypee& outsize)=0;
    // void setIncommingFQ(vertexTypee globalstart, vertexTypee size, FQ_T* startaddr, vertexTypee& insize_max)=0;
    // bool istheresomethingnew()=0;           //to detect if finished
    // void setStartVertex(const vertexTypee start)=0;
    // void runLocalBFS()=0;
    // For accelerators with own memory

    void getBackPredecessor(); // expected to be used afet the application finished
    void getBackOutqueue();
    void setBackInqueue();
    void generatOwenMask();

    // Uses the device memory calls to copy the MPI buffer. This buffer is created on the device (CPU, CUDA, OPENGL, etc)
    // using a malloc will cause a crash if the app is not in CPU mode. Implemented on the clidren class where also, the
    // buffer is created.
    void bfsMemCpy(FQ_T *&dst, FQ_T *src, size_t size);

public:
    /**
     * Constructor & destructor declaration
     */
    GlobalBFS(STORE &_store);
    ~GlobalBFS();

    typename STORE::vertexType *getPredecessor();

#ifdef _COMPRESSION
    typedef Compression<FQ_T, compressionType> CompressionClassT;
#endif

#ifdef INSTRUMENTED
#ifdef _COMPRESSION
    void runBFS(typename STORE::vertexType startVertex, double &lexp, double &lqueue, double &rowcom, double &colcom,
                double &predlistred, const CompressionClassT &schema);
#else
    void runBFS(typename STORE::vertexType startVertex, double &lexp, double &lqueue, double &rowcom, double &colcom,
                double &predlistred);
#endif

#else

#ifdef _COMPRESSION
    void runBFS(typename STORE::vertexType startVertex, const CompressionClassT &schema);
#else
    void runBFS(typename STORE::vertexType startVertex);
#endif
#endif

};

/**
 * Constructor
 *
 */
template<typename Derived, typename FQ_T, typename MType, typename STORE>
GlobalBFS<Derived, FQ_T, MType, STORE>::GlobalBFS(STORE &_store) : store(_store)
{
    int mtypesize = 8 * sizeof(MType);
    int local_column = store.getLocalColumnID(), local_row = store.getLocalRowID();
    // Split communicator into row and column communicator
    // Split by row, rank by column
    MPI_Comm_split(MPI_COMM_WORLD, local_row, local_column, &row_comm);
    // Split by column, rank by row
    MPI_Comm_split(MPI_COMM_WORLD, local_column, local_row, &col_comm);
    fold_fq_props = store.getFoldProperties();
    mask_size = (store.getLocColLength() / mtypesize) + ((store.getLocColLength() % mtypesize > 0) ? 1 : 0);
    owenmask = new MType[mask_size];
    tmpmask = new MType[mask_size];
}

/**
 * Destructor
 *
 */
template<typename Derived, typename FQ_T, typename MType, typename STORE>
GlobalBFS<Derived, FQ_T, MType, STORE>::~GlobalBFS()
{
    delete[] owenmask;
    delete[] tmpmask;
}

/**
 * getPredecessor()
 *
 */
template<typename Derived, typename FQ_T, typename MType, typename STORE>
typename STORE::vertexType *GlobalBFS<Derived, FQ_T, MType, STORE>::getPredecessor()
{
    return predecessor;
}

/*
 * allReduceBitCompressed()
 * Bitmap compression on predecessor reduction
 *
 */
template<typename Derived, typename FQ_T, typename MType, typename STORE>
void GlobalBFS<Derived, FQ_T, MType, STORE>::allReduceBitCompressed(typename STORE::vertexType *&owen,
        typename STORE::vertexType *&tmp, MType *&owenmap,
        MType *&tmpmap)
{
    MPI_Status status;
    int communicatorSize, communicatorRank;
    const int psize = mask_size;
    const int mtypesize = 8 * sizeof(MType);
    //step 1
    MPI_Comm_size(col_comm, &communicatorSize);
    MPI_Comm_rank(col_comm, &communicatorRank);

    const int intLdSize = ilogb(static_cast<double>(communicatorSize)); //integer log_2 of size
    const int power2intLdSize = 1 << intLdSize; // 2^n
    const int residuum = communicatorSize - power2intLdSize;


    //step 2
    if (communicatorRank < 2 * residuum)
    {
        if ((communicatorRank & 1) == 0)   // even
        {
            MPI_Sendrecv(owenmap, psize, bm_type, communicatorRank + 1, 0, tmpmap, psize, bm_type, communicatorRank + 1, 0,
                         col_comm, &status);
            for (int i = 0; i < psize; ++i)
            {
                tmpmap[i] &= ~owenmap[i];
                owenmap[i] |= tmpmap[i];
            }

            MPI_Recv(tmp, store.getLocColLength(), fq_tp_type, communicatorRank + 1, 1, col_comm, &status);
            // set recived elements where the bit maps indicate it
            int p = 0;
            for (int i = 0; i < psize; ++i)
            {
                MType tmpm = tmpmap[i];
                const int size = i * mtypesize;
                while (tmpm != 0)
                {
                    int last = ffsl(tmpm) - 1;
                    owen[size + last] = tmp[p];
                    ++p;
                    tmpm ^= (1 << last);
                }
            }
        }
        else     // odd
        {
            MPI_Sendrecv(owenmap, psize, bm_type, communicatorRank - 1, 0, tmpmap, psize, bm_type, communicatorRank - 1, 0,
                         col_comm, &status);
            for (int i = 0; i < psize; ++i)
            {
                tmpmap[i] = ~tmpmap[i] & owenmap[i];
            }
            int p = 0;
            for (int i = 0; i < psize; ++i)
            {
                MType tmpm = tmpmap[i];
                const int size = i * mtypesize;
                while (tmpm != 0)
                {
                    const int last = ffsl(tmpm) - 1;
                    tmp[p] = owen[size + last];
                    ++p;
                    tmpm ^= (1 << last);
                }
            }
            MPI_Send(tmp, p, fq_tp_type, communicatorRank - 1, 1, col_comm);
        }
    }
    const function <int(int)> newRank = [&residuum](int oldr)
    {
        return (oldr < 2 * residuum) ? oldr / 2 : oldr - residuum;
    };
    const function <int(int)> oldRank = [&residuum](int newr)
    {
        return (newr < residuum) ? newr * 2 : newr + residuum;
    };

    if ((((communicatorRank & 1) == 0) && (communicatorRank < 2 * residuum)) || (communicatorRank >= 2 * residuum))
    {
        int ssize, offset, size, index;

        ssize = psize;
        const int vrank = newRank(communicatorRank);
        offset = 0;

        for (int it = 0; it < intLdSize; ++it)
        {
            int orankEven, orankOdd, iterator2, iterator3;
            const int lowers = (int) ssize * 0.5f; //lower slice size
            const int uppers = ssize - lowers; //upper slice size
            size = lowers * mtypesize;
            orankEven = oldRank((vrank + (1 << it)) & (power2intLdSize - 1));
            orankOdd = oldRank((power2intLdSize + vrank - (1 << it)) & (power2intLdSize - 1));
            iterator2 = (it << 1) + 2;
            iterator3 = (it << 1) + 3;

            if (((vrank >> it) & 1) == 0)  // even
            {

                //Transmission of the the bitmap
                MPI_Sendrecv(owenmap + offset, ssize, bm_type, orankEven, iterator2,
                             tmpmap + offset, ssize, bm_type, orankEven, iterator2,
                             col_comm, &status);

                for (int i = 0; i < lowers; ++i)
                {
                    tmpmap[i + offset] &= ~owenmap[i + offset];
                    owenmap[i + offset] |= tmpmap[i + offset];
                }
                for (int i = lowers; i < ssize; ++i)
                {
                    tmpmap[i + offset] = (~tmpmap[i + offset]) & owenmap[i + offset];
                }
                //Generation of foreign updates
                int p = 0;
                for (int i = 0; i < uppers; ++i)
                {
                    MType tmpm = tmpmap[i + offset + lowers];
                    index = (i + offset + lowers) * mtypesize;
                    while (tmpm != 0)
                    {
                        int last = ffsl(tmpm) - 1;
                        tmp[size + p] = owen[index + last];
                        ++p;
                        tmpm ^= (1 << last);
                    }
                }
                //Transmission of updates
                MPI_Sendrecv(tmp + size, p, fq_tp_type,
                             orankEven, iterator3,
                             tmp, size, fq_tp_type,
                             orankEven, iterator3,
                             col_comm, &status);
                //Updates for own data
                p = 0;
                for (int i = 0; i < lowers; ++i)
                {
                    MType tmpm = tmpmap[offset + i];
                    index = (i + offset) * mtypesize;
                    while (tmpm != 0)
                    {
                        int last = ffsl(tmpm) - 1;
                        owen[index + last] = tmp[p];
                        ++p;
                        tmpm ^= (1 << last);
                    }
                }
                ssize = lowers;
            }
            else     // odd
            {
                //Transmission of the the bitmap
                MPI_Sendrecv(owenmap + offset, ssize, bm_type,
                             orankOdd, iterator2,
                             tmpmap + offset, ssize, bm_type,
                             orankOdd, iterator2,
                             col_comm, &status);
                for (int i = 0; i < lowers; ++i)
                {
                    tmpmap[i + offset] = (~tmpmap[i + offset]) & owenmap[i + offset];
                }
                for (int i = lowers; i < ssize; ++i)
                {
                    tmpmap[i + offset] &= ~owenmap[i + offset];
                    owenmap[i + offset] |= tmpmap[i + offset];
                }
                //Generation of foreign updates
                int p = 0;
                for (int i = 0; i < lowers; ++i)
                {
                    MType tmpm = tmpmap[i + offset];
                    while (tmpm != 0)
                    {
                        const int last = ffsl(tmpm) - 1;
                        tmp[p] = owen[(i + offset) * mtypesize + last];
                        ++p;
                        tmpm ^= (1 << last);
                    }
                }
                //Transmission of updates
                MPI_Sendrecv(tmp, p, fq_tp_type,
                             orankOdd, iterator3,
                             tmp + size, uppers * mtypesize, fq_tp_type,
                             orankOdd, iterator3,
                             col_comm, &status);

                //Updates for own data
                p = 0;
                for (int i = 0; i < uppers; ++i)
                {
                    MType tmpm = tmpmap[offset + lowers + i];
                    int lindex = (i + offset + lowers) * mtypesize;
                    while (tmpm != 0)
                    {
                        const int last = ffsl(tmpm) - 1;
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
    // Computation of displacements
    // It is based on the slice selection in the iterative part above.
    // It tries to do it iterative insted of recursive.
    //vector<int> sizes(communicatorSize);
    //vector<int> disps(communicatorSize);
    int *sizes = (int *)malloc(communicatorSize * sizeof(int));
    int *disps = (int *)malloc(communicatorSize * sizeof(int));

    const unsigned int maskLengthRes = psize % (1 << intLdSize);
    unsigned int lastReversedSliceIDs = 0;
    unsigned int lastTargetNode = oldRank(lastReversedSliceIDs);

    sizes[lastTargetNode] = (psize >> intLdSize) * mtypesize;
    disps[lastTargetNode] = 0;

    for (unsigned int slice = 1; slice < power2intLdSize; ++slice)
    {
        const unsigned int reversedSliceIDs = reverse(slice, intLdSize);
        const unsigned int targetNode = oldRank(reversedSliceIDs);
        sizes[targetNode] = ((psize >> intLdSize) + (((power2intLdSize - reversedSliceIDs - 1) < maskLengthRes) ? 1 : 0)) *
                            mtypesize;
        disps[targetNode] = disps[lastTargetNode] + sizes[lastTargetNode];
        lastTargetNode = targetNode;
    }
    sizes[lastTargetNode] = std::min(sizes[lastTargetNode],
                                     static_cast<int>(store.getLocColLength() - disps[lastTargetNode]));

    //nodes without a partial resulty
    int index;
    for (unsigned int node = 0; node < residuum; ++node)
    {
        index = 2 * node + 1;
        sizes[index] = 0;
        disps[index] = 0;
    }
    // Transmission of the final results
    MPI_Allgatherv(MPI_IN_PLACE, sizes[communicatorRank], fq_tp_type,
                   owen, sizes, disps, fq_tp_type, col_comm);
    free(sizes);
    free(disps);
}

template<typename Derived, typename FQ_T, typename MType, typename STORE>
void GlobalBFS<Derived, FQ_T, MType, STORE>::getBackPredecessor() { }

template<typename Derived, typename FQ_T, typename MType, typename STORE>
void GlobalBFS<Derived, FQ_T, MType, STORE>::getBackOutqueue() { }

template<typename Derived, typename FQ_T, typename MType, typename STORE>
void GlobalBFS<Derived, FQ_T, MType, STORE>::setBackInqueue() { }

/*
 * Generates a map of the vertex with predecessor
 */
template<typename Derived, typename FQ_T, typename MType, typename STORE>
void GlobalBFS<Derived, FQ_T, MType, STORE>::generatOwenMask()
{
    const long mtypesize = 8 * sizeof(MType);
    const long store_col_length = store.getLocColLength();

#ifdef _CUDA_OPENMP
    #pragma omp parallel
    {
        #pragma omp for schedule (guided, OMP_CHUNK)
#endif

        for (long i = 0L; i < mask_size; ++i)
        {
            MType tmp = 0;
            const long iindex = i * mtypesize;
            const long mask_word_end = std::min(mtypesize, store_col_length - iindex);
            for (long j = 0L; j < mask_word_end; ++j)
            {
                const long jindex = iindex + j;
                if (predecessor[jindex] != -1)
                {
                    tmp |= 1L << j;
                }
            }
            owenmask[i] = tmp;
        }

#ifdef _CUDA_OPENMP
    }
#endif
}

/**
 * runBFS
 *
 * BFS search:
 * 0) Node 0 sends start vertex to all nodes
 * 1) Nodes test, if they are responsible for this vertex and push it,
 *    if they are in there fq
 * 2) Local expansion
 * 3) Test if anything is done
 * 4) global expansion: Column Communication
 * 5) global fold: Row Communication
 */
template<typename Derived, typename FQ_T, typename MType, typename STORE>
#ifdef INSTRUMENTED
#ifdef _COMPRESSION
void GlobalBFS<Derived, FQ_T, MType, STORE>::runBFS(typename STORE::vertexType startVertex, double &lexp,
        double &lqueue,
        double &rowcom, double &colcom, double &predlistred, const CompressionClassT &schema)
#else
void GlobalBFS<Derived, FQ_T, MType, STORE>::runBFS(typename STORE::vertexType startVertex, double &lexp,
        double &lqueue,
        double &rowcom, double &colcom, double &predlistred)
#endif
#else
#ifdef _COMPRESSION
void GlobalBFS<Derived, FQ_T, MType, STORE>::runBFS(typename STORE::vertexType startVertex, const CompressionClassT &schema)
#else
void GlobalBFS<Derived, FQ_T, MType, STORE>::runBFS(typename STORE::vertexType startVertex)
#endif
#endif
{
    int communicatorRank;
    MPI_Comm_rank(col_comm, &communicatorRank); // current rank

#ifdef INSTRUMENTED
    double tstart, tend;
    lexp = 0;
    lqueue = 0;
    double comtstart, comtend;
    rowcom = 0;
    colcom = 0;
#endif

#ifdef _SCOREP_USER_INSTRUMENTATION
    SCOREP_USER_REGION_DEFINE(vertexBroadcast_handle)
    SCOREP_USER_REGION_DEFINE(localExpansion_handle)
    SCOREP_USER_REGION_DEFINE(columnCommunication_handle)
    SCOREP_USER_REGION_DEFINE(rowCommunication_handle)
    SCOREP_USER_REGION_DEFINE(allReduceBC_handle)
#endif

#ifdef _SCOREP_USER_INSTRUMENTATION
    SCOREP_USER_REGION_BEGIN(vertexBroadcast_handle, "BFSRUN_region_vertexBroadcast", SCOREP_USER_REGION_TYPE_COMMON)
#endif

// 0) Node 0 sends start vertex to all nodes
    MPI_Bcast(&startVertex, 1, MPI_LONG, 0, MPI_COMM_WORLD);

#ifdef _SCOREP_USER_INSTRUMENTATION
    SCOREP_USER_REGION_END(vertexBroadcast_handle)
#endif

// 1) Nodes test, if they are responsible for this vertex and push it, if they are in there fq
#ifdef INSTRUMENTED
    tstart = MPI_Wtime();
#endif

    static_cast<Derived *>(this)->setStartVertex(startVertex);

#ifdef INSTRUMENTED
    tend = MPI_Wtime();
    lqueue += tend - tstart;
#endif

// 2) Local expansion
    int iter = 0;


#ifdef _COMPRESSION
    size_t uncompressedsize, compressedsize;
#endif

    // moved anonymous functions outside loop
    const function <void(FQ_T, long, FQ_T *, int)> reduce =
        bind(static_cast<void (Derived::*)(FQ_T, long, FQ_T *, int)>(&Derived::reduce_fq_out),
             static_cast<Derived *>(this), _1, _2, _3, _4);
    const function <void(FQ_T, long, FQ_T *&, int &)> get =
        bind(static_cast<void (Derived::*)(FQ_T, long, FQ_T *&, int &)>(&Derived::getOutgoingFQ),
             static_cast<Derived *>(this), _1, _2, _3, _4);

    /**
     * Todo: refactor-extract
     *
     */
    while (true)
    {

#ifdef INSTRUMENTED
        tstart = MPI_Wtime();
#endif

#ifdef _SCOREP_USER_INSTRUMENTATION
        SCOREP_USER_REGION_BEGIN(localExpansion_handle, "BFSRUN_region_localExpansion", SCOREP_USER_REGION_TYPE_COMMON)
#endif

        static_cast<Derived *>(this)->runLocalBFS();

#ifdef _SCOREP_USER_INSTRUMENTATION
        SCOREP_USER_REGION_END(localExpansion_handle)
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

        anynewnodes = static_cast<Derived *>(this)->istheresomethingnew();

#ifdef INSTRUMENTED
        tend = MPI_Wtime();
        lqueue += tend - tstart;
#endif

        MPI_Allreduce(&anynewnodes, &anynewnodes_global, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

        if (!anynewnodes_global)
        {

#ifdef INSTRUMENTED
            tstart = MPI_Wtime();
#endif

            static_cast<Derived *>(this)->getBackPredecessor();

#ifdef INSTRUMENTED
            tend = MPI_Wtime();
            lqueue += tend - tstart;
#endif

#ifdef _SCOREP_USER_INSTRUMENTATION
            SCOREP_USER_REGION_BEGIN(allReduceBC_handle, "BFSRUN_region_allReduceBC", SCOREP_USER_REGION_TYPE_COMMON)
#endif

            // MPI_Allreduce(MPI_IN_PLACE, predecessor ,store.getLocColLength(),MPI_LONG,MPI_MAX,col_comm);
            static_cast<Derived *>(this)->generatOwenMask();
            allReduceBitCompressed(predecessor,
                                   fq_64, // have to be changed for bitmap queue
                                   owenmask, tmpmask);

#ifdef _SCOREP_USER_INSTRUMENTATION
            SCOREP_USER_REGION_END(allReduceBC_handle)
#endif

#ifdef INSTRUMENTED
            tend = MPI_Wtime();
            predlistred = tend - tstart;
#endif

            return; // There is nothing to do. Finish iteration.
        }

// 4) global expansion
#ifdef INSTRUMENTED
        comtstart = MPI_Wtime();
        tstart = MPI_Wtime();
#endif

#ifdef _SCOREP_USER_INSTRUMENTATION
        SCOREP_USER_REGION_BEGIN(columnCommunication_handle, "BFSRUN_region_columnCommunication",
                                 SCOREP_USER_REGION_TYPE_COMMON)
#endif

        static_cast<Derived *>(this)->getBackOutqueue();

#ifdef INSTRUMENTED
        tend = MPI_Wtime();
        lqueue += tend - tstart;
#endif

        int _outsize; //really int, because mpi supports no long message sizes :(

        vreduce(reduce, get,
#ifdef _COMPRESSION
                schema,
                fq_tp_typeC,
#endif
                fq_64,
                _outsize,
                store.getLocColLength(),
                fq_tp_type,
                col_comm

#ifdef INSTRUMENTED
                , lqueue
#endif
               );
/*
#ifdef _COMPRESSION
        mm = 0xffffff;
        str = "frameofreference";
        schema.reconfigure(mm, str);
#endif
*/
        static_cast<Derived *>(this)->setModOutgoingFQ(fq_64, _outsize);

#ifdef _SCOREP_USER_INSTRUMENTATION
        SCOREP_USER_REGION_END(columnCommunication_handle)
#endif

#ifdef INSTRUMENTED
        comtend = MPI_Wtime();
        colcom += comtend - comtstart;
#endif

// 5) global fold
#ifdef INSTRUMENTED
        comtstart = MPI_Wtime();
#endif

#ifdef _SCOREP_USER_INSTRUMENTATION
        SCOREP_USER_REGION_BEGIN(rowCommunication_handle, "BFSRUN_region_rowCommunication",
                                 SCOREP_USER_REGION_TYPE_LOOP)
#endif

        int root_rank;
        for (typename vector<typename STORE::fold_prop>::iterator it = fold_fq_props.begin();
             it != fold_fq_props.end(); ++it)
        {
            root_rank = it->sendColSl;
            if (root_rank == store.getLocalColumnID())
            {

                int originalsize;
                FQ_T *startaddr;
#ifdef _COMPRESSION
                compressionType *compressed_fq;
                FQ_T *uncompressed_fq;
#endif

#ifdef INSTRUMENTED
                tstart = MPI_Wtime();
#endif
                static_cast<Derived *>(this)->getOutgoingFQ(it->startvtx, it->size, startaddr, originalsize);

#ifdef INSTRUMENTED
                tend = MPI_Wtime();
                lqueue += tend - tstart;
#endif

#ifdef _COMPRESSION

#if defined(_COMPRESSIONDEBUG)
                schema.debugCompression(startaddr, originalsize);
#endif

                uncompressedsize = static_cast<size_t>(originalsize);
                schema.compress(startaddr, uncompressedsize, &compressed_fq, compressedsize);

#if defined(_COMPRESSIONVERIFY)
                schema.decompress(compressed_fq, compressedsize,  &uncompressed_fq, uncompressedsize);
#endif

#endif

#ifdef _COMPRESSION
                int *vectorizedsize = (int *)malloc(2 * sizeof(int));
                vectorizedsize[0] = originalsize;
                vectorizedsize[1] = compressedsize;
                MPI_Bcast(vectorizedsize, 1, MPI_2INT, root_rank, row_comm);
#if defined(_COMPRESSIONVERIFY)
		bool isCompressed = schema.isCompressed(originalsize, compressedsize);
		if (isCompressed)
		{
                MPI_Bcast(startaddr, originalsize, fq_tp_type, root_rank, row_comm);
		}
#endif

                MPI_Bcast(compressed_fq, compressedsize, fq_tp_typeC, root_rank, row_comm);

#else
                MPI_Bcast(&originalsize, 1, MPI_INT, root_rank, row_comm);
                MPI_Bcast(startaddr, originalsize, fq_tp_type, root_rank, row_comm);
#endif

#ifdef _COMPRESSION

                if (communicatorRank != root_rank)
                {
                    uncompressedsize = static_cast<size_t>(originalsize);
                    schema.decompress(compressed_fq, compressedsize,  &uncompressed_fq,  uncompressedsize);
                }

#if defined(_COMPRESSIONVERIFY)
                if (communicatorRank != root_rank)
                {
                        assert(memcmp(startaddr, uncompressed_fq, originalsize * sizeof(FQ_T)) == 0);
                    	assert(is_sorted(uncompressed_fq, uncompressed_fq + uncompressedsize));
                    	schema.verifyCompression(startaddr, uncompressed_fq, originalsize);
                }
                else
                {
                    assert(is_sorted(startaddr, startaddr + originalsize));
                }
#endif


#endif

#ifdef INSTRUMENTED
                tstart = MPI_Wtime();
#endif


#ifdef _COMPRESSION
                if (communicatorRank != root_rank)
        		{
        			static_cast<Derived *>(this)->setIncommingFQ(it->startvtx, it->size, uncompressed_fq, originalsize);
        		}
        		else
        		{
        			static_cast<Derived *>(this)->setIncommingFQ(it->startvtx, it->size, startaddr, originalsize);
        		}
#else
                static_cast<Derived *>(this)->setIncommingFQ(it->startvtx, it->size, startaddr, originalsize);
#endif

#ifdef _COMPRESSION
                    if (communicatorRank != root_rank)
                    {
                        free(uncompressed_fq);
                    }
                    free(compressed_fq);
                free(vectorizedsize);
#endif

#ifdef INSTRUMENTED
                tend = MPI_Wtime();
                lqueue += tend - tstart;
#endif

            }
            else
            {

#ifdef _COMPRESSION
                compressionType *compressed_fq = NULL;
                FQ_T *uncompressed_fq = NULL;
                int originalsize, compressedsize;
                int *vectorizedsize = (int *)malloc(2 * sizeof(int));

                MPI_Bcast(vectorizedsize, 1, MPI_2INT, root_rank, row_comm);
                originalsize = vectorizedsize[0];
                compressedsize = vectorizedsize[1];
       	        compressed_fq = (compressionType *)malloc(compressedsize * sizeof(compressionType));


#if defined(_COMPRESSIONVERIFY)
		bool isCompressed = schema.isCompressed(originalsize, compressedsize);
                FQ_T *startaddr = NULL;
                if (isCompressed)
                {
                	startaddr = (FQ_T *)malloc(originalsize * sizeof(FQ_T));
                	if (startaddr == NULL)
                	{
                    		printf("\nERROR: Memory allocation error!");
                    		abort();
                	}
                	MPI_Bcast(startaddr, originalsize, fq_tp_type, root_rank, row_comm);
                }
#endif
                // Please Note: fq_64 buffer has been alloceted in the children class.
                // In case of CUDA, it has been allocated using cudaMalloc. In that case,
                // it would be pinned memory and therefore can not be modified using
                // normal C/C++ memory functions. Further info on the
                // (this)->bfsMemCpy() call below.
                // MPI_Bcast(fq_64, compressedsize, fq_tp_typeC, root_rank, row_comm);
                // compressed_fq is type: fq_tp_typeC

                MPI_Bcast(compressed_fq, compressedsize, fq_tp_typeC, root_rank, row_comm);

                uncompressedsize = static_cast<size_t>(originalsize);
                schema.decompress(compressed_fq, compressedsize, /*Out*/ &uncompressed_fq, /*In Out*/ uncompressedsize);

                static_cast<Derived *>(this)->bfsMemCpy(fq_64, uncompressed_fq, originalsize);

#if defined(_COMPRESSION) && defined(_COMPRESSIONVERIFY)
                assert(is_sorted(fq_64, fq_64 + originalsize));
#endif


#else
                int originalsize;
                MPI_Bcast(&originalsize, 1, MPI_INT, root_rank, row_comm);
                MPI_Bcast(fq_64, originalsize, fq_tp_type, root_rank, row_comm);
#endif

#ifdef INSTRUMENTED
                tstart = MPI_Wtime();
#endif

                static_cast<Derived *>(this)->setIncommingFQ(it->startvtx, it->size, fq_64, originalsize);

#ifdef INSTRUMENTED
                tend = MPI_Wtime();
                lqueue += tend - tstart;
#endif

#ifdef _COMPRESSION
                free(vectorizedsize);

                free(uncompressed_fq);
                free(compressed_fq);
#if defined(_COMPRESSIONVERIFY)
                if (isCompressed)
                {
                    free(startaddr);
                }
#endif

#if defined(_COMPRESSIONVERIFY)
                free(startaddr);
#endif

#endif
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
        SCOREP_USER_REGION_END(rowCommunication_handle)
#endif
        ++iter;
    }
}
#endif // GLOBALBFS_HH
