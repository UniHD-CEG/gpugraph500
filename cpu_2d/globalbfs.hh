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
#include "config.h"

#if _OPENMP
#include <parallel/algorithm>
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

#ifndef ALIGNMENT
#if HAVE_AVX
#define ALIGNMENT 32UL
#else
#define ALIGNMENT 16UL
#endif
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
    int err;
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
    int64_t mtypesize = sizeof(MType) << 3; // * 2^3
    int64_t local_column = store.getLocalColumnID(), local_row = store.getLocalRowID();
    // Split communicator into row and column communicator
    // Split by row, rank by column
    MPI_Comm_split(MPI_COMM_WORLD, local_row, local_column, &row_comm);
    // Split by column, rank by row
    MPI_Comm_split(MPI_COMM_WORLD, local_column, local_row, &col_comm);
    fold_fq_props = store.getFoldProperties();
    mask_size = (store.getLocColLength() / mtypesize) + ((store.getLocColLength() % mtypesize > 0) ? 1LL : 0LL);
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
    int32_t communicatorSize, communicatorRank;
    const int32_t psize = static_cast<int32_t>(mask_size); // may result in overflow 64-bit -> 32-bit
    const int32_t mtypesize = sizeof(MType) << 3; // * 8
    //step 1
    MPI_Comm_size(col_comm, &communicatorSize);
    MPI_Comm_rank(col_comm, &communicatorRank);
    const int32_t intLdSize = ilogbf(static_cast<float>(communicatorSize)); //integer log_2 of size
    const int32_t power2intLdSize = 1 << intLdSize; // 2^n
    const int32_t residuum = communicatorSize - power2intLdSize;
    const int32_t twoTimesResiduum = residuum << 1;

    const function <int32_t(int32_t)> newRank = [&residuum](uint32_t oldr)
    {
        return (oldr < (residuum << 1)) ? (oldr >> 1) : oldr - residuum;
    };
    const function <int32_t(int32_t)> oldRank = [&residuum](uint32_t newr)
    {
        return (newr < residuum) ? (newr << 1) : newr + residuum;
    };

    //step 2
    if (communicatorRank < twoTimesResiduum)
    {
        if ((communicatorRank & 1) == 0)   // even
        {
            MPI_Sendrecv(owenmap, psize, bm_type, communicatorRank + 1, 0, tmpmap, psize, bm_type, communicatorRank + 1, 0,
                         col_comm, &status);

            for (int32_t i = 0; i < psize; ++i)
            {
                tmpmap[i] &= ~owenmap[i];
                owenmap[i] |= tmpmap[i];
            }
            MPI_Recv(tmp, store.getLocColLength(), fq_tp_type, communicatorRank + 1, 1, col_comm, &status);
            // set recived elements where the bit maps indicate it
            int32_t p = 0;
            for (int32_t i = 0; i < psize; ++i)
            {
                MType tmpm = tmpmap[i];
                const int32_t size = i * mtypesize;
                while (tmpm != 0)
                {
                    int32_t last = ffsl(tmpm) - 1;
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

            for (int32_t i = 0; i < psize; ++i)
            {
                tmpmap[i] = ~tmpmap[i] & owenmap[i];
            }
            int32_t p = 0;
            for (int32_t i = 0; i < psize; ++i)
            {
                MType tmpm = tmpmap[i];
                const int32_t size = i * mtypesize;
                while (tmpm != 0)
                {
                    const int32_t last = ffsl(tmpm) - 1;
                    tmp[p] = owen[size + last];
                    ++p;
                    tmpm ^= (1 << last);
                }
            }
            MPI_Send(tmp, p, fq_tp_type, communicatorRank - 1, 1, col_comm);
        }
    }

    if ((((communicatorRank & 1) == 0) &&
        (communicatorRank < twoTimesResiduum)) || (communicatorRank >= twoTimesResiduum))
    {
        int32_t ssize, offset;
        ssize = psize;
        const int32_t vrank = newRank(communicatorRank);
        offset = 0;
        // intLdSize: ~2 to 4 iteractions (scale 22, 16 gpus)
        for (int32_t it = 0; it < intLdSize; ++it)
        {
            int32_t orankEven, orankOdd, iterator2, iterator3;
            const int32_t lowers = ssize >> 1; //lower slice size
            const int32_t uppers = ssize - lowers; //upper slice size
            int32_t size = lowers * mtypesize;
            orankEven = oldRank((vrank + (1 << it)) & (power2intLdSize - 1));
            orankOdd = oldRank((power2intLdSize + vrank - (1 << it)) & (power2intLdSize - 1));
            const int32_t twoTimesIterator = it << 1;
            iterator2 = twoTimesIterator + 2;
            iterator3 = twoTimesIterator + 3;

            if (((vrank >> it) & 1) == 0)  // even
            {
                //Transmission of the the bitmap
                MPI_Sendrecv(owenmap + offset, ssize, bm_type, orankEven, iterator2,
                             tmpmap + offset, ssize, bm_type, orankEven, iterator2,
                             col_comm, &status);

                // lowers: ~65k iteractions per MPI node (scale 22, 16 gpus)
#ifdef _OPENMP
                #pragma omp parallel for
#endif
                for (int32_t i = 0; i < lowers; ++i)
                {
                    const int32_t iOffset = i + offset;
                    tmpmap[iOffset] &= ~owenmap[iOffset];
                    owenmap[iOffset] |= tmpmap[iOffset];
                }

                // lowers: ~65k iteractions per MPI node (scale 22, 16 gpus)
#ifdef _OPENMP
                #pragma omp parallel for
#endif
                for (int32_t i = lowers; i < ssize; ++i)
                {
                    const int32_t iOffset = i + offset;
                    tmpmap[iOffset] = (~tmpmap[iOffset]) & owenmap[iOffset];
                }

                //Generation of foreign updates
                // uppers: ~65k iteractions per MPI node (scale 22, 16 gpus)
                int32_t p = 0;
                for (int32_t i = 0; i < uppers; ++i)
                {
                    const int32_t iOffset = i + offset;
                    const int32_t iOffsetLowers = iOffset + lowers;
                    const int32_t index = iOffsetLowers * mtypesize;
                    MType tmpm = tmpmap[iOffsetLowers];
                    while (tmpm != 0)
                    {
                        int32_t last = ffsl(tmpm) - 1;
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
                // lowers: ~65k iteractions per MPI node (scale 22, 16 gpus)
                for (int32_t i = 0; i < lowers; ++i)
                {
                    const int32_t iOffset = i + offset;
                    const int32_t index = iOffset * mtypesize;
                    MType tmpm = tmpmap[iOffset];
                    while (tmpm != 0)
                    {
                        int32_t last = ffsl(tmpm) - 1;
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

                // lowers: ~65k iteractions per MPI node (scale 22, 16 gpus)
#ifdef _OPENMP
                #pragma omp parallel for
#endif
                for (int32_t i = 0; i < lowers; ++i)
                {
                    const int32_t iOffset = i + offset;
                    tmpmap[iOffset] = (~tmpmap[iOffset]) & owenmap[iOffset];
                }

                // lowers: ~65k iteractions per MPI node (scale 22, 16 gpus)
#ifdef _OPENMP
                #pragma omp parallel for
#endif
                for (int32_t i = lowers; i < ssize; ++i)
                {
                    const int32_t iOffset = i + offset;
                    tmpmap[iOffset] &= ~owenmap[iOffset];
                    owenmap[iOffset] |= tmpmap[iOffset];
                }
                //Generation of foreign updates
                // inner p: ~50k iteractions per MPI node (scale 22, 16 gpus)
                int32_t p = 0;
                for (int32_t i = 0; i < lowers; ++i)
                {
                    const int32_t iOffset = i + offset;
                    const int32_t iOffsetMtype = iOffset * mtypesize;
                    MType tmpm = tmpmap[iOffset];
                    while (tmpm != 0)
                    {
                        const int32_t last = ffsl(tmpm) - 1;
                        tmp[p] = owen[iOffsetMtype + last];
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
                // inner p: ~50k iteractions per MPI node (scale 22, 16 gpus)
                p = 0;
                for (int32_t i = 0; i < uppers; ++i)
                {
                    const int32_t iOffset = offset + lowers + i;
                    const int32_t lindex = iOffset * mtypesize;
                    MType tmpm = tmpmap[iOffset];
                    while (tmpm != 0)
                    {
                        const int32_t last = ffsl(tmpm) - 1;
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

    //int *sizes = (int *)malloc(communicatorSize * sizeof(int));
    //int *disps = (int *)malloc(communicatorSize * sizeof(int));

    int * restrict sizes;
    int * restrict disps;
    const int err1 = posix_memalign((void **)&sizes, ALIGNMENT, communicatorSize * sizeof(int));
    const int err2 = posix_memalign((void **)&disps, ALIGNMENT, communicatorSize * sizeof(int));
    if (err1 || err2) {
        throw "Memory error.";
    }

    const int maskLengthRes = psize % (1 << intLdSize);
    uint32_t lastReversedSliceIDs = 0U;
    int32_t lastTargetNode = oldRank(lastReversedSliceIDs);

    sizes[lastTargetNode] = (psize >> intLdSize) * mtypesize;
    disps[lastTargetNode] = 0;

    for (int slice = 1; slice < power2intLdSize; ++slice)
    {
        const uint32_t reversedSliceIDs = reverse(slice, intLdSize);
        const int32_t targetNode = oldRank(reversedSliceIDs);
        sizes[targetNode] = ((psize >> intLdSize) + (((power2intLdSize - reversedSliceIDs - 1) < maskLengthRes) ? 1 : 0)) *
                            mtypesize;
        disps[targetNode] = disps[lastTargetNode] + sizes[lastTargetNode];
        lastTargetNode = targetNode;
    }
    sizes[lastTargetNode] = std::min(sizes[lastTargetNode],
                                     static_cast<int>(store.getLocColLength() - disps[lastTargetNode]));

    //nodes without a partial resulty
    for (int node = 0; node < residuum; ++node)
    {
        const int index = (node * 2) + 1;
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

#ifdef _DISABLED_CUDA_OPENMP
    #pragma omp parallel
    {
        #pragma omp for schedule (guided, OMP_CHUNK)
#endif


        for (int64_t i = 0L; i < mask_size; ++i)
        {
            MType tmp = 0;
            const int64_t iindex = i * mtypesize;
            const int64_t mask_word_end = std::min(mtypesize, store_col_length - iindex);
            for (int64_t j = 0L; j < mask_word_end; ++j)
            {
                const int64_t jindex = iindex + j;
                if (predecessor[jindex] != -1)
                {
                    tmp |= 1L << j;
                }
            }
            owenmask[i] = tmp;
        }

#ifdef _DISABLED_CUDA_OPENMP
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
    lqueue += MPI_Wtime() - tstart;
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
        lexp += MPI_Wtime() - tstart;
#endif

// 3) Test if anything is done
        int anynewnodes, anynewnodes_global;

#ifdef INSTRUMENTED
        tstart = MPI_Wtime();
#endif

        anynewnodes = static_cast<Derived *>(this)->istheresomethingnew();

#ifdef INSTRUMENTED
        lqueue += MPI_Wtime() - tstart;
#endif

        MPI_Allreduce(&anynewnodes, &anynewnodes_global, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

        if (!anynewnodes_global)
        {

#ifdef INSTRUMENTED
            tstart = MPI_Wtime();
#endif

            static_cast<Derived *>(this)->getBackPredecessor();

#ifdef INSTRUMENTED
            lqueue += MPI_Wtime() - tstart;
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
            predlistred = MPI_Wtime() - tstart;
#endif

            return; // There is nothing to do. Finish iteration.
        }

// 4) global expansion
#ifdef INSTRUMENTED
        comtstart = MPI_Wtime();
        tstart = comtstart;
#endif

#ifdef _SCOREP_USER_INSTRUMENTATION
        SCOREP_USER_REGION_BEGIN(columnCommunication_handle, "BFSRUN_region_columnCommunication",
                                 SCOREP_USER_REGION_TYPE_COMMON)
#endif

        static_cast<Derived *>(this)->getBackOutqueue();

#ifdef INSTRUMENTED
        lqueue +=  MPI_Wtime() - tstart;
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
        comtstart = comtend;
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
                lqueue += MPI_Wtime() - tstart;
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

                int * restrict vectorizedsize = NULL;
                err = posix_memalign((void **)&vectorizedsize, ALIGNMENT, 2 * sizeof(int));
                if (err) {
                    throw "Memory error.";
                }

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
                lqueue += MPI_Wtime() - tstart;
#endif

            }
            else
            {

#ifdef _COMPRESSION
                compressionType *compressed_fq = NULL;
                FQ_T *uncompressed_fq = NULL;
                int originalsize, compressedsize;
		int * restrict vectorizedsize = NULL;
                err = posix_memalign((void **)&vectorizedsize, ALIGNMENT, 2 * sizeof(int));
                if (err) {
                    throw "Memory error.";
                }

                MPI_Bcast(vectorizedsize, 1, MPI_2INT, root_rank, row_comm);
                originalsize = vectorizedsize[0];
                compressedsize = vectorizedsize[1];
                err = posix_memalign((void **)&compressed_fq, ALIGNMENT, compressedsize * sizeof(compressionType));
                if (err) {
                    throw "Memory error.";
                }

#if defined(_COMPRESSIONVERIFY)
        bool isCompressed = schema.isCompressed(originalsize, compressedsize);
                FQ_T *startaddr = NULL;
                if (isCompressed)
                {
                    err = posix_memalign((void **)&startaddr, ALIGNMENT, originalsize * sizeof(FQ_T));
                    if (err) {
                        throw "Memory error.";
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
                lqueue += MPI_Wtime() - tstart;
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
        rowcom += tend - comtstart;
#endif

#ifdef _SCOREP_USER_INSTRUMENTATION
        SCOREP_USER_REGION_END(rowCommunication_handle)
#endif
        ++iter;
    }
}
#endif // GLOBALBFS_HH
