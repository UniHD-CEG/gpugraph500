/*
 * Inspiered by
 * Rolf Rabenseifner, A new optimized MPI reduce and allreduce algorithm
 * https://fs.hlrs.de/projects/par/mpi//myreduce.html
 *
 * Adaption for set operations: Matthias Hauck, 2014
 *
 */

#ifndef VREDUCE_HPP
#define VREDUCE_HPP

#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <functional>
#include <sstream>
#include "bitlevelfunctions.h"

using std::function;
using std::is_sorted;

template<class T>
void vreduce(function<void(T, long, T *, int)> &reduce,
             function < void(T, long, T *& /*Out*/, int & /*Out*/) > &get,
#ifdef _COMPRESSION
             function<void(T *, const size_t &, T **, size_t &)> &compress,
             function < void(T *, const int, T ** /*Out*/, size_t & /*In-Out*/) > &decompress,
#ifdef _COMPRESSIONDEBUG
             function <void (T *, const int)> &debugCompression,
#endif
             const function <bool (const size_t, const size_t)> &isCompressed,
#endif
             T *recv_buff, /* Out */
             int &rsize, /* Out */ // size of the final result
             int ssize,  // size of the slice
             MPI_Datatype type,
             MPI_Comm comm
#ifdef INSTRUMENTED
             , double &timeQueueProcessing // time of work
#endif
            )
{

    int communicatorSize, communicatorRank, intLdSize , power2intLdSize, residuum, previousRank;

#ifdef _COMPRESSION
    size_t compressedsize, uncompressedsize;
    T *compressed_fq, *uncompressed_fq, *compressed_recv_buff;
#endif

    // auxiliar lambdas
    const function<int (int)> newRank = [&residuum](int oldr)
    {
        return (oldr < 2 * residuum) ? oldr / 2 : oldr - residuum;
    };
    const function<int (int)> oldRank = [&residuum](int newr)
    {
        return (newr <  residuum) ? newr * 2 : newr + residuum;
    };

    //time mesurement
#ifdef INSTRUMENTED
    double startTimeQueueProcessing;
    double endTimeQueueProcessing;
#endif

    //step 1
    MPI_Comm_size(comm, &communicatorSize);
    MPI_Comm_rank(comm, &communicatorRank);
    intLdSize = ilogb(static_cast<double>(communicatorSize)); //integer log_2 of size
    power2intLdSize = 1 << intLdSize;
    residuum = communicatorSize - (1 << intLdSize);

    //step 2
    if (communicatorRank < 2 * residuum)
    {
        if ((communicatorRank & 1) == 0)  // even
        {

            MPI_Status status;
            int psize_from;
            MPI_Recv(recv_buff, ssize, type, communicatorRank + 1, 1, comm, &status);
            MPI_Get_count(&status, type, &psize_from);

#ifdef INSTRUMENTED
            startTimeQueueProcessing = MPI_Wtime();
#endif

            reduce(0, ssize, recv_buff, psize_from);

#ifdef INSTRUMENTED
            endTimeQueueProcessing = MPI_Wtime();
            timeQueueProcessing += (endTimeQueueProcessing - startTimeQueueProcessing);
#endif
        }
        else     // odd
        {
            int psize_to;
            T *send;

#ifdef INSTRUMENTED
            startTimeQueueProcessing = MPI_Wtime();
#endif

            get(0, ssize, send/*Out*/, psize_to/*Out*/);

#ifdef INSTRUMENTED
            endTimeQueueProcessing = MPI_Wtime();
            timeQueueProcessing += endTimeQueueProcessing - startTimeQueueProcessing;
#endif

            MPI_Send(send, psize_to, type, communicatorRank - 1, 1, comm);
        }
    }

    /**
     *
     *
     *
     *
     *
     *
     *
     *
     *
     */
    int psizeTo;
    T *send;

    // step 3
    if ((((communicatorRank & 1) == 0)
         && (communicatorRank < 2 * residuum)) || (communicatorRank >= 2 * residuum))
    {

        int vrank;
        int currentSliceSize;
        int offset;
        int lowerId;
        int upperId;

        vrank  = newRank(communicatorRank);
        currentSliceSize  = ssize;
        offset = 0;

        for (int it = 0; it < intLdSize; ++it)
        {

            lowerId = currentSliceSize / 2;
            upperId = currentSliceSize - lowerId;

            if (((vrank >> it) & 1) == 0)   // even
            {
                MPI_Status status;
                int psizeFrom;
#ifdef _COMPRESSION
                int originalsize;
#endif

#ifdef INSTRUMENTED
                startTimeQueueProcessing = MPI_Wtime();
#endif

                get(offset + lowerId, upperId, send/*Out*/, psizeTo/*Out*/);

#ifdef _COMPRESSIONVERIFY
                assert(is_sorted(send, send + psizeTo));
#endif

#ifdef _COMPRESSION
                compress(send, psizeTo, &compressed_fq, compressedsize);
#endif

#ifdef _COMPRESSIONDEBUG
                debugCompression(send, psizeTo);
#endif

#ifdef INSTRUMENTED
                endTimeQueueProcessing = MPI_Wtime();
                timeQueueProcessing += endTimeQueueProcessing - startTimeQueueProcessing;
#endif

                previousRank = oldRank((vrank + (1 << it)) & (power2intLdSize - 1));

#ifdef _COMPRESSION
                MPI_Sendrecv(&psizeTo, 1, MPI_LONG,
                             previousRank, it + 2,
                             &originalsize, 1, MPI_LONG,
                             previousRank, it + 2,
                             comm, MPI_STATUS_IGNORE);
                MPI_Sendrecv(compressed_fq, compressedsize, type,
                             previousRank, it + 2,
                             recv_buff, lowerId, type,
                             previousRank, it + 2,
                             comm, &status);
                MPI_Get_count(&status, type, &psizeFrom);
#else
                MPI_Sendrecv(send, psizeTo, type,
                             previousRank, it + 2,
                             recv_buff, lowerId, type,
                             previousRank, it + 2,
                             comm, &status);
                MPI_Get_count(&status, type, &psizeFrom);
#endif



                assert(psizeFrom != MPI_UNDEFINED);

#ifdef INSTRUMENTED
                startTimeQueueProcessing = MPI_Wtime();
#endif

#ifdef _COMPRESSIONVERIFY
                assert(psizeFrom <= lowerId);
                assert(psizeFrom <= originalsize);
#endif

#ifdef _COMPRESSION
                uncompressedsize = originalsize;
                decompress(recv_buff, psizeFrom, &uncompressed_fq, uncompressedsize);
#endif

#ifdef _COMPRESSIONVERIFY
                assert(uncompressedsize == originalsize);
                assert(is_sorted(uncompressed_fq, uncompressed_fq + originalsize));
#endif

#ifdef _COMPRESSION
                reduce(offset, lowerId, uncompressed_fq, uncompressedsize);
#else
                reduce(offset, lowerId, recv_buff, psizeFrom);
#endif

#ifdef _COMPRESSION
                if (isCompressed(originalsize, psizeFrom))
                {

                    free(uncompressed_fq);
                }
#endif

#ifdef INSTRUMENTED
                endTimeQueueProcessing = MPI_Wtime();
                timeQueueProcessing += endTimeQueueProcessing - startTimeQueueProcessing;
#endif

                currentSliceSize = lowerId;
            }
            else     // odd
            {
                MPI_Status status;
                int psizeFrom;
#ifdef _COMPRESSION
                int originalsize;
#endif

#ifdef INSTRUMENTED
                startTimeQueueProcessing = MPI_Wtime();
#endif

                get(offset, lowerId, send/*Out*/, psizeTo/*Out*/);

#ifdef _COMPRESSIONVERIFY
                assert(is_sorted(send, send + psizeTo));
#endif

#ifdef _COMPRESSION
                compress(send, psizeTo, &compressed_fq, compressedsize);
#endif

#ifdef _COMPRESSIONDEBUG
                debugCompression(send, psizeTo);
#endif

#ifdef INSTRUMENTED
                endTimeQueueProcessing = MPI_Wtime();
                timeQueueProcessing += endTimeQueueProcessing - startTimeQueueProcessing;
#endif

                previousRank = oldRank((power2intLdSize + vrank - (1 << it)) & (power2intLdSize - 1));

#ifdef _COMPRESSION
                MPI_Sendrecv(&psizeTo, 1, MPI_LONG,
                             previousRank, it + 2,
                             &originalsize, 1, MPI_LONG,
                             previousRank, it + 2,
                             comm, MPI_STATUS_IGNORE);
                MPI_Sendrecv(compressed_fq, compressedsize, type,
                             previousRank, it + 2,
                             recv_buff, upperId, type,
                             previousRank, it + 2,
                             comm, &status);
                MPI_Get_count(&status, type, &psizeFrom);
#else
                MPI_Sendrecv(send, psizeTo, type,
                             previousRank, it + 2,
                             recv_buff, upperId, type,
                             previousRank, it + 2,
                             comm, &status);
                MPI_Get_count(&status, type, &psizeFrom);
#endif

                assert(psizeFrom != MPI_UNDEFINED);

#ifdef INSTRUMENTED
                startTimeQueueProcessing = MPI_Wtime();
#endif

#ifdef _COMPRESSIONVERIFY
                assert(psizeFrom <= lowerId);
                assert(psizeFrom <= originalsize);
#endif

#ifdef _COMPRESSION
                uncompressedsize = originalsize;
                decompress(recv_buff, psizeFrom, &uncompressed_fq, uncompressedsize);
#endif

#ifdef _COMPRESSIONVERIFY
                assert(uncompressedsize == originalsize);
                assert(is_sorted(uncompressed_fq, uncompressed_fq + originalsize));
#endif


#ifdef _COMPRESSION
                reduce(offset + lowerId, upperId, uncompressed_fq, uncompressedsize);
#else
                reduce(offset + lowerId, upperId, recv_buff, psizeFrom);
#endif

#ifdef _COMPRESSION
                if (isCompressed(originalsize, psizeFrom))
                {
                    free(uncompressed_fq);
                }
#endif

#ifdef INSTRUMENTED
                endTimeQueueProcessing = MPI_Wtime();
                timeQueueProcessing += endTimeQueueProcessing - startTimeQueueProcessing;
#endif

                offset += lowerId;
                currentSliceSize = upperId;
            }
        }

        // Data to send to the other nodes

#ifdef INSTRUMENTED
        startTimeQueueProcessing = MPI_Wtime();
#endif

        get(offset, currentSliceSize, send/*Out*/, psizeTo/*Out*/);

#ifdef _COMPRESSIONVERIFY
        assert(is_sorted(send, send + psizeTo));
#endif


#ifdef INSTRUMENTED
        endTimeQueueProcessing = MPI_Wtime();
        timeQueueProcessing += endTimeQueueProcessing - startTimeQueueProcessing;
#endif

    }
    else
    {
        psizeTo = 0;
        send = 0;
    }



    // Transmission of the final results
    int *sizes = (int *)malloc(communicatorSize * sizeof(int));
    int *disps = (int *)malloc(communicatorSize * sizeof(int));

    // Transmission of the subslice sizes
    MPI_Allgather(&psizeTo, 1, MPI_INT, sizes, 1, MPI_INT, comm);


#ifdef _COMPRESSION
    int *compressed_sizes = (int *)malloc(communicatorSize * sizeof(int));
    unsigned int lastReversedSliceIDs = 0;
    unsigned int lastTargetNode = oldRank(lastReversedSliceIDs);
    unsigned int reversedSliceIDs, targetNode;

    compress(send, psizeTo, &compressed_fq, compressedsize);
    MPI_Allgather(&compressedsize, 1, MPI_INT, compressed_sizes, 1, MPI_INT, comm);

    disps[lastTargetNode] = 0;
    for (unsigned int slice = 1; slice < power2intLdSize; ++slice)
    {
        reversedSliceIDs = reverse(slice, intLdSize);
        targetNode = oldRank(reversedSliceIDs);
        disps[targetNode] = disps[lastTargetNode] + compressed_sizes[lastTargetNode];
        lastTargetNode = targetNode;
    }
    for (unsigned int node = 0; node < residuum; ++node)
    {
        disps[2 * node + 1] = 0;
    }
    rsize = disps[lastTargetNode] + compressed_sizes[lastTargetNode];
#else
    unsigned int lastReversedSliceIDs = 0;
    unsigned int lastTargetNode = oldRank(lastReversedSliceIDs);
    unsigned int reversedSliceIDs, targetNode;

    disps[lastTargetNode] = 0;
    for (unsigned int slice = 1; slice < power2intLdSize; ++slice)
    {
        reversedSliceIDs = reverse(slice, intLdSize);
        targetNode = oldRank(reversedSliceIDs);
        disps[targetNode] = disps[lastTargetNode] + sizes[lastTargetNode];
        lastTargetNode = targetNode;
    }

    //nodes without a partial result
    for (unsigned int node = 0; node < residuum; ++node)
    {
        disps[2 * node + 1] = 0;
    }
    rsize = disps[lastTargetNode] + sizes[lastTargetNode];
#endif

std::cout << "original. for rank " << communicatorRank << std::endl;
for (int i=0; i<sizes[communicatorRank]; ++i) {
    std::cout << send[i] << " ";
}
std::cout << "end of original. for rank " << communicatorRank << std::endl;

#ifdef _COMPRESSION
    MPI_Allgatherv(send, sizes[communicatorRank],
                   type, recv_buff, sizes,
                   disps, type, comm);
    MPI_Allgatherv(send, compressed_sizes[communicatorRank],
                   type, compressed_recv_buff, compressed_sizes,
                   disps, type, comm);
#else
    MPI_Allgatherv(send, sizes[communicatorRank],
                   type, recv_buff, sizes,
                   disps, type, comm);
#endif

#ifdef _COMPRESSION
    uncompressedsize = sizes[communicatorRank];
    originalsize = compressed_sizes[communicatorRank];
    decompress(&compressed_recv_buff[disps[communicatorRank]], originalsize, &uncompressed_fq, uncompressedsize);
#endif

std::cout << "decompressed. for rank " << communicatorRank << std::endl;
for (int i=compressed_recv_buff[disps[communicatorRank]]; i<sizes[communicatorRank]; ++i) {
    std::cout << send[i] << " ";
}
std::cout << "end of decompressed. for rank " << communicatorRank << std::endl;

    free(sizes);
    free(disps);

#ifdef _COMPRESSION
    if (isCompressed(uncompressedsize, originalsize))
    {
        free(uncompressed_fq);
    }
    free(compressed_sizes);
#endif

}

#endif // VREDUCE_HPP
