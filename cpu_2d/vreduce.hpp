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

#ifdef _COMPRESSION
#include "compression/compression.hh"
#endif

using std::function;
using std::is_sorted;

#ifdef _COMPRESSION
template <typename T, typename T_C>
#else
template <typename T>
#endif
void vreduce(const function <void(T, long, T *, int)> &reduce,
             const function <void(T, long, T *& /*Out*/, int & /*Out*/)> &get,
#ifdef _COMPRESSION
             const Compression<T, T_C> &schema,
             MPI_Datatype typeC,
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

    int communicatorSize, communicatorRank, previousRank;

#ifdef _COMPRESSION
    size_t compressedsize, uncompressedsize;
    T *uncompressed_fq = NULL;
    T_C *compressed_fq = NULL;
    T_C *compressed_recv_buff = NULL;
    int err;
#endif

    //time mesurement
#ifdef INSTRUMENTED
    double startTimeQueueProcessing;
    double endTimeQueueProcessing;
#endif

    //step 1
    MPI_Comm_size(comm, &communicatorSize);
    MPI_Comm_rank(comm, &communicatorRank);
    const int intLdSize = ilogb(static_cast<double>(communicatorSize)); //integer log_2 of size
    const int power2intLdSize = 1 << intLdSize;
    const int residuum = communicatorSize - (1 << intLdSize);

    // auxiliar lambdas
    const function<int (int)> newRank = [&residuum](int oldr)
    {
        return (oldr < 2 * residuum) ? oldr / 2 : oldr - residuum;
    };
    const function<int (int)> oldRank = [&residuum](int newr)
    {
        return (newr <  residuum) ? newr * 2 : newr + residuum;
    };

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

#if defined(_COMPRESSION) && defined(_COMPRESSIONVERIFY)
                assert(is_sorted(send, send + psizeTo));
#endif

#ifdef _COMPRESSION
                schema.compress(send, psizeTo, &compressed_fq, compressedsize);

#endif

#if defined(_COMPRESSION) && defined(_COMPRESSIONDEBUG)
                schema.debugCompression(send, psizeTo);
#endif

#ifdef INSTRUMENTED
                endTimeQueueProcessing = MPI_Wtime();
                timeQueueProcessing += endTimeQueueProcessing - startTimeQueueProcessing;
#endif

                previousRank = oldRank((vrank + (1 << it)) & (power2intLdSize - 1));

#ifdef _COMPRESSION
                MPI_Sendrecv(&psizeTo, 1, MPI_INT,
                             previousRank, it + 2,
                             &originalsize, 1, MPI_INT,
                             previousRank, it + 2,
                             comm, MPI_STATUS_IGNORE);


                T_C *temporal_recv_buff = NULL;
                err = posix_memalign((void **)&temporal_recv_buff, 16, lowerId * sizeof(T_C));
                if (err) {
                        printf("memory error!\n");
                        abort();
                }

                MPI_Sendrecv(compressed_fq, compressedsize, typeC,
                         previousRank, it + 2,
                         temporal_recv_buff, lowerId, typeC,
                         previousRank, it + 2,
                         comm, &status);
                MPI_Get_count(&status, typeC, &psizeFrom);


#else
                MPI_Sendrecv(send, psizeTo, type,
                             previousRank, it + 2,
                             recv_buff, lowerId, type,
                             previousRank, it + 2,
                             comm, &status);
                MPI_Get_count(&status, type, &psizeFrom);
#endif

#ifdef INSTRUMENTED
                startTimeQueueProcessing = MPI_Wtime();
#endif

#if defined(_COMPRESSION) && defined(_COMPRESSIONVERIFY)
                assert(psizeFrom != MPI_UNDEFINED);
                assert(psizeFrom <= lowerId);
                assert(psizeFrom <= originalsize);
#endif

#ifdef _COMPRESSION
                uncompressedsize = originalsize;
                schema.decompress(temporal_recv_buff, psizeFrom, &uncompressed_fq, uncompressedsize);
#endif

#if defined(_COMPRESSION) && defined(_COMPRESSIONVERIFY)
                    assert(uncompressedsize == originalsize);
                    assert(is_sorted(uncompressed_fq, uncompressed_fq + originalsize));
#endif

#ifdef _COMPRESSION
                reduce(offset, lowerId, uncompressed_fq, uncompressedsize);
#else
                reduce(offset, lowerId, recv_buff, psizeFrom);
#endif

#ifdef _COMPRESSION
                free(uncompressed_fq);
                free(temporal_recv_buff);
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

#if defined(_COMPRESSION) && defined(_COMPRESSIONVERIFY)
                assert(is_sorted(send, send + psizeTo));
#endif

#ifdef _COMPRESSION
                schema.compress(send, psizeTo, &compressed_fq, compressedsize);
#endif

#if defined(_COMPRESSION) && defined(_COMPRESSIONDEBUG)
                schema.debugCompression(send, psizeTo);
#endif

#ifdef INSTRUMENTED
                endTimeQueueProcessing = MPI_Wtime();
                timeQueueProcessing += endTimeQueueProcessing - startTimeQueueProcessing;
#endif

                previousRank = oldRank((power2intLdSize + vrank - (1 << it)) & (power2intLdSize - 1));

#ifdef _COMPRESSION
                MPI_Sendrecv(&psizeTo, 1, MPI_INT,
                             previousRank, it + 2,
                             &originalsize, 1, MPI_INT,
                             previousRank, it + 2,
                             comm, MPI_STATUS_IGNORE);
            T_C *temporal_recv_buff = NULL;
            err = posix_memalign((void **)&temporal_recv_buff, 16, upperId * sizeof(T_C));
            if (err) {
                    printf("memory error!\n");
            }
                MPI_Sendrecv(compressed_fq, compressedsize, typeC,
                         previousRank, it + 2,
                         temporal_recv_buff, upperId, typeC,
                         previousRank, it + 2,
                         comm, &status);
                MPI_Get_count(&status, typeC, &psizeFrom);
#else
                MPI_Sendrecv(send, psizeTo, type,
                             previousRank, it + 2,
                             recv_buff, upperId, type,
                             previousRank, it + 2,
                             comm, &status);
                MPI_Get_count(&status, type, &psizeFrom);
#endif

#if defined(_COMPRESSION) && defined(_COMPRESSIONVERIFY)
                assert(psizeFrom != MPI_UNDEFINED);
#endif

#ifdef INSTRUMENTED
                startTimeQueueProcessing = MPI_Wtime();
#endif

#if defined(_COMPRESSION) && defined(_COMPRESSIONVERIFY)
                assert(psizeFrom <= lowerId);
                assert(psizeFrom <= originalsize);
#endif

#ifdef _COMPRESSION
                uncompressedsize = originalsize;
                schema.decompress(temporal_recv_buff, psizeFrom, &uncompressed_fq, uncompressedsize);
#endif

#if defined(_COMPRESSION) && defined(_COMPRESSIONVERIFY)
                    assert(uncompressedsize == originalsize);
                    assert(is_sorted(uncompressed_fq, uncompressed_fq + originalsize));
#endif


#ifdef _COMPRESSION
                    reduce(offset + lowerId, upperId, uncompressed_fq, uncompressedsize);
#else
                    reduce(offset + lowerId, upperId, recv_buff, psizeFrom);
#endif

#ifdef _COMPRESSION
                    free(uncompressed_fq);
                    free(temporal_recv_buff);
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

#if defined(_COMPRESSION) && defined(_COMPRESSIONVERIFY)
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

#ifdef _COMPRESSION
    int *compressed_sizes = (int *)malloc(communicatorSize * sizeof(int));
    int *compressed_disps = (int *)malloc(communicatorSize * sizeof(int));

    unsigned int lastReversedSliceIDs = 0;
    unsigned int lastTargetNode = oldRank(lastReversedSliceIDs);
    unsigned int reversedSliceIDs, targetNode;
    size_t csize = 0;

    schema.compress(send, psizeTo, &compressed_fq, compressedsize);

    int *composed_recv = (int *)malloc(2 * communicatorSize * sizeof(int));
    int *composed_send = (int *)malloc(2 * sizeof(int));

    composed_send[0] = psizeTo;
    composed_send[1] = compressedsize;

    MPI_Allgather(composed_send, 1, MPI_2INT, composed_recv, 1, MPI_2INT, comm);

    const int totalsize = 2 * communicatorSize;

    for (int i = 0, j = 0; i < totalsize; ++i)
    {
        if (i % 2 == 0)
        {
            sizes[j] = composed_recv[i];
            compressed_sizes[j] = composed_recv[i + 1];
            ++j;
        }
    }

    free(composed_send);
    free(composed_recv);

    disps[lastTargetNode] = 0;
    compressed_disps[lastTargetNode] = 0;
    for (unsigned int slice = 1; slice < power2intLdSize; ++slice)
    {
        reversedSliceIDs = reverse(slice, intLdSize);
        targetNode = oldRank(reversedSliceIDs);
        compressed_disps[targetNode] = compressed_disps[lastTargetNode] + compressed_sizes[lastTargetNode];
        disps[targetNode] = disps[lastTargetNode] + sizes[lastTargetNode];
        lastTargetNode = targetNode;
    }
    int index;
    for (unsigned int node = 0; node < residuum; ++node)
    {
        index = 2 * node + 1;
        disps[index] = 0;
        compressed_disps[index] = 0;
    }
    csize = compressed_disps[lastTargetNode] + compressed_sizes[lastTargetNode];
    rsize = disps[lastTargetNode] + sizes[lastTargetNode];

    err = posix_memalign((void **)&compressed_recv_buff, 16, csize * sizeof(T_C));

#else
    // Transmission of the subslice sizes
    MPI_Allgather(&psizeTo, 1, MPI_INT, sizes, 1, MPI_INT, comm);

    unsigned int lastReversedSliceIDs = 0;
    unsigned int lastTargetNode = oldRank(lastReversedSliceIDs);
    unsigned int reversedSliceIDs, targetNode;

    int disps_lastTargetNode = 0;
    disps[lastTargetNode] = 0;
    for (unsigned int slice = 1; slice < power2intLdSize; ++slice)
    {
        reversedSliceIDs = reverse(slice, intLdSize);
        targetNode = oldRank(reversedSliceIDs);
        disps[targetNode] = disps_lastTargetNode + sizes[lastTargetNode];
        lastTargetNode = targetNode;
        disps_lastTargetNode = disps[lastTargetNode];
    }

    //nodes without a partial result
    for (unsigned int node = 0; node < residuum; ++node)
    {
        disps[2 * node + 1] = 0;
    }
    rsize = disps_lastTargetNode + sizes[lastTargetNode];
#endif

#ifdef _COMPRESSION


    MPI_Allgatherv(compressed_fq, compressed_sizes[communicatorRank],
                   typeC, compressed_recv_buff, compressed_sizes,
                   compressed_disps, typeC, comm);

#else
    MPI_Allgatherv(send, sizes[communicatorRank],
                   type, recv_buff, sizes,
                   disps, type, comm);
#endif

#ifdef _COMPRESSION
    // reensamble uncompressed chunks
    for (int i = 0; i < communicatorSize; ++i)
    {
        compressedsize = compressed_sizes[i];
        uncompressedsize = sizes[i];
        if (compressedsize != 0)
        {
            schema.decompress(&compressed_recv_buff[compressed_disps[i]], compressedsize, &uncompressed_fq, uncompressedsize);
            memcpy(&recv_buff[disps[i]], uncompressed_fq, uncompressedsize * sizeof(T));
            free(uncompressed_fq);
        }
    }
#endif

    free(sizes);
    free(disps);

#ifdef _COMPRESSION
    free(compressed_sizes);
    free(compressed_disps);
    free(compressed_recv_buff);
#endif

}
#endif // VREDUCE_HPP
