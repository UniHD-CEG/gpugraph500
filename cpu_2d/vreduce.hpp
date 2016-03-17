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
#include "config.h"

#ifdef _COMPRESSION
#include "compression/compression.hh"
#endif

using std::function;
using std::is_sorted;

#ifndef ALIGNMENT
#if HAVE_AVX
#define ALIGNMENT 32UL
#else
#define ALIGNMENT 16UL
#endif
#endif


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
             T * recv_buff, /* Out */
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

    int err1, err2;

//time mesurement
#ifdef INSTRUMENTED
    double startTimeQueueProcessing;
    double endTimeQueueProcessing;
#endif

    //step 1
    MPI_Comm_size(comm, &communicatorSize);
    MPI_Comm_rank(comm, &communicatorRank);
    const int32_t intLdSize = ilogb(static_cast<float>(communicatorSize)); //integer log_2 of size
    const int32_t power2intLdSize = 1 << intLdSize;
    const int32_t residuum = communicatorSize - (1 << intLdSize);
    const int32_t twoTimesResiduum = residuum << 1;

    // auxiliar lambdas
    const function<int32_t (int32_t)> newRank = [&residuum](int32_t oldr)
    {
        return (oldr < residuum << 1) ? oldr >> 1 : oldr - residuum;
    };
    const function<int32_t (int32_t)> oldRank = [&residuum](int32_t newr)
    {
        return (newr <  residuum) ? newr << 1 : newr + residuum;
    };

    //step 2
    if (communicatorRank < twoTimesResiduum)
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
    int32_t psizeTo;
    T *send;

    // step 3
    if ((((communicatorRank & 1) == 0)
         && (communicatorRank < twoTimesResiduum)) || (communicatorRank >= twoTimesResiduum))
    {

        int32_t vrank;
        int32_t currentSliceSize;
        int32_t offset;
        int32_t lowerId;
        int32_t upperId;

        vrank  = newRank(communicatorRank);
        currentSliceSize  = ssize;
        offset = 0;

        for (int32_t it = 0; it < intLdSize; ++it)
        {
            lowerId = currentSliceSize >> 1;
            upperId = currentSliceSize - lowerId;

            if (((vrank >> it) & 1) == 0)   // even
            {
                MPI_Status status;
                int32_t psizeFrom;
#ifdef _COMPRESSION
                int32_t originalsize;
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


                T_C * restrict temporal_recv_buff = NULL;
                err = posix_memalign((void **)&temporal_recv_buff, ALIGNMENT, lowerId * sizeof(T_C));
                if (err) {
                        throw "memory error.";
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
                int32_t psizeFrom;
#ifdef _COMPRESSION
                int32_t originalsize;
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

                T_C * restrict temporal_recv_buff = NULL;
                err = posix_memalign((void **)&temporal_recv_buff, ALIGNMENT, upperId * sizeof(T_C));
                if (err) {
                        throw "memory error.";
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
    int32_t * restrict sizes;
    int32_t * restrict disps;

    err1 = posix_memalign((void **)&sizes, ALIGNMENT, communicatorSize * sizeof(int32_t));
    err2 = posix_memalign((void **)&disps, ALIGNMENT, communicatorSize * sizeof(int32_t));
    if (err1 || err2) {
        throw "Memory error.";
    }

#ifdef _COMPRESSION
    int32_t * restrict compressed_sizes;
    int32_t * restrict compressed_disps;

    err1 = posix_memalign((void **)&compressed_sizes, ALIGNMENT, communicatorSize * sizeof(int32_t));
    err2 = posix_memalign((void **)&compressed_disps, ALIGNMENT, communicatorSize * sizeof(int32_t));
    if (err1 || err2) {
        throw "Memory error.";
    }

    int32_t lastReversedSliceIDs = 0;
    int32_t lastTargetNode = oldRank(lastReversedSliceIDs);
    int32_t targetNode;
    uint32_t reversedSliceIDs;
    size_t csize = 0U;

    schema.compress(send, psizeTo, &compressed_fq, compressedsize);

    int32_t * restrict composed_recv;
    int32_t * restrict composed_send;

    err1 = posix_memalign((void **)&composed_recv, ALIGNMENT, 2U * communicatorSize * sizeof(int32_t));
    err2 = posix_memalign((void **)&composed_send, ALIGNMENT, 2U * sizeof(int32_t));

    composed_send[0U] = psizeTo;
    composed_send[1U] = compressedsize;

    MPI_Allgather(composed_send, 1, MPI_2INT, composed_recv, 1, MPI_2INT, comm);

    const int32_t totalsize = communicatorSize << 1;
    for (int32_t i = 0, j = 0; i < totalsize; ++i)
    {
        if (i % 2 == 0)
        {
            sizes[j] = composed_recv[i];
            compressed_sizes[j] = composed_recv[i + 1U];
            ++j;
        }
    }

    free(composed_send);
    free(composed_recv);

    disps[lastTargetNode] = 0;
    compressed_disps[lastTargetNode] = 0;

    for (int32_t slice = 1; slice < power2intLdSize; ++slice)
    {
        reversedSliceIDs = reverse(slice, intLdSize);
        targetNode = oldRank(reversedSliceIDs);
        compressed_disps[targetNode] = compressed_disps[lastTargetNode] + compressed_sizes[lastTargetNode];

        disps[targetNode] = disps[lastTargetNode] + sizes[lastTargetNode];
        lastTargetNode = targetNode;
    }

    for (int32_t node = 0; node < residuum; ++node)
    {
        const int32_t index = 2 * node + 1;
        disps[index] = 0;
        compressed_disps[index] = 0;
    }
    csize = compressed_disps[lastTargetNode] + compressed_sizes[lastTargetNode];
    rsize = disps[lastTargetNode] + sizes[lastTargetNode];

    err = posix_memalign((void **)&compressed_recv_buff, ALIGNMENT, csize * sizeof(T_C));
    if (err) {
        throw "Memory error.";
    }

#else
    // Transmission of the subslice sizes
    MPI_Allgather(&psizeTo, 1, MPI_INT, sizes, 1, MPI_INT, comm);

    int32_t lastReversedSliceIDs = 0;
    int32_t lastTargetNode = oldRank(lastReversedSliceIDs);
    int32_t targetNode;
    uint32_t reversedSliceIDs;

    int32_t disps_lastTargetNode = 0;
    disps[lastTargetNode] = 0;
    for (int32_t slice = 1; slice < power2intLdSize; ++slice)
    {
        reversedSliceIDs = reverse(slice, intLdSize);
        targetNode = oldRank(reversedSliceIDs);
        disps[targetNode] = disps_lastTargetNode + sizes[lastTargetNode];
        lastTargetNode = targetNode;
        disps_lastTargetNode = disps[lastTargetNode];
    }

    //nodes without a partial result
    for (int32_t node = 0; node < residuum; ++node)
    {
        const int32_t index = (node * 2) + 1;
        disps[index] = 0;
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
    for (int32_t i = 0; i < communicatorSize; ++i)
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
