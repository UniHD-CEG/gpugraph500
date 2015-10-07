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

template<class T>
void vreduce(function<void(T, long, T *, int)> &reduce,
             function<void(T, long, T *&, int &)> &get,
#ifdef _COMPRESSION
             function<void(T *, const size_t &, T **, size_t &)> &compress,
             function <void(T *, const int,/*Out*/T **, /*InOut*/size_t &)> &decompress,
             function <void (T *, const int)> &benchmarkCompression,
             const function <void (const T *, const T *, const size_t)> &verifyCompression,
             const function <bool (const size_t, const size_t)> &isCompressed,
#endif
             T *recv_buff,
             int &rsize, // size of the final result
             int ssize,  //size of the slice
             MPI_Datatype type,
             MPI_Comm comm
#ifdef INSTRUMENTED
             , double &timeQueueProcessing // time of work
#endif
            )
{

    int communicatorSize, communicatorRank, intLdSize , power2intLdSize, residuum;
    size_t compressedsize, uncompressedsize;

    T *compressed_fq, *uncompressed_fq;
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
            int *originalsize = (int *) malloc(sizeof(int));
            assert(originalsize != NULL);

            MPI_Status status; int psize_from;

            MPI_Recv(originalsize, 1, MPI_LONG, communicatorRank + 1, 1, comm, &status); // originalsize
            MPI_Recv(recv_buff, ssize, type, communicatorRank + 1, 1, comm, &status);
            MPI_Get_count(&status, type, &psize_from);
            decompress(recv_buff, psize_from, &uncompressed_fq, uncompressedsize);

            if (originalsize != NULL)
            {
                free(originalsize);
            }

#ifdef INSTRUMENTED
            startTimeQueueProcessing = MPI_Wtime();
#endif

            // reduce(0, ssize, recv_buff, psize_from);
            reduce(0, ssize, uncompressed_fq, uncompressedsize);

            if (isCompressed(originalsize, psize_from))
            {
                if (uncompressed_fq != NULL)
                {
                    free(uncompressed_fq);
                }
            }

#ifdef INSTRUMENTED
            endTimeQueueProcessing = MPI_Wtime();
            timeQueueProcessing += (endTimeQueueProcessing - startTimeQueueProcessing);
#endif

        }
        else     // odd
        {
            int psize_to;
            T *send;
            int *originalsize = (int *) malloc(sizeof(int));
            assert(originalsize != NULL);

#ifdef INSTRUMENTED
            startTimeQueueProcessing = MPI_Wtime();
#endif

            get(0, ssize, send, psize_to);
            originalsize[0] = psize_to;

#ifdef INSTRUMENTED
            endTimeQueueProcessing = MPI_Wtime();
            timeQueueProcessing += endTimeQueueProcessing - startTimeQueueProcessing;
#endif

#ifdef _COMPRESSIONBENCHMARK
            benchmarkCompression(send, psize_to);
#endif

            MPI_Send(originalsize, 1, MPI_LONG, communicatorRank - 1, 1, comm); // originalsize
            compress(send, psize_to, &compressed_fq, compressedsize);
            MPI_Send(compressed_fq, compressedsize, type, communicatorRank - 1, 1, comm);
            //MPI_Send(send, psize_to, type, communicatorRank - 1, 1, comm);

            if (originalsize != NULL)
            {
                free(originalsize);
            }
        }
    }
    const function<int (int)> newRank = [&residuum](int oldr)
    {
        return (oldr < 2 * residuum) ? oldr / 2 : oldr - residuum;
    };
    const function<int (int)> oldRank = [&residuum](int newr)
    {
        return (newr <  residuum) ? newr * 2 : newr + residuum;
    };

    MPI_Status status;
    int psizeTo;
    int psizeFrom;
    T *send;
    int *originalsize = (int *) malloc(sizeof(int));

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

        // get(offset, csize, send, psize_to);


        for (int it = 0; it < intLdSize; ++it)
        {

            lowerId = currentSliceSize / 2;
            upperId = currentSliceSize - lowerId;

            if (((vrank >> it) & 1) == 0)   // even
            {

#ifdef INSTRUMENTED
                startTimeQueueProcessing = MPI_Wtime();
#endif

                get(offset + lowerId, upperId, send, psizeTo);
                compress(send, psizeTo, &compressed_fq, compressedsize);

#ifdef _COMPRESSIONBENCHMARK
                benchmarkCompression(send, psizeTo);
#endif

#ifdef INSTRUMENTED
                endTimeQueueProcessing = MPI_Wtime();
                timeQueueProcessing += endTimeQueueProcessing - startTimeQueueProcessing;
#endif

                originalsize[0] = psizeTo;
                MPI_Sendrecv(originalsize, 1, MPI_LONG,
                             oldRank((vrank + (1 << it)) & (power2intLdSize - 1)), it + 2,
                             originalsize, 1, MPI_LONG,
                             oldRank((vrank + (1 << it)) & (power2intLdSize - 1)), it + 2,
                             comm, &status);
                MPI_Sendrecv(compressed_fq, compressedsize, type,
                             oldRank((vrank + (1 << it)) & (power2intLdSize - 1)), it + 2,
                             recv_buff, lowerId, type,
                             oldRank((vrank + (1 << it)) & (power2intLdSize - 1)), it + 2,
                             comm, &status);
                MPI_Get_count(&status, type, &psizeFrom);
                //MPI_Sendrecv(send, psizeTo, type,
                //             oldRank((vrank + (1 << it)) & (power2intLdSize - 1)), it + 2,
                //             recv_buff, lowerId, type,
                //             oldRank((vrank + (1 << it)) & (power2intLdSize - 1)), it + 2,
                //             comm, &status);
                //MPI_Get_count(&status, type, &psizeFrom);

#ifdef INSTRUMENTED
                startTimeQueueProcessing = MPI_Wtime();
#endif

                decompress(recv_buff, psizeFrom, &uncompressed_fq, uncompressedsize);

                if (originalsize != NULL)
                {
                    free(originalsize);
                }

                //reduce(offset, lowerId, recv_buff, psizeFrom);
                reduce(offset, lowerId, uncompressed_fq, uncompressedsize);

                if (isCompressed(originalsize, psize_from))
                {
                    if (uncompressed_fq != NULL)
                    {
                        free(uncompressed_fq);
                    }
                    if (compressed_fq != NULL)
                    {
                        free(compressed_fq);
                    }
                }

#ifdef INSTRUMENTED
                endTimeQueueProcessing = MPI_Wtime();
                timeQueueProcessing += endTimeQueueProcessing - startTimeQueueProcessing;
#endif

                currentSliceSize = lowerId;
            }
            else     // odd
            {

#ifdef INSTRUMENTED
                startTimeQueueProcessing = MPI_Wtime();
#endif

                get(offset, lowerId, send, psizeTo);
                compress(send, psizeTo, &compressed_fq, compressedsize);

#ifdef _COMPRESSIONBENCHMARK
                benchmarkCompression(send, psizeTo);
#endif

#ifdef INSTRUMENTED
                endTimeQueueProcessing = MPI_Wtime();
                timeQueueProcessing += endTimeQueueProcessing - startTimeQueueProcessing;
#endif

                originalsize[0] = psizeTo;
                MPI_Sendrecv(originalsize, 1, MPI_LONG,
                             oldRank((power2intLdSize + vrank - (1 << it)) & (power2intLdSize - 1)), it + 2,
                             originalsize, 1, MPI_LONG,
                             oldRank((power2intLdSize + vrank - (1 << it)) & (power2intLdSize - 1)), it + 2,
                             comm, &status);
                MPI_Sendrecv(compressed_fq, compressedsize, type,
                             oldRank((power2intLdSize + vrank - (1 << it)) & (power2intLdSize - 1)), it + 2,
                             recv_buff, upperId, type,
                             oldRank((power2intLdSize + vrank - (1 << it)) & (power2intLdSize - 1)), it + 2,
                             comm, &status);
                MPI_Get_count(&status, type, &psizeFrom);
                //MPI_Sendrecv(send, psizeTo, type,
                //             oldRank((power2intLdSize + vrank - (1 << it)) & (power2intLdSize - 1)), it + 2,
                //             recv_buff, upperId, type,
                //             oldRank((power2intLdSize + vrank - (1 << it)) & (power2intLdSize - 1)), it + 2,
                //             comm, &status);
                //MPI_Get_count(&status, type, &psizeFrom);

#ifdef INSTRUMENTED
                startTimeQueueProcessing = MPI_Wtime();
#endif

                decompress(recv_buff, psizeFrom, &uncompressed_fq, uncompressedsize);

                if (originalsize != NULL)
                {
                    free(originalsize);
                }

                // reduce(offset + lowerId, upperId, recv_buff, psizeFrom);
                reduce(offset + lowerId, upperId, uncompressed_fq, uncompressedsize);

                if (isCompressed(originalsize, psize_from))
                {
                    if (uncompressed_fq != NULL)
                    {
                        free(uncompressed_fq);
                    }
                    if (compressed_fq != NULL)
                    {
                        free(compressed_fq);
                    }
                }

#ifdef INSTRUMENTED
                endTimeQueueProcessing = MPI_Wtime();
                timeQueueProcessing += endTimeQueueProcessing - startTimeQueueProcessing;
#endif

                offset += lowerId;
                currentSliceSize = upperId;
            }
        }

        // Datas to send to the other nodes

#ifdef INSTRUMENTED
        startTimeQueueProcessing = MPI_Wtime();
#endif

        get(offset, currentSliceSize, send, psizeTo);

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

    if (originalsize != NULL)
    {
        free(originalsize);
    }


    //
    // todo: add compression-decompression here.
    //
    //

    // Transmission of the final results
    int *sizes = (int *)malloc(communicatorSize * sizeof(int));
    assert(sizes != NULL);
    int *disps = (int *)malloc(communicatorSize * sizeof(int));
    assert(disps != NULL);

    // Transmission of the subslice sizes
    MPI_Allgather(&psizeTo, 1, MPI_INT, &sizes[0], 1, MPI_INT, comm);
    //Computation of displacements
    unsigned int lastReversedSliceIDs = 0;
    unsigned int lastTargetNode = oldRank(lastReversedSliceIDs);
    disps[lastTargetNode] = 0;

    for (unsigned int slice = 1; slice < power2intLdSize; ++slice)
    {
        unsigned int reversedSliceIDs = reverse(slice, intLdSize);
        unsigned int targetNode = oldRank(reversedSliceIDs);
        disps[targetNode] = disps[lastTargetNode] + sizes[lastTargetNode];
        lastTargetNode = targetNode;
    }

    //nodes without a partial result
    for (unsigned int node = 0; node < residuum; ++node)
    {
        disps[2 * node + 1] = 0;
    }

    rsize = disps[lastTargetNode] + sizes[lastTargetNode];

    MPI_Allgatherv(send, sizes[communicatorRank],
                   type, recv_buff, &sizes[0],
                   &disps[0], type, comm);

    free(sizes);
    free(disps);

}

#endif // VREDUCE_HPP
