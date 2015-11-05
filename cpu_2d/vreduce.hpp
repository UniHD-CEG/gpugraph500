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
    //T *compressed_fq=NULL, *uncompressed_fq=NULL, *compressed_recv_buff=NULL, *uncompressed_recv_buff=NULL;
    T *compressed_fq, *uncompressed_fq, *compressed_recv_buff, *uncompressed_recv_buff;
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
                MPI_Sendrecv(&psizeTo, 1, MPI_INT,
                             previousRank, it + 2,
                             &originalsize, 1, MPI_INT,
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
                MPI_Sendrecv(&psizeTo, 1, MPI_INT,
                             previousRank, it + 2,
                             &originalsize, 1, MPI_INT,
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
    int *compressed_disps = (int *)malloc(communicatorSize * sizeof(int));
    unsigned int lastReversedSliceIDs = 0;
    unsigned int lastTargetNode = oldRank(lastReversedSliceIDs);
    unsigned int reversedSliceIDs, targetNode;
    size_t csize=0;

    compress(send, psizeTo, &compressed_fq, compressedsize);
    MPI_Allgather(&compressedsize, 1, MPI_INT, compressed_sizes, 1, MPI_INT, comm);

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

    compressed_recv_buff = (T *)malloc(csize*sizeof(T));
/*
    std::cout << "csize1: " << csize << std::endl;

    int totlen = 0;
    for (int i=0; i<communicatorSize; ++i) {
        totlen += compressed_sizes[i];
    }
    std::cout << "csize2: " << totlen << std::endl;
*/
#else
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
/*
std::cout << std::endl << "*** START original buffer. for rank: " << communicatorRank << std::endl;
for (int i=0; i<sizes[communicatorRank]; ++i) {
    std::cout << send[i] << " ";
}
std::cout << std::endl << "*** END original buffer. for rank: " << communicatorRank << std::endl;
std::cout << std::endl << "*** START compressed buffer. for rank: " << communicatorRank << std::endl;
for (int i=0; i<compressed_sizes[communicatorRank]; ++i) {
    std::cout <<  compressed_fq[i] << " ";
}
std::cout << std::endl << "*** END compressed buffer. for rank: " << communicatorRank << std::endl;
*/
std::cout << std::endl << "*** START sizes. totalsize: "<< rsize << " rank: " << communicatorRank << std::endl;
for (int i=0; i<communicatorSize; ++i) {
    std::cout << sizes[i] << " ";
}
std::cout << std::endl << "*** END sizes. rank: " << communicatorRank << std::endl;
std::cout << std::endl << "*** START disps. rank: " << communicatorRank << std::endl;
for (int i=0; i<communicatorSize; ++i) {
    std::cout << disps[i] << " ";
}
std::cout << std::endl << "*** END DIPS. rank: " << communicatorRank << std::endl;
std::cout << std::endl << "*** START compressed_sizes. totalsize: "<< csize << "  rank: " << communicatorRank << std::endl;
for (int i=0; i<communicatorSize; ++i) {
    std::cout << compressed_sizes[i] << " ";
}
std::cout << std::endl << "*** END compressed_sizes. rank: " << communicatorRank << std::endl;
std::cout << std::endl << "*** START compressed_disps. rank: " << communicatorRank << std::endl;
for (int i=0; i<communicatorSize; ++i) {
    std::cout << compressed_disps[i] << " ";
}
std::cout << std::endl << "*** END compressed_DIPS. rank: " << communicatorRank << std::endl;


compressedsize = compressed_sizes[communicatorRank];
uncompressedsize = sizes[communicatorRank];
decompress(compressed_fq, compressedsize, &uncompressed_fq , uncompressedsize);

assert(uncompressedsize == sizes[communicatorRank]);
assert(std::is_sorted(uncompressed_fq, uncompressed_fq + uncompressedsize));
assert(memcmp(send, uncompressed_fq, uncompressedsize * sizeof(T)) == 0);
std::cout << "****** PASSED ASSERTS" << std::endl;

#endif

#ifdef _COMPRESSION
/*
std::cout << std::endl << "*** Rsize: " <<rsize << std::endl;
std::cout << std::endl << "*** Csize: " <<csize << std::endl;

std::cout << std::endl << "*** size: " << sizes[communicatorRank] << std::endl;
std::cout << std::endl << "*** compressed_size: " << compressed_sizes[communicatorRank] << std::endl;
*/
    MPI_Allgatherv(send, sizes[communicatorRank],
                   type, recv_buff, sizes,
                   disps, type, comm);

    MPI_Allgatherv(compressed_fq, compressed_sizes[communicatorRank],
                   type, compressed_recv_buff, compressed_sizes,
                   compressed_disps, type, comm);
#else
    MPI_Allgatherv(send, sizes[communicatorRank],
                   type, recv_buff, sizes,
                   disps, type, comm);
#endif

#ifdef _COMPRESSION
/*
std::cout << "*** 2sizes. rank " << communicatorRank << std::endl;
for (int i=0; i<communicatorSize; ++i) {
    std::cout << sizes[i] << " ";
}
std::cout << "*** 2END sizes. rank " << communicatorRank << std::endl;

std::cout << "*** 2compressed_sizes. rank " << communicatorRank << std::endl;
for (int i=0; i<communicatorSize; ++i) {
    std::cout << compressed_sizes[i] << " ";
}
std::cout << "*** 2END compressed_sizes. rank " << communicatorRank << std::endl;
*/
    compressedsize = compressed_sizes[communicatorRank];

// std::cout << "*** START compressed buffer. " << "compressed size: " << compressedsize << "rank: " << communicatorRank << std::endl;
// for (int i=0; i<compressedsize; ++i) {
//     std::cout << compressed_recv_buff[compressed_disps[i]] << " ";
// }
// std::cout << "*** END compressed buffer. rank: " << communicatorRank << std::endl;

    // uncompressedsize = sizes[communicatorRank];
    // decompress(&compressed_recv_buff[compressed_disps[communicatorRank]], compressedsize, &uncompressed_recv_buff, uncompressedsize);

    // assert(uncompressedsize == sizes[communicatorRank]);
    // assert(std::is_sorted(uncompressed_recv_buff, uncompressed_recv_buff + uncompressedsize));
    // std::cout << "****** PASSED ASSERTS" << std::endl;
#endif

#ifdef _COMPRESSION
/*std::cout << "decompressed. for rank " << communicatorRank << std::endl;
for (int i=0; i<uncompressedsize; ++i) {
    std::cout << uncompressed_recv_buff[i] << " ";
}
std::cout << "end of decompressed. for rank " << communicatorRank << std::endl;
*/
#endif

    free(sizes);
    free(disps);

#ifdef _COMPRESSION
    // if (isCompressed(uncompressedsize, compressedsize))
    // {
    //     free(uncompressed_recv_buff);
    // }
    free(compressed_sizes);
    free(compressed_disps);
    //free(compressed_recv_buff);
#endif

}

#endif // VREDUCE_HPP
