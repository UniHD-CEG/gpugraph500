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
    //T_C *temporal_recv_buff = NULL;
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
                bool isCompressed = false;
#endif

#ifdef INSTRUMENTED
                startTimeQueueProcessing = MPI_Wtime();
#endif

                get(offset + lowerId, upperId, send/*Out*/, psizeTo/*Out*/);

#if defined(_COMPRESSION) && defined(_COMPRESSIONVERIFY)
                assert(is_sorted(send, send + psizeTo));
#endif

#ifdef _COMPRESSION
                assert(is_sorted(send, send + psizeTo));
                schema.compress(send, psizeTo, &compressed_fq, compressedsize);
                isCompressed = schema.isCompressed(psizeTo, compressedsize);
//std::cout << "a1: origsize: " << psizeTo << " compsize: " << compressedsize << " isCompressed: " << isCompressed << std::endl;

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


/*
		if (isCompressed)
		{

*/
			T_C *temporal_recv_buff = NULL;
			temporal_recv_buff = (T_C *)malloc (lowerId * sizeof(T_C));
        		/*err = posix_memalign((void **)&temporal_recv_buff, 16, lowerId * sizeof(T_C));
        		if (err) {
                		printf("memory error!\n");
                		abort();
        		}*/
                	MPI_Sendrecv(compressed_fq, compressedsize, typeC,
                             previousRank, it + 2,
                             temporal_recv_buff, lowerId, typeC,
			     //recv_buff, lowerId, type,
                             previousRank, it + 2,
                             comm, &status);
                	MPI_Get_count(&status, typeC, &psizeFrom);

/*
		}
		else
		{
                	MPI_Sendrecv(send, psizeTo, type,
                             previousRank, it + 2,
                             //temporal_recv_buff, lowerId, type,
			     recv_buff, lowerId, type,
                             previousRank, it + 2,
                             comm, &status);
                	MPI_Get_count(&status, type, &psizeFrom);
		}
*/
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
//std::cout << "a2: isCompressed: " << isCompressed << std::endl;
		//isCompressed = schema.isCompressed(originalsize, psizeFrom);
                uncompressedsize = originalsize;
		//if (isCompressed)
		//{
                schema.decompress(temporal_recv_buff, psizeFrom, &uncompressed_fq, uncompressedsize);
                isCompressed = schema.isCompressed(uncompressedsize, psizeFrom);
		//}
		//isCompressed =  schema.decompress(recv_buff, psizeFrom, &uncompressed_fq, uncompressedsize);
//std::cout << "a3: isCompressed: " << isCompressed << std::endl;
#endif

#if defined(_COMPRESSION) && defined(_COMPRESSIONVERIFY)
//std::cout << "a4: isCompressed: " << isCompressed << std::endl;
		if (isCompressed)
		{
                	assert(uncompressedsize == originalsize);
                	assert(is_sorted(uncompressed_fq, uncompressed_fq + originalsize));
		}
#endif

#ifdef _COMPRESSION
//std::cout << "a5: isCompressed: " << isCompressed << std::endl;
		//if (isCompressed)
		//{
                	reduce(offset, lowerId, uncompressed_fq, uncompressedsize);
		/*}
		else
		{
			//reduce(offset, lowerId, temporal_recv_buff, psizeFrom);
			reduce(offset, lowerId, recv_buff, psizeFrom);
		}*/
#else
                reduce(offset, lowerId, recv_buff, psizeFrom);
#endif

#ifdef _COMPRESSION
//std::cout << "a6: isCompressed: " << isCompressed << std::endl;
                //if (isCompressed)
                //{
                //    free(uncompressed_fq);
                //}

                //// free(compressed_fq);
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
		bool isCompressed = false;
#endif

#ifdef INSTRUMENTED
                startTimeQueueProcessing = MPI_Wtime();
#endif

                get(offset, lowerId, send/*Out*/, psizeTo/*Out*/);

#if defined(_COMPRESSION) && defined(_COMPRESSIONVERIFY)
                assert(is_sorted(send, send + psizeTo));
#endif

#ifdef _COMPRESSION
                assert(is_sorted(send, send + psizeTo));
                schema.compress(send, psizeTo, &compressed_fq, compressedsize);
                isCompressed = schema.isCompressed(psizeTo, compressedsize);
//std::cout << "b1: origsize: " << psizeTo << " compsize: " << compressedsize << " isCompressed: " << isCompressed << std::endl;
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
		//if (isCompressed)
		//{
			T_C *temporal_recv_buff = NULL;
			temporal_recv_buff = (T_C *)malloc (upperId * sizeof(T_C));
                        /*err = posix_memalign((void **)&temporal_recv_buff, 16, upperId * sizeof(T_C));
                        if (err) {
                                printf("memory error!\n");
                                abort();
                        }*/

                	MPI_Sendrecv(compressed_fq, compressedsize, typeC,
                             previousRank, it + 2,
                             temporal_recv_buff, upperId, typeC,
			     //recv_buff, upperId, type,
                             previousRank, it + 2,
                             comm, &status);
                	MPI_Get_count(&status, typeC, &psizeFrom);
/*
   std::cout << "-- b1 sent buffer32-rcv -- (" << psizeFrom <<")" << std::endl;
   for (int i =0; i< psizeFrom; ++i)
   {
        std::cout << temporal_recv_buff[i] << ", ";
   }
   std::cout << "-- end sent buffer32-rcv --" << std::endl;
*/
		//}
		/*else
		{
                	MPI_Sendrecv(send, psizeTo, type,
                             previousRank, it + 2,
                             //temporal_recv_buff, upperId, type,
			     recv_buff, upperId, type,
                             previousRank, it + 2,
                             comm, &status);
                	MPI_Get_count(&status, type, &psizeFrom);
		}*/
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
//std::cout << "b2: isCompressed: " << isCompressed << std::endl;
		//isCompressed = schema.isCompressed(originalsize, psizeFrom);
                uncompressedsize = originalsize;
		//if (isCompressed)
		//{
//std::cout << "b2: decompressing from this: ( " << psizeFrom <<  "/"<<uncompressedsize <<")"<< std::endl;
/*
   for (int i =0; i< psizeFrom; ++i)
   {
        std::cout << temporal_recv_buff[i] << ", ";
   }
*/
   //std::cout << "-- end --" << std::endl;

                schema.decompress(temporal_recv_buff, psizeFrom, &uncompressed_fq, uncompressedsize);
                isCompressed = schema.isCompressed(uncompressedsize, psizeFrom);
//std::cout << "b2: to this: ( " << uncompressedsize<<  ")"<< std::endl;
/*
   for (int i =0; i< uncompressedsize; ++i)
   {
        std::cout << uncompressed_fq[i] << ", ";
   }
*/
   //std::cout << "-- end --" << std::endl;

		//}
		//isCompressed = schema.decompress(recv_buff, psizeFrom, &uncompressed_fq, uncompressedsize);
#endif

#if defined(_COMPRESSION) && defined(_COMPRESSIONVERIFY)
//std::cout << "b3: isCompressed: " << isCompressed << std::endl;
        if (isCompressed)
        {
                    assert(uncompressedsize == originalsize);
                    assert(is_sorted(uncompressed_fq, uncompressed_fq + originalsize));
        }
#endif


#ifdef _COMPRESSION
//std::cout << "b4: isCompressed: " << isCompressed << std::endl;
		//if (isCompressed)
		//{
                	reduce(offset + lowerId, upperId, uncompressed_fq, uncompressedsize);
		/*}
		else
		{
			//reduce(offset + lowerId, upperId, temporal_recv_buff, psizeFrom);
			reduce(offset + lowerId, upperId, recv_buff, psizeFrom);
		}*/
#else
                reduce(offset + lowerId, upperId, recv_buff, psizeFrom);
#endif

#ifdef _COMPRESSION
//std::cout << "b5: isCompressed: " << isCompressed << std::endl;
                /*if (isCompressed)
                {
                    free(uncompressed_fq);
                }*/
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
    bool isCompressed = false;

    assert(is_sorted(send, send + psizeTo));
    schema.compress(send, psizeTo, &compressed_fq, compressedsize);
    isCompressed = schema.isCompressed(psizeTo, compressedsize);
//std::cout << "c1: isCompressed: " << isCompressed << std::endl;

    //MPI_Datatype MPI_3INT;

    int *composed_sizes = (int *)malloc(2 * communicatorSize * sizeof(int));
    int *composed_ints = (int *)malloc(2 * sizeof(int));

    composed_ints[0] = psizeTo;
    composed_ints[1] = compressedsize;
    //composed_ints[2] = (isCompressed) ? 1 : 0;

    //MPI_Type_contiguous(3, MPI_INT, &MPI_3INT);
    //MPI_Type_commit(&MPI_3INT);

    MPI_Allgather(composed_ints, 1, MPI_2INT, composed_sizes, 1, MPI_2INT, comm);


    const int totalsize = 2 * communicatorSize;

    for (int i = 0, j = 0; i < totalsize; ++i)
    {
        if (i % 2 == 0)
        {
            sizes[j] = composed_sizes[i];
            compressed_sizes[j] = composed_sizes[i + 1];
            ++j;
        }
    }

    free(composed_ints);
    free(composed_sizes);

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

    compressed_recv_buff = (T_C *)malloc(csize * sizeof(T_C));
    /*err = posix_memalign((void **)&compressed_recv_buff, 16, csize * sizeof(T_C));
    if (err) {
        printf("memory error!\n");
        abort();
    }*/
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
    //if(isCompressed)
    //{
        MPI_Allgatherv(compressed_fq, compressed_sizes[communicatorRank],
                   typeC, compressed_recv_buff, compressed_sizes,
                   compressed_disps, typeC, comm);
    /*}
    else
    {
        MPI_Allgatherv(send, compressed_sizes[communicatorRank],
                   type, compressed_recv_buff, compressed_sizes,
                   compressed_disps, type, comm);
    }*/
#else
    MPI_Allgatherv(send, sizes[communicatorRank],
                   type, recv_buff, sizes,
                   disps, type, comm);
#endif

#ifdef _COMPRESSION

#ifdef _COMPRESSIONVERIFY
    int total_uncompressedsize = 0;
#endif

    std:cout<< "enter 2 ..." << std::endl;
    // reensamble uncompressed chunks
    for (int i = 0; i < communicatorSize; ++i)
    {
        compressedsize = compressed_sizes[i];
        if (compressedsize != 0)
        {
            uncompressedsize = sizes[i];
            //isCompressed = schema.isCompressed(uncompressedsize, compressedsize);
            schema.decompress(&compressed_recv_buff[compressed_disps[i]], compressedsize, &uncompressed_fq, uncompressedsize);
            isCompressed = schema.isCompressed(uncompressedsize, compressedsize);

#ifdef _COMPRESSIONVERIFY
            if (isCompressed)
            {
            	assert(uncompressedsize == sizes[i]);
            	assert(std::is_sorted(uncompressed_fq, uncompressed_fq + uncompressedsize));
	    }
            total_uncompressedsize += uncompressedsize;
#endif

	    //if (isCompressed)
	    //{

            memcpy(&recv_buff[disps[i]], uncompressed_fq, uncompressedsize * sizeof(T));
            free(uncompressed_fq);
	    /*}
	    else
	    {
                memcpy(&recv_buff[disps[i]], &compressed_recv_buff[compressed_disps[i]], uncompressedsize * sizeof(T_C));
	    }*/
        }
    }
    std:cout<< "exit 2 ..." << std::endl;
#ifdef _COMPRESSIONVERIFY
   /*std::cout << "-- start reensabled buffer --" << std::endl;
   for (int i =0; i< total_uncompressedsize; ++i)
   {
	std::cout << recv_buff[i] << ", ";
   }
   std::cout << "-- end reensabled buffer --" << std::endl;*/
    assert(std::is_sorted(recv_buff, recv_buff + total_uncompressedsize));
#endif
#endif

    free(sizes);
    free(disps);

#ifdef _COMPRESSION
    // MPI_Type_free(&MPI_3INT);
    free(compressed_sizes);
    free(compressed_disps);
    //free(temporal_recv_buff);
    free(compressed_recv_buff);
#endif

}
#endif // VREDUCE_HPP
