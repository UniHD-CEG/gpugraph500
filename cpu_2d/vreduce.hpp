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
                try {
		schema.compress(send, psizeTo, &compressed_fq, compressedsize);
		} catch (...) {
			std::cout << "-----> exception 1";
			exit(1);
		}
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
			//temporal_recv_buff = (T_C *)malloc (lowerId * sizeof(T_C));
    		err = posix_memalign((void **)&temporal_recv_buff, 16, lowerId * sizeof(T_C));
    		if (err) {
            		printf("memory error!\n");
            		abort();
    		}

                	MPI_Sendrecv(compressed_fq, compressedsize, typeC,
                             previousRank, it + 2,
                             temporal_recv_buff, lowerId, typeC,
			     //recv_buff, lowerId, type,
                             previousRank, it + 2,
                             comm, &status);
                	MPI_Get_count(&status, typeC, &psizeFrom);

		/*debug*/
                    /*
                MPI_Sendrecv(send, psizeTo, type,
                             previousRank, it + 2,
                             recv_buff, lowerId, type,
                             previousRank, it + 2,
                             comm, &status);
                MPI_Get_count(&status, type, &psizeFrom);
                */

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
                //std::cout<< "enter 4 ..." << std::endl;
		try {
                schema.decompress(temporal_recv_buff, psizeFrom, &uncompressed_fq, uncompressedsize);
		 /*if (memcmp(recv_buff, uncompressed_fq, uncompressedsize * sizeof(T)) != 0) {
			std::cout << "error in execption 2 check" << std::endl;
			exit(1);
		}*/

                } catch (...) {
                        std::cout << "-----> exception 2";
                        exit(1);
                }
                isCompressed = schema.isCompressed(uncompressedsize, psizeFrom);
                //std::cout<< "exit 4 ..." << std::endl;
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
		try {
                schema.compress(send, psizeTo, &compressed_fq, compressedsize);
		} catch (...) {
                        std::cout << "-----> exception 3";
                        exit(1);
                }
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
			//temporal_recv_buff = (T_C *)malloc (upperId * sizeof(T_C));
            err = posix_memalign((void **)&temporal_recv_buff, 16, upperId * sizeof(T_C));
            if (err) {
                    printf("memory error!\n");
                    throw "Memory error.";
                    exit(1);
            }

                	MPI_Sendrecv(compressed_fq, compressedsize, typeC,
                             previousRank, it + 2,
                             temporal_recv_buff, upperId, typeC,
			     //recv_buff, upperId, type,
                             previousRank, it + 2,
                             comm, &status);
                	MPI_Get_count(&status, typeC, &psizeFrom);

		/*debug*/
                    /*
                MPI_Sendrecv(send, psizeTo, type,
                             previousRank, it + 2,
                             recv_buff, upperId, type,
                             previousRank, it + 2,
                             comm, &status);
                MPI_Get_count(&status, type, &psizeFrom);
*/
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

                //std::cout<< "enter 3 ..." << std::endl;
		try {
                schema.decompress(temporal_recv_buff, psizeFrom, &uncompressed_fq, uncompressedsize);

                /*if (memcmp(recv_buff, uncompressed_fq, uncompressedsize * sizeof(T)) != 0) {
                        std::cout << "error in execption 4 check" << std::endl;
			exit(1);
                }*/
                } catch (...) {
                        std::cout << "-----> exception 4";
                        exit(1);
                }
                isCompressed = schema.isCompressed(uncompressedsize, psizeFrom);
                //std::cout<< "exit 3 ..." << std::endl;
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
    //int *compressed_flags = (int *)malloc(communicatorSize * sizeof(int));

    unsigned int lastReversedSliceIDs = 0;
    unsigned int lastTargetNode = oldRank(lastReversedSliceIDs);
    unsigned int reversedSliceIDs, targetNode;
    size_t csize = 0;
    bool isCompressed = false;

    assert(is_sorted(send, send + psizeTo));
    try {
    	schema.compress(send, psizeTo, &compressed_fq, compressedsize);
    } catch (...) {
        std::cout << "-----> exception 5";
        exit(1);
    }
    size_t uncompressedsize_temp = psizeTo;
    T *uncompressed_buffer_temp = NULL;
    try {
        schema.decompress(compressed_fq, compressedsize, &uncompressed_buffer_temp, uncompressedsize_temp);
        assert(psizeTo == uncompressedsize_temp);
        assert(memcmp(send, uncompressed_buffer_temp, psizeTo * sizeof(T)) == 0);
    	assert(is_sorted(uncompressed_buffer_temp, uncompressed_buffer_temp + uncompressedsize_temp));
    } catch (...) {
        std::cout << "-----> exception 5b";
        exit(1);
    }
    isCompressed = schema.isCompressed(psizeTo, compressedsize);
//std::cout << "c1: isCompressed: " << isCompressed << std::endl;

    //MPI_Datatype MPI_3INT;

    int *composed_recv = (int *)malloc(2 * communicatorSize * sizeof(int));
    int *composed_send = (int *)malloc(2 * sizeof(int));

    composed_send[0] = psizeTo;
    composed_send[1] = compressedsize;
    //composed_send[2] = (isCompressed) ? 1 : 0;

    //MPI_Type_contiguous(3, MPI_INT, &MPI_3INT);
    //MPI_Type_commit(&MPI_3INT);

    MPI_Allgather(composed_send, 1, MPI_2INT, composed_recv, 1, MPI_2INT, comm);

    const int totalsize = 2 * communicatorSize;

    for (int i = 0, j = 0; i < totalsize; ++i)
    {
        if (i % 2 == 0)
        {
            sizes[j] = composed_recv[i];
            compressed_sizes[j] = composed_recv[i + 1];
	    //compressed_flags[j] = composed_recv[i + 2];
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

    //compressed_recv_buff = (T_C *)malloc(csize * sizeof(T_C));
    err = posix_memalign((void **)&compressed_recv_buff, 16, csize * sizeof(T_C));
    if (err) {
        printf("memory error!\n");
        throw "Error allocating memory.";
    }

    std::cout << "disps: ";
    for (int a=0; a < communicatorSize; ++a) {
    std::cout << "("<< communicatorRank <<") " <<disps[a] << " ";
    }
    std::cout << std::endl;

    std::cout << "sizes: ";
    for (int a=0; a < communicatorSize; ++a) {
    std::cout << "("<< communicatorRank <<") " <<sizes[a] << " ";
    }
    std::cout << std::endl;
    std::cout << "rsize: " << rsize << std::endl;

    std::cout << "compressed_disps: ";
    for (int a=0; a < communicatorSize; ++a) {
    std::cout << "("<< communicatorRank <<") " <<compressed_disps[a] << " ";
    }
    std::cout << std::endl;

    std::cout << "compressed_sizes: ";
    for (int a=0; a < communicatorSize; ++a) {
    std::cout << "("<< communicatorRank <<") " <<compressed_sizes[a] << " ";
    }
    std::cout << std::endl;
    std::cout << "csize: " << csize << std::endl;

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

/*
    MPI_Allgatherv(send, sizes[communicatorRank],
                   type, recv_buff, sizes,
                   disps, type, comm);
    std::cout << "allgather comm_rank: " << communicatorRank << std::endl;
*/
/*
    T *recv_buff_tmp=NULL;
    err = posix_memalign((void **)&recv_buff_tmp, 16, rsize * sizeof(T));
    if (err) {
        printf("memory error!\n");
        throw "Error allocating memory.";
    }

    MPI_Allgatherv(send, sizes[communicatorRank],
                   type, recv_buff_tmp, sizes,
                   disps, type, comm);
*/

    MPI_Allgatherv(compressed_fq, compressed_sizes[communicatorRank],
                   typeC, compressed_recv_buff, compressed_sizes,
                   compressed_disps, typeC, comm);

#else
    MPI_Allgatherv(send, sizes[communicatorRank],
                   type, recv_buff, sizes,
                   disps, type, comm);
#endif



    //free(sizes);
    //free(disps);



#ifdef _COMPRESSION



//#ifdef _COMPRESSIONVERIFY
    int total_uncompressedsize = 0;
//#endif

    // reensamble uncompressed chunks
    for (int i = 0; i < communicatorSize; ++i)
    {
        compressedsize = compressed_sizes[i]+1;
        uncompressedsize = sizes[i]+1;

        if (compressedsize != 0)
        {
            try {

                schema.decompress(&compressed_recv_buff[compressed_disps[i]], compressedsize, &uncompressed_fq, uncompressedsize);

            } catch (...) {
                std::cout << "-----> exception 6" << std::endl;
                exit(1);
            }
            total_uncompressedsize += uncompressedsize;
            memcpy(&recv_buff[disps[i]], uncompressed_fq, uncompressedsize * sizeof(T));
            free(uncompressed_fq);
        }
    }
    std::cout<< "rsize: " << rsize<< " total_size: " << total_uncompressedsize<< std::endl;

//#ifdef _COMPRESSIONVERIFY
    if (!std::is_sorted(recv_buff, recv_buff + rsize)) {
        std::cout << "---> unordered decomp buffer (quitting...) " << std::endl;
        exit(1);
    } else {
        std::cout << "---> ordered buffer (correct) " << std::endl;
    }
//#endif

#endif

    free(sizes);
    free(disps);

#ifdef _COMPRESSION
    // MPI_Type_free(&MPI_3INT);
    free(compressed_sizes);
    free(compressed_disps);
    free(compressed_recv_buff);
#endif

}
#endif // VREDUCE_HPP
