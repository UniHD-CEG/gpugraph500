/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Multi-GPU out-of-core BFS implementation (BFS level grid launch)
 ******************************************************************************/

#pragma once

#include <vector>

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/spine.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/operators.cuh>

#include <b40c/graph/bfs/csr_problem.cuh>
#include <b40c/graph/bfs/enactor_base.cuh>
#include <b40c/graph/bfs/problem_type.cuh>

#include <b40c/graph/bfs/two_phase/contract_atomic/kernel.cuh>
#include <b40c/graph/bfs/two_phase/contract_atomic/kernel_policy.cuh>

#include <b40c/graph/bfs/two_phase/expand_atomic/kernel.cuh>
#include <b40c/graph/bfs/two_phase/expand_atomic/kernel_policy.cuh>

#include <b40c/graph/bfs/partition_contract/policy.cuh>
#include <b40c/graph/bfs/partition_contract/upsweep/kernel.cuh>
#include <b40c/graph/bfs/partition_contract/upsweep/kernel_policy.cuh>
#include <b40c/graph/bfs/partition_contract/downsweep/kernel.cuh>
#include <b40c/graph/bfs/partition_contract/downsweep/kernel_policy.cuh>

#include <b40c/graph/bfs/copy/kernel.cuh>
#include <b40c/graph/bfs/copy/kernel_policy.cuh>


namespace b40c {
namespace graph {
namespace bfs {



/**
 * Multi-GPU out-of-core BFS implementation (BFS level grid launch)
 *
 * Each iteration is performed by its own kernel-launch.
 *
 * All GPUs must be of the same SM architectural version (e.g., SM2.0).
 */
template <bool INSTRUMENT>							// Whether or not to collect per-CTA clock-count statistics
class EnactorMultiNode : public EnactorBase
{
public :

	//---------------------------------------------------------------------
	// Policy Structures
	//---------------------------------------------------------------------

	template <typename CsrProblem, int SM_ARCH>
	struct Policy;

	/**
	 * SM2.0 policy
	 */
	template <typename CsrProblem>
	struct Policy<CsrProblem, 200>
	{
		typedef typename CsrProblem::VertexId 		VertexId;
		typedef typename CsrProblem::SizeT 			SizeT;

		// Expansion kernel config
		typedef two_phase::expand_atomic::KernelPolicy<
			typename CsrProblem::ProblemType,
			200,					// CUDA_ARCH
			INSTRUMENT, 			// INSTRUMENT
			8,						// CTA_OCCUPANCY
			7,						// LOG_THREADS
			0,						// LOG_LOAD_VEC_SIZE
			0,						// LOG_LOADS_PER_TILE
			5,						// LOG_RAKING_THREADS
			util::io::ld::cg,		// QUEUE_READ_MODIFIER,
			util::io::ld::NONE,		// COLUMN_READ_MODIFIER,
			util::io::ld::cg,		// ROW_OFFSET_ALIGNED_READ_MODIFIER,
			util::io::ld::NONE,		// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
			util::io::st::cg,		// QUEUE_WRITE_MODIFIER,
			true,					// WORK_STEALING
			32,						// WARP_GATHER_THRESHOLD
			128 * 4, 				// CTA_GATHER_THRESHOLD,
			6>				 		// LOG_SCHEDULE_GRANULARITY
				ExpandPolicy;


		// Contraction kernel config
		typedef two_phase::contract_atomic::KernelPolicy<
			typename CsrProblem::ProblemType,
			200,					// CUDA_ARCH
			INSTRUMENT, 			// INSTRUMENT
			0, 						// SATURATION_QUIT
			false, 					// DEQUEUE_PROBLEM_SIZE
			8,						// CTA_OCCUPANCY
			7,						// LOG_THREADS
			0,						// LOG_LOAD_VEC_SIZE (must be vec-1 since we may be unaligned)
			(sizeof(VertexId) > 4) ? 0 : 2,						// LOG_LOADS_PER_TILE
			5,						// LOG_RAKING_THREADS
			util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
			util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
			false,					// WORK_STEALING
			-1,						// END_BITMASK_CULL (always cull)
			6> 						// LOG_SCHEDULE_GRANULARITY
				ContractPolicy;


		// Partition kernel config (make sure we satisfy the tuning constraints in partition::[up|down]sweep::tuning_policy.cuh)
		typedef partition_contract::Policy<
			// Problem Type
			typename CsrProblem::ProblemType,
			200,
			INSTRUMENT, 			// INSTRUMENT
			CsrProblem::ProblemType::LOG_MAX_GPUS,		// LOG_BINS
			9,						// LOG_SCHEDULE_GRANULARITY
			util::io::ld::NONE,		// CACHE_MODIFIER
			util::io::st::NONE,		// CACHE_MODIFIER

			8,						// UPSWEEP_CTA_OCCUPANCY
			7,						// UPSWEEP_LOG_THREADS
			0,						// UPSWEEP_LOG_LOAD_VEC_SIZE
			2,						// UPSWEEP_LOG_LOADS_PER_TILE

			7,						// SPINE_LOG_THREADS
			2,						// SPINE_LOG_LOAD_VEC_SIZE
			0,						// SPINE_LOG_LOADS_PER_TILE
			5,						// SPINE_LOG_RAKING_THREADS

			partition::downsweep::SCATTER_DIRECT,		// DOWNSWEEP_SCATTER_STRATEGY
			8,						// DOWNSWEEP_CTA_OCCUPANCY
			7,						// DOWNSWEEP_LOG_THREADS
			1,						// DOWNSWEEP_LOG_LOAD_VEC_SIZE
			(sizeof(VertexId) > 4) ? 0 : 1,		// DOWNSWEEP_LOG_LOADS_PER_CYCLE
			0,						// DOWNSWEEP_LOG_CYCLES_PER_TILE
			6> 						// DOWNSWEEP_LOG_RAKING_THREADS
				PartitionPolicy;

		// Copy kernel config
		typedef copy::KernelPolicy<
			typename CsrProblem::ProblemType,
			200,
			INSTRUMENT, 			// INSTRUMENT
			false, 					// DEQUEUE_PROBLEM_SIZE
			6,						// LOG_SCHEDULE_GRANULARITY
			8,						// CTA_OCCUPANCY
			6,						// LOG_THREADS
			0,						// LOG_LOAD_VEC_SIZE (must be vec-1 since we may be unaligned)
			0,						// LOG_LOADS_PER_TILE
			util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
			util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
			false> 					// WORK_STEALING
				CopyPolicy;
	};


protected:

	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Management structure for each GPU
	 */
	struct GpuControlBlock
	{
		//---------------------------------------------------------------------
		// Members
		//---------------------------------------------------------------------

		bool DEBUG;

		// GPU index
		int gpu;

		// GPU cuda properties
		util::CudaProperties cuda_props;

		// Queue size counters and accompanying functionality
		util::CtaWorkProgressLifetime work_progress;

		// Partitioning spine storage
		util::Spine spine;
		int spine_elements;

		int contract_grid_size;			// Contraction grid size
		int expand_grid_size;			// Expansion grid size
		int partition_grid_size;		// Partition/contract grid size
		int copy_grid_size;				// Copy grid size

		long long iteration;			// BFS iteration
		long long queue_index;			// Queuing index
		long long steal_index;			// Work stealing index
		long long queue_length;			// Current queue size

		// Kernel duty stats
		util::KernelRuntimeStatsLifetime contract_kernel_stats;
		util::KernelRuntimeStatsLifetime expand_kernel_stats;
		util::KernelRuntimeStatsLifetime partition_kernel_stats;
		util::KernelRuntimeStatsLifetime copy_kernel_stats;


		//---------------------------------------------------------------------
		// Methods
		//---------------------------------------------------------------------

		/**
		 * Constructor
		 */
		GpuControlBlock(int gpu, bool DEBUG = false) :
			gpu(gpu),
			DEBUG(DEBUG),
			cuda_props(gpu),
			spine(true),				// Host-mapped spine
			spine_elements(0),
			contract_grid_size(0),
			expand_grid_size(0),
			partition_grid_size(0),
			copy_grid_size(0),
			iteration(0),
			steal_index(0),
			queue_index(0),
			queue_length(0)
		{}


		/**
		 * Returns the default maximum number of threadblocks that should be
		 * launched for this GPU.
		 */
		int MaxGridSize(int cta_occupancy, int max_grid_size)
		{
			if (max_grid_size <= 0) {
				// No override: Fully populate all SMs
				max_grid_size = cuda_props.device_props.multiProcessorCount * cta_occupancy;
			}

			return max_grid_size;
		}


		/**
		 * Setup / lazy initialization
		 */
	    template <
	    	typename ContractPolicy,
	    	typename ExpandPolicy,
	    	typename PartitionPolicy,
	    	typename CopyPolicy>
		cudaError_t Setup(int max_grid_size, int num_gpus)
		{
	    	cudaError_t retval = cudaSuccess;

			do {
		    	// Determine grid size(s)
				int contract_min_occupancy 		= ContractPolicy::CTA_OCCUPANCY;
				contract_grid_size 				= MaxGridSize(contract_min_occupancy, max_grid_size);

				int expand_min_occupancy 		= ExpandPolicy::CTA_OCCUPANCY;
				expand_grid_size 				= MaxGridSize(expand_min_occupancy, max_grid_size);

				int partition_min_occupancy		= B40CG_MIN((int) PartitionPolicy::Upsweep::MAX_CTA_OCCUPANCY, (int) PartitionPolicy::Downsweep::MAX_CTA_OCCUPANCY);
				partition_grid_size 			= MaxGridSize(partition_min_occupancy, max_grid_size);

				int copy_min_occupancy			= CopyPolicy::CTA_OCCUPANCY;
				copy_grid_size 					= MaxGridSize(copy_min_occupancy, max_grid_size);

				// Setup partitioning spine
				spine_elements = (partition_grid_size * PartitionPolicy::Upsweep::BINS) + 1;
				if (retval = spine.template Setup<typename PartitionPolicy::SizeT>(spine_elements)) break;

				if (DEBUG) printf("Gpu %d contract  min occupancy %d, grid size %d\n",
					gpu, contract_min_occupancy, contract_grid_size);
				if (DEBUG) printf("Gpu %d expand min occupancy %d, grid size %d\n",
					gpu, expand_min_occupancy, expand_grid_size);
				if (DEBUG) printf("Gpu %d partition min occupancy %d, grid size %d, spine elements %d\n",
					gpu, partition_min_occupancy, partition_grid_size, spine_elements);
				if (DEBUG) printf("Gpu %d copy min occupancy %d, grid size %d\n",
					gpu, copy_min_occupancy, copy_grid_size);

				// Setup work progress
				if (retval = work_progress.Setup()) break;

			} while (0);

			// Reset statistics
			iteration = 0;
			queue_index = 0;
			steal_index = 0;
			queue_length = 0;

			return retval;
		}


	    /**
	     * Updates queue length from work progress
	     *
	     * (SizeT may be different for each graph search)
	     */
		template <typename SizeT>
	    cudaError_t UpdateQueueLength()
	    {
	    	SizeT length;
	    	cudaError_t retval = work_progress.GetQueueLength(queue_index, length);
	    	queue_length = length;

	    	return retval;
	    }
	};

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Vector of GpuControlBlocks (one for each GPU)
	std::vector <GpuControlBlock *> control_blocks;

	bool DEBUG2;

	//---------------------------------------------------------------------
	// Utility Methods
	//---------------------------------------------------------------------


public:

	/**
	 * Constructor
	 */
	EnactorMultiNode(bool DEBUG = false) :
		EnactorBase(MULTI_GPU_FRONTIERS, DEBUG),
		DEBUG2(false)
	{}

	/**
	 * Resets control blocks
	 */
	void ResetControlBlocks()
	{
		// Cleanup control blocks on the heap
		for (typename std::vector<GpuControlBlock*>::iterator itr = control_blocks.begin();
			itr != control_blocks.end();
			itr++)
		{
			if (*itr) delete (*itr);
		}

		control_blocks.clear();
	}


	/**
	 * Destructor
	 */
	virtual ~EnactorMultiNode()
	{
		ResetControlBlocks();
	}


    /**
     * Obtain statistics about the last BFS search enacted
     */
	template <typename VertexId>
    void GetStatistics(
    	long long &total_queued,
    	VertexId &search_depth,
    	double &avg_live)
    {
		// TODO
    	total_queued = 0;
    	search_depth = 0;
    	avg_live = 0;
    }


	/**
	 * Search setup / lazy initialization
	 */
    template <
    	typename ContractPolicy,
    	typename ExpandPolicy,
    	typename PartitionPolicy,
    	typename CopyPolicy,
    	typename CsrProblem>
	cudaError_t Setup(
		CsrProblem 		&csr_problem,
		int 			max_grid_size)
    {
		typedef typename CsrProblem::SizeT 			SizeT;
		typedef typename CsrProblem::VertexId 		VertexId;
		typedef typename CsrProblem::VisitedMask 	VisitedMask;

		cudaError_t retval = cudaSuccess;

    	do {
			// Check if last run was with an different number of GPUs (in which
			// case the control blocks are all misconfigured)
			if (control_blocks.size() != csr_problem.num_gpus) {

				ResetControlBlocks();

				for (int i = 0; i < csr_problem.num_gpus; i++) {

					// Set device
					if (retval = util::B40CPerror(cudaSetDevice(csr_problem.graph_slices[i]->gpu),
						"EnactorMultiNode cudaSetDevice failed", __FILE__, __LINE__)) break;

					control_blocks.push_back(
						new GpuControlBlock(csr_problem.graph_slices[i]->gpu,
						DEBUG));
				}
			}

			// Setup control blocks
			for (int i = 0; i < csr_problem.num_gpus; i++) {

				// Set device
				if (retval = util::B40CPerror(cudaSetDevice(csr_problem.graph_slices[i]->gpu),
					"EnactorMultiNode cudaSetDevice failed", __FILE__, __LINE__)) break;

				if (retval = control_blocks[i]->template Setup<ContractPolicy, ExpandPolicy, PartitionPolicy, CopyPolicy>(
					max_grid_size, csr_problem.num_gpus)) break;

				// Bind bitmask textures
				int bytes = (csr_problem.nodes + 8 - 1) / 8;
				cudaChannelFormatDesc bitmask_desc = cudaCreateChannelDesc<char>();
				if (retval = util::B40CPerror(cudaBindTexture(
						0,
						two_phase::contract_atomic::BitmaskTex<VisitedMask>::ref,
						csr_problem.graph_slices[i]->d_visited_mask,
						bitmask_desc,
						bytes),
					"EnactorMultiNode cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;

				// Bind row-offsets texture
				cudaChannelFormatDesc row_offsets_desc = cudaCreateChannelDesc<SizeT>();
				if (retval = util::B40CPerror(cudaBindTexture(
						0,
						two_phase::expand_atomic::RowOffsetTex<SizeT>::ref,
						csr_problem.graph_slices[i]->d_row_offsets,
						row_offsets_desc,
						(csr_problem.graph_slices[i]->nodes + 1) * sizeof(SizeT)),
					"EnactorMultiNode cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

			}
			if (retval) break;

    	} while (0);

    	return retval;
    }

	/**
	 * Check if BFS is done
	*/
	bool isBFSDone(bool done, int world_rank, int num_nodes){

		int doneInt=0;
		if(done)
			doneInt = 1;


		//rank 0 receives all the status and broadcast the result back to all the other nodes
		//if rank 0, receives
		int *receiveBuf = (int *)malloc(num_nodes*sizeof(int));

		//MPI_Gather - rank 0 gathers all status from other nodes
		MPI_Gather(&doneInt, 1, MPI_INT, receiveBuf, 1, MPI_INT, 0, MPI_COMM_WORLD);

		int ind = 0;
		bool isAllDone = true;
		//if world_rank is done, check for others
		if(done&&world_rank==0){
			for (ind = 1; ind<num_nodes; ind++){
				if(receiveBuf[ind]==0){
					isAllDone = false;
					break;
				}
			}
		}
		else{
			isAllDone = false;
		}

		if(isAllDone){
			doneInt = 1;
		}

		//MPI_Bcast - rank 0 broadcast the results to others
		MPI_Bcast(&doneInt, 1, MPI_INT, 0, MPI_COMM_WORLD);

		//check for done
		if(doneInt == 1)
			return true;
		else
			return false;
	}

    template <
    	typename CopyPolicy,
	typename ContractPolicy,
    	typename CsrProblem>
	cudaError_t nodeExchange(std::vector <GpuControlBlock*> control_blocks, CsrProblem &csr_problem,cudaError_t retval, int num_nodes, int world_rank, int bins_per_gpu){
		typedef typename CsrProblem::VertexId			VertexId;
		typedef typename CsrProblem::SizeT				SizeT;
		typedef typename CsrProblem::GraphSlice			GraphSlice;

		SizeT offsetArray[num_nodes];

		//For MPI_Alltoall and MPI_Alltoallv
		int *sendArray,  *recvArray;
		int *sendArray2, *recvArray2;
		int *sendDisp, *recvDisp;
		int *sendCounts, *recvCounts;
		int sendBufferSize=0, recvBufferSize=0;

		sendCounts = (int*)malloc(sizeof(int)*num_nodes);
		recvCounts = (int*)malloc(sizeof(int)*num_nodes);
		sendDisp   = (int*)malloc(sizeof(int)*num_nodes);
		recvDisp   = (int*)malloc(sizeof(int)*num_nodes);

		//only one gpu per node for now
		int peer =						0;
		GpuControlBlock *peer_control 		= control_blocks[peer];
		GraphSlice *peer_slice 				= csr_problem.graph_slices[peer];
		SizeT *peer_spine 			= (SizeT*) peer_control->spine.h_spine;

		//iterate through all nodes
		for(int i=0; i<num_nodes; i++){

			SizeT queue_offset 	= peer_spine[bins_per_gpu * i * peer_control->partition_grid_size];
			SizeT queue_oob 	= peer_spine[bins_per_gpu * (i + 1) * peer_control->partition_grid_size];
			SizeT num_elements	= queue_oob - queue_offset;
			// Check for vertex frontier overflow
			if (num_elements > peer_slice->frontier_elements[0]) {
				retval = util::B40CPerror(cudaErrorInvalidConfiguration, "Frontier queue overflow.  Please increase queue-sizing factor. ", __FILE__, __LINE__);
			}
			sendCounts[i] = num_elements;
			offsetArray[i]= queue_offset;
		}

		// exchange num_elements using MPI_Alltoall
		MPI_Alltoall(sendCounts, 1, MPI_INT, recvCounts, 1, MPI_INT, MPI_COMM_WORLD);

		// calculate displacements
		sendDisp[0] = 0;
		for(int i=1; i<num_nodes; i++){
			sendDisp[i] = sendCounts[i-1] + sendDisp[i-1];
		}
		recvDisp[0] = 0;
		for(int i=1; i<num_nodes; i++){
			recvDisp[i] = recvCounts[i-1] + recvDisp[i-1];
		}

		// calculate size of arrays
		for(int i=0; i<num_nodes; i++){
			sendBufferSize+=sendCounts[i];
			recvBufferSize+=recvCounts[i];
		}

		// allocate for send and receive total array
		sendArray =  (int*)malloc(sizeof(int)*sendBufferSize);
		recvArray =  (int*)malloc(sizeof(int)*recvBufferSize);
		sendArray2 = (int*)malloc(sizeof(int)*sendBufferSize);
		recvArray2 = (int*)malloc(sizeof(int)*recvBufferSize);

		for(int i=0; i<num_nodes; i++){
			VertexId *hostd_keys= (VertexId*)malloc(sendCounts[i]*sizeof(VertexId));
			VertexId *hostd_values= (VertexId*)malloc(sendCounts[i]*sizeof(VertexId));
			//need to copy to the host before sending to other nodes
			cudaMemcpy(hostd_keys,peer_slice->frontier_queues.d_keys[2]+offsetArray[i],sizeof(VertexId)*sendCounts[i],cudaMemcpyDeviceToHost);
			cudaMemcpy(hostd_values,peer_slice->frontier_queues.d_values[2]+offsetArray[i],sizeof(VertexId)*sendCounts[i],cudaMemcpyDeviceToHost);
			memcpy(sendArray+sendDisp[i],hostd_keys,sizeof(int)*sendCounts[i]);
			memcpy(sendArray2+sendDisp[i],hostd_values,sizeof(int)*sendCounts[i]);
		}


		// exchange using MPI_Alltoallv
		printf("DEBUG:: MPI_Alltoallv - EnactorMultiNode\n");
		MPI_Alltoallv(sendArray, sendCounts, sendDisp, MPI_INT, recvArray, recvCounts, recvDisp, MPI_INT, MPI_COMM_WORLD);
		MPI_Alltoallv(sendArray2, sendCounts, sendDisp, MPI_INT, recvArray2, recvCounts, recvDisp, MPI_INT, MPI_COMM_WORLD);

		for(int i=0; i<num_nodes; i++){
			// its own
			if(i == world_rank){
				util::CtaWorkDistribution<SizeT> work_decomposition;
				work_decomposition.template Init<CopyPolicy::LOG_SCHEDULE_GRANULARITY>(
					recvCounts[i], peer_control->copy_grid_size);
				// Simply copy from our own GPU
				copy::Kernel<CopyPolicy>
					<<<peer_control->copy_grid_size, CopyPolicy::THREADS, 0, peer_slice->stream>>>(
						peer_control->iteration,
						sendCounts[i],
						peer_control->queue_index,
						peer_control->steal_index,
						num_nodes,//2,//csr_problem.num_gpus,
						peer_slice->frontier_queues.d_keys[2] + offsetArray[i],			// in local sorted, filtered edge frontier
						peer_slice->frontier_queues.d_keys[0],							// out local vertex frontier
						peer_slice->frontier_queues.d_values[2] + offsetArray[i],			// in local sorted, filtered predecessors
						peer_slice->d_labels,
						peer_control->work_progress,
						peer_control->copy_kernel_stats);

				if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(),
					"EnactorMultiNode copy::Kernel failed ", __FILE__, __LINE__))) break;
			}
			// from other nodes
			else{
						VertexId *deviced_keys;
						VertexId *deviced_values;
						if (retval = util::B40CPerror(cudaMalloc((void**) &deviced_keys,recvCounts[i]*sizeof(VertexId)),
				"device cudaMalloc edge frontier failed", __FILE__, __LINE__)) break;
						if (retval = util::B40CPerror(cudaMalloc((void**) &deviced_values,recvCounts[i]*sizeof(VertexId)),
				"device cudaMalloc predecessor failed", __FILE__, __LINE__)) break;

						cudaMemcpy(deviced_keys, recvArray+recvDisp[i],sizeof(VertexId)*recvCounts[i], cudaMemcpyHostToDevice);
						cudaMemcpy(deviced_values, recvArray+recvDisp[i],sizeof(VertexId)*recvCounts[i], cudaMemcpyHostToDevice);
						// Contraction from peer GPU
						two_phase::contract_atomic::Kernel<ContractPolicy>
							<<<peer_control->contract_grid_size, ContractPolicy::THREADS, 0, peer_slice->stream>>>(
							-1,															// source (not used)
							peer_control->iteration,
							sendCounts[i],
							peer_control->queue_index,
							peer_control->steal_index,
							num_nodes,//2,//csr_problem.num_gpus,
							NULL,														// d_done (not used)
							deviced_keys,		// in remote sorted, filtered edge frontier
							peer_slice->frontier_queues.d_keys[0],							// out local vertex frontier
							deviced_values,		// in remote sorted, filtered predecessors
							peer_slice->d_labels,
							peer_slice->d_visited_mask,
							peer_control->work_progress,
							peer_slice->frontier_elements[2],								// max edge frontier vertices
							peer_slice->frontier_elements[0],								// max vertex frontier vertices
							peer_control->expand_kernel_stats);
					if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(),
						"EnactorMultiNode contract_atomic::Kernel failed ", __FILE__, __LINE__))) break;
			}
			peer_control->steal_index++;
		}
		peer_control->queue_index++;
		return retval;
	}

	/**
	 * Enacts a breadth-first-search on the specified graph problem.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <
    	typename ContractPolicy,
    	typename ExpandPolicy,
    	typename PartitionPolicy,
    	typename CopyPolicy,
    	typename CsrProblem>
	cudaError_t EnactSearch(
		CsrProblem 							&csr_problem,
		typename CsrProblem::VertexId 		src,
		int 								world_rank,
		int								num_nodes,
		int 								max_grid_size = 0)
	{
		typedef typename CsrProblem::VertexId			VertexId;
		typedef typename CsrProblem::SizeT				SizeT;
		typedef typename CsrProblem::GraphSlice			GraphSlice;

		typedef typename PartitionPolicy::Upsweep		PartitionUpsweep;
		typedef typename PartitionPolicy::Spine			PartitionSpine;
		typedef typename PartitionPolicy::Downsweep		PartitionDownsweep;

		typedef Policy<CsrProblem, 200> CsrPolicy;

		cudaError_t retval = cudaSuccess;
		bool done;
		DEBUG=true;
		DEBUG2=true;
		do {
			// Number of partitioning bins per GPU (in case we over-partition)
//			int bins_per_gpu = (csr_problem.num_gpus == 1) ?
//				PartitionPolicy::Upsweep::BINS :
//				1;
			int bins_per_gpu =1;

			// Search setup / lazy initialization
			if (retval = Setup<ContractPolicy, ExpandPolicy, PartitionPolicy, CopyPolicy>(
				csr_problem, max_grid_size)) break;

			// Mask in owner gpu of source;
			VertexId src_owner = csr_problem.GpuIndex(src);
			src |= (src_owner << CsrProblem::ProblemType::GPU_MASK_SHIFT);

			//---------------------------------------------------------------------
			// Contract work queues (first iteration)
			//---------------------------------------------------------------------

			for (int i = 0; i < csr_problem.num_gpus; i++) {

				GpuControlBlock *control 	= control_blocks[i];
				GraphSlice *slice 			= csr_problem.graph_slices[i];

				// Set device
				if (retval = util::B40CPerror(cudaSetDevice(control->gpu),
					"EnactorMultiNode cudaSetDevice failed", __FILE__, __LINE__)) break;

				bool owns_source = (world_rank == src_owner);
				if (owns_source) {
					printf("Node %d owns source 0x%llX\n", world_rank, (long long) src);
				}
				// Contraction
				two_phase::contract_atomic::Kernel<ContractPolicy>
						<<<control->contract_grid_size, ContractPolicy::THREADS, 0, slice->stream>>>(
					(owns_source) ? src : -1,
					control->iteration,
					(owns_source) ? 1 : 0,
					control->queue_index,
					control->steal_index,
					num_nodes,//2,
					NULL,										// d_done (not used)
					slice->frontier_queues.d_keys[2],			// in filtered edge frontier
					slice->frontier_queues.d_keys[0],			// out vertex frontier
					slice->frontier_queues.d_values[2],			// in predecessors
					slice->d_labels,
					slice->d_visited_mask,
					control->work_progress,
					slice->frontier_elements[2],				// max edge frontier vertices
					slice->frontier_elements[0],				// max vertex frontier vertices
					control->expand_kernel_stats);

				if (DEBUG && (retval = util::B40CPerror(cudaDeviceSynchronize(),
					"EnactorMultiNode expand_atomic::Kernel failed", __FILE__, __LINE__))) break;

				control->queue_index++;
				control->steal_index++;

				if (DEBUG){
					// Get contraction queue length
					if (retval = control->template UpdateQueueLength<SizeT>()) break;
					printf("Gpu %d contracted queue length: %lld\n", i, (long long) control->queue_length);
					fflush(stdout);
				}

			}
			if (retval) break;

			// BFS passes
			while (true) {

				//---------------------------------------------------------------------
				// Expand work queues
				//---------------------------------------------------------------------
				//change
				for (int i = 0; i < csr_problem.num_gpus; i++) {

					GpuControlBlock *control 	= control_blocks[i];
					GraphSlice *slice 			= csr_problem.graph_slices[i];

					// Set device
					if (retval = util::B40CPerror(cudaSetDevice(control->gpu),
						"EnactorMultiNode cudaSetDevice failed", __FILE__, __LINE__)) break;

					two_phase::expand_atomic::Kernel<ExpandPolicy>
							<<<control->expand_grid_size, ExpandPolicy::THREADS, 0, slice->stream>>>(
						control->queue_index,
						control->steal_index,
						num_nodes,//2,//csr_problem.num_gpus,
						NULL,										// d_done (not used)
						slice->frontier_queues.d_keys[0],			// in local vertex frontier
						slice->frontier_queues.d_keys[1],			// out local edge frontier
						slice->frontier_queues.d_values[1],			// out local predecessors
						slice->d_column_indices,
						slice->d_row_offsets,
						control->work_progress,
						slice->frontier_elements[0],				// max vertex frontier vertices
						slice->frontier_elements[1],				// max edge frontier vertices
						control->expand_kernel_stats);

					if (DEBUG && (retval = util::B40CPerror(cudaDeviceSynchronize(),
						"EnactorMultiNode expand_atomic::Kernel failed", __FILE__, __LINE__))) break;

					control->queue_index++;
					control->steal_index++;
					control->iteration++;

					if (DEBUG) {
						// Get expansion queue length
						if (retval = control->template UpdateQueueLength<SizeT>()) break;
						printf("Node %d expansion queue length: %lld\n", world_rank, (long long) control->queue_length);
						fflush(stdout);
					}
				}
				if (retval) break;

				//---------------------------------------------------------------------
				// Partition/contract work queues
				//---------------------------------------------------------------------

				for (int i = 0; i < csr_problem.num_gpus; i++) {

					GpuControlBlock *control 	= control_blocks[i];
					GraphSlice *slice 			= csr_problem.graph_slices[i];

					// Set device
					if (retval = util::B40CPerror(cudaSetDevice(control->gpu),
						"EnactorMultiNode cudaSetDevice failed", __FILE__, __LINE__)) break;

					// Upsweep
					partition_contract::upsweep::Kernel<PartitionUpsweep>
							<<<control->partition_grid_size, PartitionUpsweep::THREADS, 0, slice->stream>>>(
						control->queue_index,
						num_nodes,//2,//csr_problem.num_gpus,
						slice->frontier_queues.d_keys[1],			// in local edge frontier
						slice->d_filter_mask,
						(SizeT *) control->spine.d_spine,
						slice->d_visited_mask,
						control->work_progress,
						slice->frontier_elements[1],					// max local edge frontier vertices
						control->partition_kernel_stats);

					if (DEBUG && (retval = util::B40CPerror(cudaDeviceSynchronize(),
						"EnactorMultiNode partition_contract::upsweep::Kernel failed", __FILE__, __LINE__))) break;

					if (DEBUG2) {
						printf("Presorted spine on node %d (%lld elements)\n",
							world_rank,
							(long long) control->spine_elements);
						DisplayDeviceResults((SizeT *) control->spine.d_spine, control->spine_elements);
					}

					// Spine
					PartitionPolicy::SpineKernel()<<<1, PartitionSpine::THREADS, 0, slice->stream>>>(
						(SizeT*) control->spine.d_spine,
						(SizeT*) control->spine.d_spine,
						control->spine_elements);

					if (DEBUG && (retval = util::B40CPerror(cudaDeviceSynchronize(),
						"EnactorMultiNode SpineKernel failed", __FILE__, __LINE__))) break;

					if (DEBUG2) {
						printf("Postsorted spine on node %d (%lld elements)\n",
							world_rank,
							(long long) control->spine_elements);

						DisplayDeviceResults((SizeT *) control->spine.d_spine, control->spine_elements);
					}

					// Downsweep
					partition_contract::downsweep::Kernel<PartitionDownsweep>
							<<<control->partition_grid_size, PartitionDownsweep::THREADS, 0, slice->stream>>>(
						control->queue_index,
						num_nodes,//2,//csr_problem.num_gpus,
						slice->frontier_queues.d_keys[1],				// in local edge frontier
						slice->frontier_queues.d_keys[2],				// out local sorted, filtered edge frontier
						slice->frontier_queues.d_values[1],				// in local predecessors
						slice->frontier_queues.d_values[2],				// out local sorted, filtered predecessors
						slice->d_filter_mask,
						(SizeT *) control->spine.d_spine,
						control->work_progress,
						slice->frontier_elements[1],					// max local edge frontier vertices
						control->partition_kernel_stats);

					if (DEBUG && (retval = util::B40CPerror(cudaDeviceSynchronize(),
						"EnactorMultiNode DownsweepKernel failed", __FILE__, __LINE__))) break;

					control->queue_index++;
				}
				if (retval) break;

				//---------------------------------------------------------------------
				// Synchronization point (to make spines coherent)
				//---------------------------------------------------------------------

				done = true;
				for (int i = 0; i < csr_problem.num_gpus; i++) {

					GpuControlBlock *control 	= control_blocks[i];
					GraphSlice *slice 			= csr_problem.graph_slices[i];
					if (retval = util::B40CPerror(cudaSetDevice(control->gpu),
						"EnactorMultiNode cudaSetDevice failed", __FILE__, __LINE__)) break;

					// The memcopy for spine sync synchronizes this GPU
					control->spine.Sync();

					SizeT *spine = (SizeT *) control->spine.h_spine;
					if (spine[control->spine_elements - 1]) done = false;

					if (DEBUG) {
						printf("Iteration %lld sort-contracted queue on node %d (%lld elements)\n",
							(long long) control->iteration,
							world_rank,
							(long long) spine[control->spine_elements - 1]);

						if (DEBUG2) {
							DisplayDeviceResults(slice->frontier_queues.d_keys[0], spine[control->spine_elements - 1]);
							printf("Source distance vector on node %d\n", world_rank);
							DisplayDeviceResults(slice->d_labels, slice->nodes);
						}
					}
				}
				if (retval) break;

				if (DEBUG2) printf("---------------------------------------------------------\n");


				//---------------------------------------------------------------------
				// Stream-contract work queues
				//---------------------------------------------------------------------
//				MPI_Barrier(MPI_COMM_WORLD);

				//Check if BFS is done
				if(isBFSDone(done, world_rank, num_nodes))
					break;
				//nodeExchange(control_blocks, csr_problem, retval,num_nodes, world_rank, bins_per_gpu);


				retval = nodeExchange<typename CsrPolicy::CopyPolicy, typename CsrPolicy::ContractPolicy,CsrProblem>(control_blocks,csr_problem, retval,num_nodes, world_rank, bins_per_gpu);
				if(retval)
					break;


			}
			if (retval) break;


			// Check if any of the frontiers overflowed due to redundant expansion
			for (int i = 0; i < csr_problem.num_gpus; i++) {

				GpuControlBlock *control = control_blocks[i];

				// Set device
				if (retval = util::B40CPerror(cudaSetDevice(control->gpu),
					"EnactorMultiNode cudaSetDevice failed", __FILE__, __LINE__)) break;

				bool overflowed = false;
				if (retval = control->work_progress.template CheckOverflow<SizeT>(overflowed)) break;
				if (overflowed) {
					retval = util::B40CPerror(cudaErrorInvalidConfiguration, "Frontier queue overflow.  Please increase queue-sizing factor. ", __FILE__, __LINE__);
					break;
				}
			}
			if (retval) break;


		} while (0);

		return retval;
	}


    /**
	 * Enacts a breadth-first-search on the specified graph problem.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <typename CsrProblem>
	cudaError_t EnactSearch(
		CsrProblem 							&csr_problem,
		typename CsrProblem::VertexId 		src,
		int					world_rank,
		int					num_nodes,
		int 								max_grid_size = 0)
	{
		typedef typename CsrProblem::VertexId			VertexId;
		typedef typename CsrProblem::SizeT				SizeT;

		if (this->cuda_props.device_sm_version >= 200) {

			typedef Policy<CsrProblem, 200> CsrPolicy;

			return EnactSearch<
				typename CsrPolicy::ContractPolicy,
				typename CsrPolicy::ExpandPolicy,
				typename CsrPolicy::PartitionPolicy,
				typename CsrPolicy::CopyPolicy>(csr_problem, src, world_rank,num_nodes,max_grid_size);
		}

		printf("Not yet tuned for this architecture\n");
		return cudaErrorInvalidDeviceFunction;
	}


};



} // namespace bfs
} // namespace graph
} // namespace b40c
