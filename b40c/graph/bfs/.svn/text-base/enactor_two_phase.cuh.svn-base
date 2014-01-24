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
 * Two-phase BFS enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/spine.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>

#include <b40c/graph/bfs/problem_type.cuh>
#include <b40c/graph/bfs/enactor_base.cuh>

#include <b40c/graph/bfs/two_phase/kernel.cuh>
#include <b40c/graph/bfs/two_phase/kernel_policy.cuh>
#include <b40c/graph/bfs/two_phase/expand_atomic/kernel.cuh>

#include <b40c/graph/bfs/two_phase/expand_atomic/kernel.cuh>
#include <b40c/graph/bfs/two_phase/expand_atomic/kernel_policy.cuh>
#include <b40c/graph/bfs/two_phase/filter_atomic/kernel.cuh>
#include <b40c/graph/bfs/two_phase/filter_atomic/kernel_policy.cuh>
#include <b40c/graph/bfs/two_phase/contract_atomic/kernel.cuh>
#include <b40c/graph/bfs/two_phase/contract_atomic/kernel_policy.cuh>

namespace b40c {
namespace graph {
namespace bfs {



/**
 * Two-phase BFS enactor
 *  
 * For each BFS iteration, visited/duplicate vertices are culled from
 * the incoming edge-frontier in global memory.  The remaining vertices are
 * compacted to a vertex-frontier in global memory.  Then these
 * vertices are read back in and expanded to construct the outgoing
 * edge-frontier in global memory.
 */
template <bool INSTRUMENT>							// Whether or not to collect per-CTA clock-count statistics
class EnactorTwoPhase : public EnactorBase
{

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

protected:

	/**
	 * CTA duty kernel stats
	 */
	util::KernelRuntimeStatsLifetime expand_kernel_stats;
	util::KernelRuntimeStatsLifetime filter_kernel_stats;
	util::KernelRuntimeStatsLifetime contract_kernel_stats;

	unsigned long long 		total_runtimes;			// Total time "worked" by each cta
	unsigned long long 		total_lifetimes;		// Total time elapsed by each cta
	unsigned long long 		total_queued;

	/**
	 * Throttle state.  We want the host to have an additional BFS iteration
	 * of kernel launches queued up for for pipeline efficiency (particularly on
	 * Windows), so we keep a pinned, mapped word that the traversal kernels will
	 * signal when done.
	 */
	volatile int 	*done;
	int 			*d_done;
	cudaEvent_t		throttle_event;

	/**
	 * Mechanism for implementing software global barriers from within
	 * a single grid invocation
	 */
	util::GlobalBarrierLifetime 		global_barrier;

	/**
	 * Current iteration (mapped into GPU space so that it can
	 * be modified by multi-iteration kernel launches)
	 */
	volatile long long 					*iteration;
	long long 							*d_iteration;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

protected:

	/**
	 * Prepare enactor for search.  Must be called prior to each search.
	 */
	template <typename CsrProblem>
	cudaError_t Setup(
		CsrProblem &csr_problem,
		int expand_grid_size,
		int contract_grid_size,
		int filter_grid_size)
	{
		typedef typename CsrProblem::SizeT 			SizeT;
		typedef typename CsrProblem::VertexId 		VertexId;
		typedef typename CsrProblem::VisitedMask 	VisitedMask;

		cudaError_t retval = cudaSuccess;

		do {

			// Make sure host-mapped "done" is initialized
			if (!done) {
				int flags = cudaHostAllocMapped;

				// Allocate pinned memory for done
				if (retval = util::B40CPerror(cudaHostAlloc((void **)&done, sizeof(int) * 1, flags),
					"EnactorContractExpand cudaHostAlloc done failed", __FILE__, __LINE__)) break;

				// Map done into GPU space
				if (retval = util::B40CPerror(cudaHostGetDevicePointer((void **)&d_done, (void *) done, 0),
					"EnactorContractExpand cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

				// Create throttle event
				if (retval = util::B40CPerror(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
					"EnactorContractExpand cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;
			}

			// Make sure host-mapped "iteration" is initialized
			if (!iteration) {

				int flags = cudaHostAllocMapped;

				// Allocate pinned memory
				if (retval = util::B40CPerror(cudaHostAlloc((void **)&iteration, sizeof(long long) * 1, flags),
					"EnactorContractExpand cudaHostAlloc iteration failed", __FILE__, __LINE__)) break;

				// Map into GPU space
				if (retval = util::B40CPerror(cudaHostGetDevicePointer((void **)&d_iteration, (void *) iteration, 0),
					"EnactorContractExpand cudaHostGetDevicePointer iteration failed", __FILE__, __LINE__)) break;
			}

			// Make sure software global barriers are initialized
			if (retval = global_barrier.Setup(expand_grid_size)) break;

			// Make sure our runtime stats are initialized
			if (retval = expand_kernel_stats.Setup(expand_grid_size)) break;
			if (retval = filter_kernel_stats.Setup(filter_grid_size)) break;
			if (retval = contract_kernel_stats.Setup(contract_grid_size)) break;

			// Reset statistics
			iteration[0] 		= 0;
			total_runtimes 		= 0;
			total_lifetimes 	= 0;
			total_queued 		= 0;
			done[0] 			= -1;

			// Single-gpu graph slice
			typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

			// Bind bitmask texture
			int bytes = (graph_slice->nodes + 8 - 1) / 8;
			cudaChannelFormatDesc bitmask_desc = cudaCreateChannelDesc<char>();
			if (retval = util::B40CPerror(cudaBindTexture(
					0,
					two_phase::contract_atomic::BitmaskTex<VisitedMask>::ref,
					graph_slice->d_visited_mask,
					bitmask_desc,
					bytes),
				"EnactorTwoPhase cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;

			// Bind row-offsets texture
			cudaChannelFormatDesc row_offsets_desc = cudaCreateChannelDesc<SizeT>();
			if (retval = util::B40CPerror(cudaBindTexture(
					0,
					two_phase::expand_atomic::RowOffsetTex<SizeT>::ref,
					graph_slice->d_row_offsets,
					row_offsets_desc,
					(graph_slice->nodes + 1) * sizeof(SizeT)),
				"EnactorTwoPhase cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

		} while (0);

		return retval;
	}


public: 	
	
	/**
	 * Constructor
	 */
	EnactorTwoPhase(bool DEBUG = false) :
		EnactorBase(EDGE_FRONTIERS, DEBUG),
		iteration(NULL),
		d_iteration(NULL),
		total_queued(0),
		done(NULL),
		d_done(NULL)
	{}


	/**
	 * Destructor
	 */
	virtual ~EnactorTwoPhase()
	{
		if (iteration) {
			util::B40CPerror(cudaFreeHost((void *) iteration), "EnactorTwoPhase cudaFreeHost iteration failed", __FILE__, __LINE__);
		}
		if (done) {
			util::B40CPerror(cudaFreeHost((void *) done),
					"EnactorTwoPhase cudaFreeHost done failed", __FILE__, __LINE__);

			util::B40CPerror(cudaEventDestroy(throttle_event),
				"EnactorTwoPhase cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
		}
	}


    /**
     * Obtain statistics about the last BFS search enacted 
     */
	template <typename VertexId>
    void GetStatistics(
    	long long &total_queued,
    	VertexId &search_depth,
    	double &avg_duty)
    {
		cudaThreadSynchronize();

		total_queued = this->total_queued;
    	search_depth = this->iteration[0] - 1;

    	avg_duty = (total_lifetimes > 0) ?
    		double(total_runtimes) / total_lifetimes :
    		0.0;
    }

    
	/**
	 * Enacts a breadth-first-search on the specified graph problem.  Invokes
	 * a single grid kernel that itself steps over BFS iterations.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <
		typename KernelPolicy,
		typename CsrProblem>
	cudaError_t EnactFusedSearch(
		CsrProblem 						&csr_problem,
		typename CsrProblem::VertexId 	src,
		int 							max_grid_size = 0)
	{
		typedef typename CsrProblem::SizeT 			SizeT;
		typedef typename CsrProblem::VertexId 		VertexId;
		typedef typename CsrProblem::VisitedMask 	VisitedMask;

		cudaError_t retval = cudaSuccess;

		do {

			// Determine grid size
			int occupancy = KernelPolicy::CTA_OCCUPANCY;
			int grid_size = MaxGridSize(occupancy, max_grid_size);

			if (DEBUG) {
				printf("DEBUG: BFS occupancy %d, grid size %d\n", occupancy, grid_size);
				fflush(stdout);
			}

			// Single-gpu graph slice
			typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

			// Lazy initialization
			if (retval = Setup(csr_problem, grid_size)) break;

			// Initiate single-grid kernel
			two_phase::Kernel<KernelPolicy>
					<<<grid_size, KernelPolicy::THREADS>>>(
				0,												// iteration
				0,												// queue_index
				0,												// steal_index
				src,

				graph_slice->frontier_queues.d_keys[1],			// edge frontier
				graph_slice->frontier_queues.d_keys[0],			// vertex frontier
				graph_slice->frontier_queues.d_values[1],		// predecessor edge frontier

				graph_slice->d_column_indices,
				graph_slice->d_row_offsets,
				graph_slice->d_labels,
				graph_slice->d_visited_mask,
				this->work_progress,
				graph_slice->frontier_elements[1],				// max edge frontier vertices
				graph_slice->frontier_elements[0],				// max vertex frontier vertices
				this->global_barrier,

				this->expand_kernel_stats,
				(VertexId *) d_iteration);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "EnactorFusedTwoPhase Kernel failed ", __FILE__, __LINE__))) break;

			if (INSTRUMENT) {
				// Get stats
				if (retval = expand_kernel_stats.Accumulate(
					grid_size,
					total_runtimes,
					total_lifetimes,
					total_queued)) break;
			}

			// Check if any of the frontiers overflowed due to redundant expansion
			bool overflowed = false;
			if (retval = work_progress.CheckOverflow<SizeT>(overflowed)) break;
			if (overflowed) {
				retval = util::B40CPerror(cudaErrorInvalidConfiguration, "Frontier queue overflow.  Please increase queue-sizing factor. ", __FILE__, __LINE__);
				break;
			}

		} while (0);

		return retval;
	}


    /**
 	 * Enacts a breadth-first-search on the specified graph problem.  Invokes
	 * a single grid kernel that itself steps over BFS iterations.
 	 *
 	 * @return cudaSuccess on success, error enumeration otherwise
 	 */
     template <typename CsrProblem>
 	cudaError_t EnactFusedSearch(
 		CsrProblem 						&csr_problem,
 		typename CsrProblem::VertexId 	src,
 		int 							max_grid_size = 0)
 	{
 		if (this->cuda_props.device_sm_version >= 200) {

 			// Fused-iteration two-phase tuning configuration
 			typedef two_phase::KernelPolicy<
 				typename CsrProblem::ProblemType,
 				200,					// CUDA_ARCH
 				INSTRUMENT, 			// INSTRUMENT
 				0, 						// SATURATION_QUIT

 				// Tunable parameters (generic)
 				8,						// MIN_CTA_OCCUPANCY
 				7,						// LOG_THREADS
 				util::io::ld::cg,		// QUEUE_READ_MODIFIER,
 				util::io::ld::NONE,		// COLUMN_READ_MODIFIER,
 				util::io::ld::cg,		// ROW_OFFSET_ALIGNED_READ_MODIFIER,
 				util::io::ld::NONE,		// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
 				util::io::st::cg,		// QUEUE_WRITE_MODIFIER,

 				// Tunable parameters (contract)
 				0,						// CONTRACT_LOG_LOAD_VEC_SIZE
 				2,						// CONTRACT_LOG_LOADS_PER_TILE
 				5,						// CONTRACT_LOG_RAKING_THREADS
 				false,					// CONTRACT_WORK_STEALING
				3,						// CONTRACT_END_BITMASK_CULL
				6, 						// CONTRACT_LOG_SCHEDULE_GRANULARITY

 				0,						// EXPAND_LOG_LOAD_VEC_SIZE
 				0,						// EXPAND_LOG_LOADS_PER_TILE
 				5,						// EXPAND_LOG_RAKING_THREADS
 				true,					// EXPAND_WORK_STEALING
 				32,						// EXPAND_WARP_GATHER_THRESHOLD
 				128 * 4, 				// EXPAND_CTA_GATHER_THRESHOLD,
 				6> 						// EXPAND_LOG_SCHEDULE_GRANULARITY
					KernelPolicy;

 			return EnactFusedSearch<KernelPolicy, CsrProblem>(
 				csr_problem, src, max_grid_size);

 		}

		printf("Not yet tuned for this architecture\n");
		return cudaErrorInvalidConfiguration;
 	}


    /**
	 * Enacts a breadth-first-search on the specified graph problem. Invokes
	 * new expansion and contraction grid kernels for each BFS iteration.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <
		typename ExpandPolicy,
		typename FilterPolicy,
		typename ContractPolicy,
		typename CsrProblem>
	cudaError_t EnactIterativeSearch(
		CsrProblem 						&csr_problem,
		typename CsrProblem::VertexId 	src,
		int 							max_grid_size = 0)
	{
		typedef typename CsrProblem::SizeT 			SizeT;
		typedef typename CsrProblem::VertexId 		VertexId;
		typedef typename CsrProblem::VisitedMask 	VisitedMask;

		cudaError_t retval = cudaSuccess;

		do {
			// Determine grid size(s)
			int expand_occupancy 			= ExpandPolicy::CTA_OCCUPANCY;
			int expand_grid_size 			= MaxGridSize(expand_occupancy, max_grid_size);

			int filter_occupancy			= FilterPolicy::CTA_OCCUPANCY;
			int filter_grid_size 			= MaxGridSize(filter_occupancy, max_grid_size);

			int contract_occupancy			= ContractPolicy::CTA_OCCUPANCY;
			int contract_grid_size 			= MaxGridSize(contract_occupancy, max_grid_size);

			if (DEBUG) {
				printf("BFS expand occupancy %d, level-grid size %d\n",
					expand_occupancy, expand_grid_size);
				printf("BFS filter occupancy %d, level-grid size %d\n",
					filter_occupancy, filter_grid_size);
				printf("BFS contract occupancy %d, level-grid size %d\n",
					contract_occupancy, contract_grid_size);
				printf("Iteration, Filter queue, Contraction queue, Expansion queue\n");
				printf("0, 0");
			}

			// Lazy initialization
			if (retval = Setup(csr_problem, expand_grid_size, filter_grid_size, contract_grid_size)) break;

			// Single-gpu graph slice
			typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

			SizeT queue_length;
			VertexId queue_index 		= 0;					// Work stealing/queue index
			int selector 				= 0;

			// Step through BFS iterations
			while (done[0] < 0) {

				//
				// Contraction
				//

				two_phase::contract_atomic::Kernel<ContractPolicy>
					<<<contract_grid_size, ContractPolicy::THREADS>>>(
						src,
						iteration[0],
						0,														// num_elements (unused: we obtain this from device-side counters instead)
						queue_index,											// queue counter index
						queue_index,											// steal counter index
						1,														// number of GPUs
						d_done,
						graph_slice->frontier_queues.d_keys[selector ^ 1],		// filtered edge frontier in
						graph_slice->frontier_queues.d_keys[selector],			// vertex frontier out
						graph_slice->frontier_queues.d_values[selector ^ 1],	// predecessor in
						graph_slice->d_labels,
						graph_slice->d_visited_mask,
						this->work_progress,
						graph_slice->frontier_elements[selector ^ 1],			// max filtered edge frontier vertices
						graph_slice->frontier_elements[selector],				// max vertex frontier vertices
						this->contract_kernel_stats);

				if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "contract_atomic::Kernel failed ", __FILE__, __LINE__))) break;
				cudaEventQuery(throttle_event);	// give host memory mapped visibility to GPU updates

				queue_index++;
				selector ^= 1;

				if (DEBUG) {
					if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
					printf(", %lld", (long long) queue_length);
				}
				if (INSTRUMENT) {
					if (retval = contract_kernel_stats.Accumulate(
						contract_grid_size,
						total_runtimes,
						total_lifetimes)) break;
				}

				// Throttle
				if (iteration[0] & 1) {
					if (retval = util::B40CPerror(cudaEventRecord(throttle_event),
						"EnactorTwoPhase cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
				} else {
					if (retval = util::B40CPerror(cudaEventSynchronize(throttle_event),
						"EnactorTwoPhase cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
				};

				// Check if done
				if (done[0] == 0) break;

				//
				// Expansion
				//

				two_phase::expand_atomic::Kernel<ExpandPolicy>
					<<<expand_grid_size, ExpandPolicy::THREADS>>>(
						queue_index,											// queue counter index
						queue_index,											// steal counter index
						1,														// number of GPUs
						d_done,
						graph_slice->frontier_queues.d_keys[selector ^ 1],		// vertex frontier in
						graph_slice->frontier_queues.d_keys[selector],			// edge frontier out
						graph_slice->frontier_queues.d_values[selector],		// predecessor out
						graph_slice->d_column_indices,
						graph_slice->d_row_offsets,
						this->work_progress,
						graph_slice->frontier_elements[selector ^ 1],			// max vertex frontier vertices
						graph_slice->frontier_elements[selector],				// max edge frontier vertices
						this->expand_kernel_stats);

				if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "expand_atomic::Kernel failed ", __FILE__, __LINE__))) break;
				cudaEventQuery(throttle_event);	// give host memory mapped visibility to GPU updates

				queue_index++;
				selector ^= 1;
				iteration[0]++;

				if (INSTRUMENT || DEBUG) {
					if (work_progress.GetQueueLength(queue_index, queue_length)) break;
					total_queued += queue_length;
					if (DEBUG) printf(", %lld", (long long) queue_length);
					if (INSTRUMENT) {
						if (retval = expand_kernel_stats.Accumulate(
							expand_grid_size,
							total_runtimes,
							total_lifetimes)) break;
					}
				}

				if (DEBUG) printf("\n%lld", (long long) iteration[0]);

				// Check if done
				if (done[0] == 0) break;

				//
				// Filter
				//

				two_phase::filter_atomic::Kernel<FilterPolicy>
					<<<filter_grid_size, FilterPolicy::THREADS>>>(
						queue_index,											// queue counter index
						queue_index,											// steal counter index
						d_done,
						graph_slice->frontier_queues.d_keys[selector ^ 1],		// edge frontier in
						graph_slice->frontier_queues.d_keys[selector],			// vertex frontier out
						graph_slice->frontier_queues.d_values[selector ^ 1],	// predecessor in
						graph_slice->frontier_queues.d_values[selector],		// predecessor out
						graph_slice->d_visited_mask,
						this->work_progress,
						graph_slice->frontier_elements[selector ^ 1],			// max edge frontier vertices
						graph_slice->frontier_elements[selector],				// max vertex frontier vertices
						this->filter_kernel_stats);

				if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "filter_atomic::Kernel failed ", __FILE__, __LINE__))) break;

				queue_index++;
				selector ^= 1;

				if (DEBUG) {
					if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
					printf(", %lld", (long long) queue_length);
				}
				if (INSTRUMENT) {
					if (retval = filter_kernel_stats.Accumulate(
						filter_grid_size,
						total_runtimes,
						total_lifetimes)) break;
				}

			}
			if (retval) break;

			// Check if any of the frontiers overflowed due to redundant expansion
			bool overflowed = false;
			if (retval = work_progress.CheckOverflow<SizeT>(overflowed)) break;
			if (overflowed) {
				retval = util::B40CPerror(cudaErrorInvalidConfiguration, "Frontier queue overflow.  Please increase queue-sizing factor. ", __FILE__, __LINE__);
				break;
			}

		} while(0);

		if (DEBUG) printf("\n");

		return retval;
	}


    /**
	 * Enacts a breadth-first-search on the specified graph problem. Invokes
	 * new expansion and contraction grid kernels for each BFS iteration.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <typename CsrProblem>
	cudaError_t EnactIterativeSearch(
		CsrProblem 						&csr_problem,
		typename CsrProblem::VertexId 	src,
		int 							max_grid_size = 0)
	{
		typedef typename CsrProblem::VertexId 		VertexId;
		typedef typename CsrProblem::SizeT 			SizeT;

		// GF100
		if (this->cuda_props.device_sm_version >= 200) {

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
				7>						// LOG_SCHEDULE_GRANULARITY
					ExpandPolicy;

			// Filter kernel config
			typedef two_phase::filter_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				200,					// CUDA_ARCH
				INSTRUMENT, 			// INSTRUMENT
				0, 						// SATURATION_QUIT
				8,						// CTA_OCCUPANCY
				7,						// LOG_THREADS
				1,						// LOG_LOAD_VEC_SIZE
				1,						// LOG_LOADS_PER_TILE
				5,						// LOG_RAKING_THREADS
				util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
				util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
				false,					// WORK_STEALING
				9> 						// LOG_SCHEDULE_GRANULARITY
					FilterPolicy;

			// Contraction kernel config
			typedef two_phase::contract_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				200,					// CUDA_ARCH
				INSTRUMENT, 			// INSTRUMENT
				0, 						// SATURATION_QUIT
				true, 					// DEQUEUE_PROBLEM_SIZE
				8,						// CTA_OCCUPANCY
				7,						// LOG_THREADS
				1,						// LOG_LOAD_VEC_SIZE
				0,						// LOG_LOADS_PER_TILE
				5,						// LOG_RAKING_THREADS
				util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
				util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
				false,					// WORK_STEALING
				0,						// END_BITMASK_CULL (never cull b/c filter does the bitmask culling)
				8> 						// LOG_SCHEDULE_GRANULARITY
					ContractPolicy;

			return EnactIterativeSearch<ExpandPolicy, FilterPolicy, ContractPolicy>(
				csr_problem, src, max_grid_size);
		}

/* Commented out to reduce compile time. Uncomment for GT200

		// GT200
		if (this->cuda_props.device_sm_version >= 130) {

			// Expansion kernel config
			typedef two_phase::expand_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				130,					// CUDA_ARCH
				INSTRUMENT, 			// INSTRUMENT
				1,						// CTA_OCCUPANCY
				8,						// LOG_THREADS
				0,						// LOG_LOAD_VEC_SIZE
				1, 						// LOG_LOADS_PER_TILE
				5,						// LOG_RAKING_THREADS
				util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
				util::io::ld::NONE,		// COLUMN_READ_MODIFIER,
				util::io::ld::NONE,		// ROW_OFFSET_ALIGNED_READ_MODIFIER,
				util::io::ld::NONE,		// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
				util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
				false,					// WORK_STEALING
				32,						// WARP_GATHER_THRESHOLD
				128 * 4, 				// CTA_GATHER_THRESHOLD,
				6>						// LOG_SCHEDULE_GRANULARITY
					ExpandPolicy;

			// Filter kernel config
			typedef two_phase::filter_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				130,					// CUDA_ARCH
				INSTRUMENT, 			// INSTRUMENT
				0, 						// SATURATION_QUIT
				8,						// CTA_OCCUPANCY
				7,						// LOG_THREADS
				1,						// LOG_LOAD_VEC_SIZE
				0,						// LOG_LOADS_PER_TILE
				5,						// LOG_RAKING_THREADS
				util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
				util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
				false,					// WORK_STEALING
				8> 						// LOG_SCHEDULE_GRANULARITY
					FilterPolicy;

			// Contraction kernel config
			typedef two_phase::contract_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				130,					// CUDA_ARCH
				INSTRUMENT, 			// INSTRUMENT
				0, 						// SATURATION_QUIT
				true, 					// DEQUEUE_PROBLEM_SIZE
				1,						// CTA_OCCUPANCY
				8,						// LOG_THREADS
				1,						// LOG_LOAD_VEC_SIZE
				1,						// LOG_LOADS_PER_TILE
				6,						// LOG_RAKING_THREADS
				util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
				util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
				false,					// WORK_STEALING
				0,						// END_BITMASK_CULL (never cull b/c filter does the bitmask culling)
				6>						// LOG_SCHEDULE_GRANULARITY
					ContractPolicy;

			return EnactIterativeSearch<ExpandPolicy, FilterPolicy, ContractPolicy>(
				csr_problem, src, max_grid_size);

		}
*/
		printf("Not yet tuned for this architecture\n");
		return cudaErrorInvalidDeviceFunction;
	}
};



} // namespace bfs
} // namespace graph
} // namespace b40c
