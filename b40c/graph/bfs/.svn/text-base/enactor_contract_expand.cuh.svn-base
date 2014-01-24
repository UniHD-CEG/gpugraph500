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
 * Contract+expand BFS enactor
 ******************************************************************************/

#pragma once


#include <b40c/util/global_barrier.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>

#include <b40c/graph/bfs/enactor_base.cuh>
#include <b40c/graph/bfs/problem_type.cuh>

#include <b40c/graph/bfs/contract_expand_atomic/kernel_policy.cuh>
#include <b40c/graph/bfs/contract_expand_atomic/kernel.cuh>


namespace b40c {
namespace graph {
namespace bfs {



/**
 * Contract+expand BFS enactor.
 *
 * For each BFS iteration, visited/duplicate vertices are culled from
 * the incoming edge-frontier in global memory.  The neighbor lists
 * of the remaining vertices are expanded to construct the outgoing
 * edge-frontier in global memory.
 */
template <bool INSTRUMENT>							// Whether or not to collect per-CTA clock-count statistics
class EnactorContractExpand : public EnactorBase
{

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

protected:

	/**
	 * Per-CTA clock-count and related statistics
	 */
	util::KernelRuntimeStatsLifetime 	kernel_stats;
	unsigned long long 					total_runtimes;			// Total time "worked" by each cta
	unsigned long long 					total_lifetimes;		// Total time elapsed by each cta
	unsigned long long 					total_queued;

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
		int grid_size)
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
			if (retval = global_barrier.Setup(grid_size)) break;

			// Make sure our runtime stats are initialized
			if (retval = kernel_stats.Setup(grid_size)) break;

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
					contract_expand_atomic::BitmaskTex<VisitedMask>::ref,
					graph_slice->d_visited_mask,
					bitmask_desc,
					bytes),
				"EnactorContractExpand cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;

			// Bind row-offsets texture
			cudaChannelFormatDesc row_offsets_desc = cudaCreateChannelDesc<SizeT>();
			if (retval = util::B40CPerror(cudaBindTexture(
					0,
					contract_expand_atomic::RowOffsetTex<SizeT>::ref,
					graph_slice->d_row_offsets,
					row_offsets_desc,
					(graph_slice->nodes + 1) * sizeof(SizeT)),
				"EnactorContractExpand cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

		} while (0);

		return retval;
	}


public:

	/**
	 * Constructor
	 */
	EnactorContractExpand(bool DEBUG = false) :
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
	virtual ~EnactorContractExpand()
	{
		if (iteration) {
			util::B40CPerror(cudaFreeHost((void *) iteration), "EnactorContractExpand cudaFreeHost iteration failed", __FILE__, __LINE__);
		}
		if (done) {
			util::B40CPerror(cudaFreeHost((void *) done),
					"EnactorContractExpand cudaFreeHost done failed", __FILE__, __LINE__);

			util::B40CPerror(cudaEventDestroy(throttle_event),
				"EnactorContractExpand cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
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
    	search_depth = iteration[0] - 1;

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
			if (DEBUG) printf("DEBUG: BFS occupancy %d, grid size %d\n", occupancy, grid_size); fflush(stdout);

			// Lazy initialization
			if (retval = Setup(csr_problem, grid_size)) break;

			// Single-gpu graph slice
			typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

			// Contract+expand kernel
			contract_expand_atomic::KernelGlobalBarrier<KernelPolicy>
					<<<grid_size, KernelPolicy::THREADS>>>(
				0,												// start iteration
				0,												// queue_index
				0,												// steal_index
				src,
				graph_slice->frontier_queues.d_keys[0],			// edge frontier (ping)
				graph_slice->frontier_queues.d_keys[1],			// edge frontier (pong)
				graph_slice->frontier_queues.d_values[0],		// in predecessors
				graph_slice->frontier_queues.d_values[1],		// out predecessors
				graph_slice->d_column_indices,
				graph_slice->d_row_offsets,
				graph_slice->d_labels,
				graph_slice->d_visited_mask,
				this->work_progress,
				graph_slice->frontier_elements[0],				// max frontier vertices (all queues should be the same size)
				this->global_barrier,
				this->kernel_stats,
				(VertexId *) d_iteration);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "contract_expand_atomic::KernelGlobalBarrier failed ", __FILE__, __LINE__))) break;

			if (INSTRUMENT) {
				// Get stats
				if (retval = kernel_stats.Accumulate(
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
		int								max_grid_size = 0)
	{
    	typedef typename CsrProblem::VertexId 	VertexId;
    	typedef typename CsrProblem::SizeT 		SizeT;

    	// GF100
		if (this->cuda_props.device_sm_version >= 200) {

			// Contract+expand kernel config
			typedef contract_expand_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				200,					// CUDA_ARCH
				INSTRUMENT, 			// INSTRUMENT
				0, 						// SATURATION_QUIT
				(sizeof(VertexId) > 4) ? 7 : 8,		// CTA_OCCUPANCY
				7,						// LOG_THREADS
				0,						// LOG_LOAD_VEC_SIZE
				0,						// LOG_LOADS_PER_TILE
				5,						// LOG_RAKING_THREADS
				util::io::ld::cg, 		// QUEUE_READ_MODIFIER,
				util::io::ld::NONE,		// COLUMN_READ_MODIFIER,
				util::io::ld::cg,		// ROW_OFFSET_ALIGNED_READ_MODIFIER,
				util::io::ld::NONE,		// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
				util::io::st::cg, 		// QUEUE_WRITE_MODIFIER
				false,					// WORK_STEALING
				32,						// WARP_GATHER_THRESHOLD
				128 * 4, 				// CTA_GATHER_THRESHOLD,
				0,						// END_BITMASK_CULL (never cull))
				6> 						// LOG_SCHEDULE_GRANULARITY
					KernelPolicy;

			return EnactFusedSearch<KernelPolicy>(csr_problem, src, max_grid_size);
		}

		// GT200
		if (this->cuda_props.device_sm_version >= 130) {

			// Contract+expand kernel config
			typedef contract_expand_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				130,					// CUDA_ARCH
				INSTRUMENT, 			// INSTRUMENT
				0, 						// SATURATION_QUIT
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
				0,						// END_BITMASK_CULL (never cull))
				6> 						// LOG_SCHEDULE_GRANULARITY
					KernelPolicy;

			return EnactFusedSearch<KernelPolicy>(csr_problem, src, max_grid_size);
		}

		printf("Not yet tuned for this architecture\n");
		return cudaErrorInvalidConfiguration;
	}


    /**
	 * Enacts a breadth-first-search on the specified graph problem. Invokes
	 * a new grid kernel for each BFS iteration.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <
    	typename KernelPolicy,
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

			// Determine grid size
			int occupancy = KernelPolicy::CTA_OCCUPANCY;
			int grid_size = MaxGridSize(occupancy, max_grid_size);
			if (DEBUG) printf("DEBUG: BFS occupancy %d, grid size %d\n", occupancy, grid_size); fflush(stdout);

			// Lazy initialization
			if (retval = Setup<KernelPolicy>(csr_problem, grid_size)) break;

			// Single-gpu graph slice
			typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

			SizeT queue_length;
			VertexId queue_index = 0;	// Work stealing/queue index

			if (INSTRUMENT && DEBUG) {
				printf("Queue\n");
				printf("1\n");
			}

			// Step through BFS iterations
			while (done[0] < 0) {

				int selector = queue_index & 1;

				// Contract+expand
				contract_expand_atomic::Kernel<KernelPolicy>
						<<<grid_size, KernelPolicy::THREADS>>>(
					iteration[0],											// iteration
					queue_index,											// queue_index
					queue_index,											// steal_index
					d_done,
					src,
					graph_slice->frontier_queues.d_keys[selector],			// in edge frontier
					graph_slice->frontier_queues.d_keys[selector ^ 1],		// out edge frontier
					graph_slice->frontier_queues.d_values[selector],		// in predecessors
					graph_slice->frontier_queues.d_values[selector ^ 1],	// out predecessors
					graph_slice->d_column_indices,
					graph_slice->d_row_offsets,
					graph_slice->d_labels,
					graph_slice->d_visited_mask,
					this->work_progress,
					graph_slice->frontier_elements[0],				// max frontier vertices (all queues should be the same size)
					this->kernel_stats);

				if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "contract_expand_atomic::Kernel failed ", __FILE__, __LINE__))) break;

				queue_index++;
				iteration[0]++;

				if (INSTRUMENT) {
					// Get queue length
					if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
					if (DEBUG) printf("%lld\n", (long long) queue_length);

					// Get stats (i.e., duty %)
					if (retval = kernel_stats.Accumulate(
						grid_size,
						total_runtimes,
						total_lifetimes)) break;
				}

				// Throttle
				if (iteration[0] & 1) {
					if (retval = util::B40CPerror(cudaEventRecord(throttle_event),
						"EnactorContractExpand cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
				} else {
					if (retval = util::B40CPerror(cudaEventSynchronize(throttle_event),
						"EnactorContractExpand cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
				};

			}
			if (retval) break;

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
	 * a new grid kernel for each BFS iteration.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <typename CsrProblem>
	cudaError_t EnactIterativeSearch(
		CsrProblem 						&csr_problem,
		typename CsrProblem::VertexId 	src,
		int								max_grid_size = 0)
	{
    	typedef typename CsrProblem::VertexId 	VertexId;
    	typedef typename CsrProblem::SizeT 		SizeT;

    	// GF100
		if (this->cuda_props.device_sm_version >= 200) {

			// Contract+expand kernel config
			typedef contract_expand_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				200,					// CUDA_ARCH
				INSTRUMENT, 			// INSTRUMENT
				0, 						// SATURATION_QUIT
				(sizeof(VertexId) > 4) ? 7 : 8,		// CTA_OCCUPANCY
				7,						// LOG_THREADS
				0,						// LOG_LOAD_VEC_SIZE
				0,						// LOG_LOADS_PER_TILE
				5,						// LOG_RAKING_THREADS
				util::io::ld::NONE, 	// QUEUE_READ_MODIFIER,
				util::io::ld::NONE,		// COLUMN_READ_MODIFIER,
				util::io::ld::cg,		// ROW_OFFSET_ALIGNED_READ_MODIFIER,
				util::io::ld::NONE,		// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
				util::io::st::NONE, 	// QUEUE_WRITE_MODIFIER
				false,					// WORK_STEALING
				32,						// WARP_GATHER_THRESHOLD
				128 * 4, 				// CTA_GATHER_THRESHOLD,
				128 * 3,				// END_BITMASK_CULL
				6> 						// LOG_SCHEDULE_GRANULARITY
					KernelPolicy;

			return EnactIterativeSearch<KernelPolicy>(csr_problem, src, max_grid_size);
		}

/* Commented out to reduce compile time. Uncomment for GT200

		// GT200
		if (this->cuda_props.device_sm_version >= 130) {

			// Contract+expand kernel config
			typedef contract_expand_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				130,					// CUDA_ARCH
				INSTRUMENT, 			// INSTRUMENT
				0, 						// SATURATION_QUIT
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
				256 * 3,				// END_BITMASK_CULL
				6> 						// LOG_SCHEDULE_GRANULARITY
					KernelPolicy;

			return EnactIterativeSearch<KernelPolicy>(csr_problem, src, max_grid_size);
		}
*/
		printf("Contract-expand not yet tuned for this architecture\n");
		return cudaErrorInvalidDeviceFunction;
	}

};



} // namespace bfs
} // namespace graph
} // namespace b40c
