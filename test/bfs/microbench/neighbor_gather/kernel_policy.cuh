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
 * "Metatype" for guiding BFS expansion granularity configuration
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/srts_grid.cuh>
#include <b40c/util/srts_soa_details.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace microbench {
namespace neighbor_gather {

/**
 * BFS atomic expansion kernel granularity configuration meta-type.  Parameterizations of this
 * type encapsulate our kernel-tuning parameters (i.e., they are reflected via
 * the static fields).
 *
 * Kernels can be specialized for problem-type, SM-version, etc. by parameterizing
 * them with different performance-tuned parameterizations of this type.  By
 * incorporating this type into the kernel code itself, we guide the compiler in
 * expanding/unrolling the kernel code for specific architectures and problem
 * types.
 */
template <
	// ProblemType type parameters
	typename _ProblemType,

	// Machine parameters
	int CUDA_ARCH,

	// Behavioral control parameters
	bool _BENCHMARK,					// Whether or not this is a microbenchmark configuration
	bool _INSTRUMENT,					// Whether or not we want instrumentation logic generated
	int _SATURATION_QUIT,				// If positive, signal that we're done with two-phase iterations if problem size drops below (SATURATION_QUIT * grid_size * TILE_SIZE)
	bool _DEQUEUE_PROBLEM_SIZE,			// Whether we obtain problem size from device-side queue counters (true), or use the formal parameter (false)

	// Tunable parameters
	int _MIN_CTA_OCCUPANCY,
	int _LOG_THREADS,
	int _LOG_LOAD_VEC_SIZE,
	int _LOG_LOADS_PER_TILE,
	int _LOG_RAKING_THREADS,
	util::io::ld::CacheModifier _QUEUE_READ_MODIFIER,
	util::io::ld::CacheModifier _COLUMN_READ_MODIFIER,
	util::io::ld::CacheModifier _ROW_OFFSET_ALIGNED_READ_MODIFIER,
	util::io::ld::CacheModifier _ROW_OFFSET_UNALIGNED_READ_MODIFIER,
	util::io::st::CacheModifier _QUEUE_WRITE_MODIFIER,
	bool _WORK_STEALING,
	int _WARP_GATHER_THRESHOLD,
	int _CTA_GATHER_THRESHOLD,
	int _LOG_SCHEDULE_GRANULARITY>

struct KernelPolicy : _ProblemType
{
	typedef _ProblemType 					ProblemType;
	typedef typename ProblemType::VertexId 	VertexId;
	typedef typename ProblemType::SizeT 	SizeT;

	static const util::io::ld::CacheModifier QUEUE_READ_MODIFIER 					= _QUEUE_READ_MODIFIER;
	static const util::io::ld::CacheModifier COLUMN_READ_MODIFIER 					= _COLUMN_READ_MODIFIER;
	static const util::io::ld::CacheModifier ROW_OFFSET_ALIGNED_READ_MODIFIER 		= _ROW_OFFSET_ALIGNED_READ_MODIFIER;
	static const util::io::ld::CacheModifier ROW_OFFSET_UNALIGNED_READ_MODIFIER 	= _ROW_OFFSET_UNALIGNED_READ_MODIFIER;
	static const util::io::st::CacheModifier QUEUE_WRITE_MODIFIER 					= _QUEUE_WRITE_MODIFIER;

	enum {

		INSTRUMENT						= _INSTRUMENT,
		SATURATION_QUIT					= _SATURATION_QUIT,
		DEQUEUE_PROBLEM_SIZE			= _DEQUEUE_PROBLEM_SIZE,

		LOG_THREADS 					= _LOG_THREADS,
		THREADS							= 1 << LOG_THREADS,

		LOG_LOAD_VEC_SIZE  				= _LOG_LOAD_VEC_SIZE,
		LOAD_VEC_SIZE					= 1 << LOG_LOAD_VEC_SIZE,

		LOG_LOADS_PER_TILE 				= _LOG_LOADS_PER_TILE,
		LOADS_PER_TILE					= 1 << LOG_LOADS_PER_TILE,

		LOG_LOAD_STRIDE					= LOG_THREADS + LOG_LOAD_VEC_SIZE,
		LOAD_STRIDE						= 1 << LOG_LOAD_STRIDE,

		LOG_RAKING_THREADS				= _LOG_RAKING_THREADS,
		RAKING_THREADS					= 1 << LOG_RAKING_THREADS,

		LOG_WARPS						= LOG_THREADS - B40C_LOG_WARP_THREADS(CUDA_ARCH),
		WARPS							= 1 << LOG_WARPS,

		LOG_TILE_ELEMENTS_PER_THREAD	= LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE,
		TILE_ELEMENTS_PER_THREAD		= 1 << LOG_TILE_ELEMENTS_PER_THREAD,

		LOG_TILE_ELEMENTS 				= LOG_TILE_ELEMENTS_PER_THREAD + LOG_THREADS,
		TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS,

		LOG_SCHEDULE_GRANULARITY		= _LOG_SCHEDULE_GRANULARITY,
		SCHEDULE_GRANULARITY			= 1 << LOG_SCHEDULE_GRANULARITY,

		WORK_STEALING					= _WORK_STEALING,
		WARP_GATHER_THRESHOLD			= _WARP_GATHER_THRESHOLD,
		CTA_GATHER_THRESHOLD			= _CTA_GATHER_THRESHOLD,

		BENCHMARK						= _BENCHMARK,
	};

	// Raking grid type for coarse
	typedef util::RakingGrid<
		CUDA_ARCH,
		SizeT,									// Partial type
		LOG_THREADS,							// Depositing threads (the CTA size)
		LOG_LOADS_PER_TILE,						// Lanes (the number of loads)
		LOG_RAKING_THREADS,						// Raking threads
		true>									// There are prefix dependences between lanes
			CoarseGrid;

	// Raking grid type for fine
	typedef util::RakingGrid<
		CUDA_ARCH,
		SizeT,									// Partial type
		LOG_THREADS,							// Depositing threads (the CTA size)
		LOG_LOADS_PER_TILE,						// Lanes (the number of loads)
		LOG_RAKING_THREADS,						// Raking threads
		true>									// There are prefix dependences between lanes
			FineGrid;


	// Tuple of partial-flag type
	typedef util::Tuple<SizeT, SizeT> TileTuple;


	/**
	 * SOA scan operator
	 */
	struct SoaScanOp
	{
		enum {
			IDENTITY_STRIDES = true,			// There is an "identity" region of warpscan storage exists for strides to index into
		};

		// SOA scan operator
		__device__ __forceinline__ TileTuple operator()(
			const TileTuple &first,
			const TileTuple &second)
		{
			return TileTuple(first.t0 + second.t0, first.t1 + second.t1);
		}

		// SOA identity operator
		__device__ __forceinline__ TileTuple operator()()
		{
			return TileTuple(0,0);
		}
	};


	// Tuple type of raking grid types
	typedef util::Tuple<
		CoarseGrid,
		FineGrid> RakingGridTuple;


	// Operational details type for raking grid type
	typedef util::RakingSoaDetails<
		TileTuple,
		RakingGridTuple> RakingSoaDetails;


	/**
	 * Shared memory structure
	 */
	struct SmemStorage
	{
		// Persistent shared state for the CTA
		struct State {
			// Type describing four shared memory channels per warp for intra-warp communication
			typedef SizeT 						WarpComm[WARPS][4];

			// Shared work-processing limits
			util::CtaWorkDistribution<SizeT>	work_decomposition;

			// Shared memory channels for intra-warp communication
			volatile WarpComm					warp_comm;
			int 								cta_comm;

			// Storage for scanning local compact-expand ranks
			SizeT 								coarse_warpscan[2][B40C_WARP_THREADS(CUDA_ARCH)];
			SizeT 								fine_warpscan[2][B40C_WARP_THREADS(CUDA_ARCH)];

			// Enqueue offset for neighbors of the current tile
			SizeT								fine_enqueue_offset;
			SizeT								coarse_enqueue_offset;

		} state;

		enum {
			// Amount of storage we can use for hashing scratch space under target occupancy
			MAX_SCRATCH_BYTES_PER_CTA		= (B40C_SMEM_BYTES(CUDA_ARCH) / _MIN_CTA_OCCUPANCY)
												- sizeof(State)
												- 72,

			SCRATCH_ELEMENT_SIZE 			= (ProblemType::MARK_PARENTS) ?
													sizeof(SizeT) + sizeof(VertexId) :			// Need both gather offset and predecessor
													sizeof(SizeT),								// Just gather offset

			OFFSET_ELEMENTS					= MAX_SCRATCH_BYTES_PER_CTA / SCRATCH_ELEMENT_SIZE,
			PARENT_ELEMENTS					= (ProblemType::MARK_PARENTS) ?  OFFSET_ELEMENTS : 0,
		};

		volatile VertexId gathered;

		union {
			// Raking elements
			struct {
				SizeT 						coarse_raking_elements[CoarseGrid::TOTAL_RAKING_ELEMENTS];
				SizeT 						fine_raking_elements[FineGrid::TOTAL_RAKING_ELEMENTS];
			};

			// Scratch elements
			struct {
				volatile SizeT 				offset_scratch[OFFSET_ELEMENTS];
				volatile VertexId 			predecessor_scratch[PARENT_ELEMENTS];
			};
		};
	};

	enum {
		THREAD_OCCUPANCY				= B40C_SM_THREADS(CUDA_ARCH) >> LOG_THREADS,
		SMEM_OCCUPANCY					= B40C_SMEM_BYTES(CUDA_ARCH) / sizeof(SmemStorage),
		CTA_OCCUPANCY  					= B40C_MIN(_MIN_CTA_OCCUPANCY, B40C_MIN(B40C_SM_CTAS(CUDA_ARCH), B40C_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY))),

		VALID							= (CTA_OCCUPANCY > 0),
	};
};


} // namespace neighbor_gather
} // namespace microbench
} // namespace bfs
} // namespace graph
} // namespace b40c

